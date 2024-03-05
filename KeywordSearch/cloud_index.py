import json
import time
import traceback
import gzip

import numpy as np
from google.cloud import firestore

from KeywordSearch.utils import TTDSIndexEncoder
from KeywordSearch.loader import LOG_PATH

SIZE_LIMIT = (1024 * 1024) - 10 # 10 is an arbitrary safety margin, equals to size of 1 int + 1 ascii character

class DocSlice:
    def __init__(self, token_id: int|str, slice_id: int, collection: str="index", doc_padding: int=32):
        """
        Size Calculation (in Bytes):
        - ASCII string: length + 1
        - Integer: 8
        - Document name: StringSize(CollectionName) + StringSize(DocumentName) + 16
        - DocumentPaddingSize: 32 (default)
        - Document: StringSize(AllFields) + 8 * AllIntegers + StringSize(AllStrings) + DocumentPaddingSize

        Document Structure:
        {
            "h" : Array[Integer]    # Header:  Book IDs
            "d" : Array[String]     # Data:    The token's Positions in each Book
            "b" : Array[Bytes]      # Bytes: Whether the data array has been compressed
        }
        """
        self.doc_name = f"{token_id}_{slice_id}"
        self.size = len(collection + self.doc_name) + 22 # 16 for doc name, 2 for header & data field name, 4 for string terminators
        self.size += doc_padding
        self.header = []
        self.data = []
        self.is_compressed = False

    def rm_slice_id(self):
        assert '_' in self.doc_name
        self.doc_name, slice_id_str = self.doc_name.split('_')
        self.size -= (len(slice_id_str) + 1)

    def try_add(self, key: int, data_str: str|bytes):
        additional_size = len(data_str) + 9 # 8 for integer key, 1 for string terminator
        if self.size + additional_size < SIZE_LIMIT:
            self.size += additional_size
            self.header.append(key)
            self.data.append(data_str)
            return True
        return False
    
    def try_add_compressed(self, key: int, data_str: str):
        data_byte = gzip.compress(data_str.encode("ascii"))
        self.is_compressed = True
        return self.try_add(key, data_byte)
    
    def to_dict(self):
        if self.is_compressed:
            return {'h' : self.header, 'b' : self.data}
        else:
            return {'h' : self.header, 'd' : self.data}

class DocIndex:
    def __init__(self, token_id: int, slices: list[DocSlice]):
        self.doc_name = str(token_id)
        self.header = [slice.header[0] for slice in slices]

    def to_dict(self):
        return {'h' : self.header}

def upload_firestore(content: DocSlice|DocIndex, index_api: firestore.CollectionReference, failures: int=0):
    try:
        index_api.document(content.doc_name).set(content.to_dict())
    except Exception as e:
        with open(LOG_PATH, 'a', encoding="UTF-8") as f:
            f.write(f"Upload firestore failure at {content.doc_name}:\n{''.join(traceback.format_exception(e))}\n")
        if failures < 50:
            time.sleep(0.1)
            upload_firestore(content, index_api, failures + 1)
        else:
            raise e

def prepare_tokendict_for_upload(token_dict: dict[str, np.ndarray[int]], token_id: int|str) -> list[DocSlice|DocIndex]:
    to_upload = []
    slice_id = 0
    slice = DocSlice(token_id, slice_id)
    all_slices = [slice]
    is_small_upload = True
    for k, v in sorted(token_dict.items()):
        content_str = json.dumps(v.tolist(), skipkeys=True, ensure_ascii=True, check_circular=False, indent=None, separators=(',', ':'))
        if not slice.try_add(k, content_str):
            is_small_upload = False
            to_upload.append(slice)
            # Exception handing for the first entry being > 1 MiB
            if not slice.data:
                slice.try_add(-1, "[]")
            slice_id += 1
            slice = DocSlice(token_id, slice_id)
            all_slices.append(slice)
            if not slice.try_add(k, content_str):
                with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                    f.write(f"WARNING: Entry at token {token_id}, book {k} too large: {len(content_str)}, try compression\n")
                if slice.try_add_compressed(k, content_str):
                    to_upload.append(slice)
                    slice_id += 1
                    slice = DocSlice(token_id, slice_id)
                    all_slices.append(slice)
                else:
                    with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                        f.write(f"ERROR: Entry at token {token_id}, book {k} can't fit after compression\n")
    if is_small_upload:
        slice.rm_slice_id()
    else:
        to_upload.append(DocIndex(token_id, all_slices))
    to_upload.append(slice)
    return to_upload

class CloudIndexDict(dict):
    def __init__(self, index_api: firestore.CollectionReference):
        self.index_api = index_api
        super().__init__(self)
    def __missing__(self, key):
        if isinstance(key, int):
            self[key] = CloudDoc(self.index_api, key)
            return self[key]

class CloudDoc:
    def __init__(self, index_api: firestore.CollectionReference, token_id: int, cloud_dict: dict=None):
        self.index_api = index_api
        self.doc_name = str(token_id)
        if cloud_dict is None:
            cloud_dict = self.index_api.document(self.doc_name).get().to_dict()
        self.is_segmented = 'd' not in cloud_dict
        self.header = np.array(cloud_dict['h'], dtype=np.uint32)
        self.accessed_slice = set()
        self.positions_cache = dict()
        if self.is_segmented:
            self.known_keys = {first_book_id : i for i, first_book_id in enumerate(cloud_dict['h'])}
            self.data = None
        else:
            self.known_keys = set(cloud_dict['h'])
            self.data = cloud_dict['d']

    def __contains__(self, i: int) -> bool:
        if i in self.known_keys:
            return True
        
        if self.is_segmented:
            best_guess = self.header.searchsorted(i)
            if best_guess in self.accessed_slice:
                return False
            guessed_slice = self.index_api.document(self.doc_name + '_' + str(best_guess)).get().to_dict()
            contained_entries = set(guessed_slice['h'])
            self.known_keys.update(dict.fromkeys(guessed_slice['h'], guessed_slice))
            self.accessed_slice.add(best_guess)
            if i in contained_entries:
                return True
        
        return False
    
    def __getitem__(self, i: int):
        """For performance consideration, assumes the key i always exist"""
        if i in self.positions_cache:
            return self.positions_cache[i]

        if self.is_segmented:
            slice_id = self.known_keys[i] if i in self.known_keys else np.searchsorted(self.header, i)
            fetched_slice = self.index_api.document(self.doc_name + '_' + str(slice_id)).get().to_dict()
            contained_entries = fetched_slice['h']
            slice_index = contained_entries.index(i)
            if 'b' in fetched_slice:
                arr = np.array(json.loads(gzip.decompress(fetched_slice['d'][0])), dtype=np.uint64)
            else:
                arr = np.array(json.loads(fetched_slice['d'][slice_index]), dtype=np.uint64) # uint64 is probably faster?
        else:
            arr = np.array(json.loads(self.data[self.header.searchsorted(i)]), dtype=np.uint64) # uint64 is probably faster?
        
        self.positions_cache[i] = arr # cache result
        return arr

class CloudIndex:
    def __init__(self, collection_: firestore.CollectionReference) -> None:
        self.index_api = collection_
        self.cache = CloudIndexDict(self.index_api)
    def __getitem__(self, i: int|str):
        return self.cache[i]