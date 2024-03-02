from typing import Iterable

import gc
import re
import os
import glob
import json
import pickle
import requests
import traceback
import concurrent.futures
from sys import platform
from collections import defaultdict

import numpy as np
import scipy.sparse

from KeywordSearch.loader import LOG_PATH, LOOKUP_TABLE_PATH, VALID_BOOKS_PATH, raw_dir, index_dir, tqdm
from KeywordSearch import loader

regex_extract_book_id = re.compile(r"\/ebooks\/([0-9]+)")
ebook_cached = "https://www.gutenberg.org/cache/epub/%d/pg%d.txt"
ebook_normal = "https://www.gutenberg.org/ebooks/%d.txt.utf-8"
is_win32 = platform.casefold() == "win32"

def fetch_update() -> bool:
    r = requests.get("https://www.gutenberg.org/browse/recent/last30")
    if r.status_code == 200:
        latest_existing = max(loader.metadata.keys())
        new_books = (int(book_id) for book_id in regex_extract_book_id.findall(r.text))
        new_books = (book_id for book_id in new_books if 
                     (book_id > latest_existing) and (book_id not in loader.processed_books))
        pass
        return True
    return False

def get_new_book(book_id: int) -> int:
    # 404 = non-existent
    r = requests.get(ebook_normal %(book_id))
    if r.status_code == 200:
        content = r.content.decode(encoding="utf-8", errors="ignore")
        if is_win32:
            content.replace("\r\n", '\n')
        if content[0] == '\ufeff' and content:
            content = content[1:]
        
        with open(os.path.join(raw_dir, f"PG{book_id}_raw.txt"), encoding="utf-8") as f:
            f.write(content.replace("\r\n", '\n') if is_win32 else content)

    return r.status_code

def construct_bool_table(index: Iterable[dict], all_tokens: tuple[str], valid_books: Iterable[int]=None):
    if valid_books is None:
        with open(VALID_BOOKS_PATH, "rb") as f:
            valid_books = tuple(pickle.load(f))
    table = scipy.sparse.dok_matrix((len(all_tokens), max(valid_books) + 1), dtype=np.bool_)
    del valid_books
    length = len(all_tokens)
    for token_id, token_dict in tqdm(enumerate(index[:length]), desc="Constructing lookup table", total=length):
        if token_dict:
            table[token_id, tuple(token_dict.keys())] = True
    gc.collect()
    table = table.tocsr()
    print("Finished converting lookup table to CSR format", flush=True)
    gc.collect()
    scipy.sparse.save_npz(LOOKUP_TABLE_PATH, table, compressed=True)
    print(f"Finished saving lookup table to {LOOKUP_TABLE_PATH}", flush=True)
    return table

# import h5py

# def save_inv_index_HDF5(filename: str, index: Iterable[dict], **kwargs):
#     with h5py.File(filename, 'w') as f:
#         for i, entry in enumerate(index):
#             group = f.create_group(str(i))
#             for book_id, occurrences in entry.items():
#                 group.create_dataset(str(book_id), data=occurrences, **kwargs)

def cast2intarr(x: Iterable, delta_encode: bool=True, *args, **kwargs):
    if len(x):
        delta = np.min(x) if delta_encode else 0
        data_range = np.max(x) - delta
        bit_depth = np.uint8 if data_range < 256 else np.uint16 if data_range < 65536 else np.uint32 if data_range < 4294967296 else np.uint64
        return np.array(x - delta, *args, dtype=bit_depth, **kwargs), delta
    else:
        return np.array(x, dtype=np.uint8), 0

class lenzip:
    def __init__(self, *iters) -> None:
        self.data = zip(*iters)
        self.length = np.min([len(iter) for iter in iters])
    def __len__(self):
        return self.length
    def __iter__(self):
        return self.data
    def __next__(self):
        return self.data.__next__()

class deltazip:
    def __init__(self, iters: Iterable[np.ndarray], deltas: Iterable[int]) -> None:
        iter_len = [iter.shape[0] for iter in iters]
        for i, (iter, delta, arr_len) in enumerate(zip(iters, deltas, iter_len)):
            if arr_len:
                data_range = int(iter.max()) + delta
                if data_range > np.iinfo(iter.dtype).max:
                    iters[i] = iter.astype(np.uint8 if data_range < 256 else 
                                        np.uint16 if data_range < 65536 else 
                                        np.uint32 if data_range < 4294967296 else 
                                        np.uint64, copy=True) + delta
                    del iter
                else:
                    iters[i] = iter + delta
        self.data = zip(*iters)
        self.length = np.min(iter_len)
    def __len__(self):
        return self.length
    def __iter__(self):
        return self.data
    def __next__(self):
        return self.data.__next__()

class TTDSIndexEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_pickle(obj: object, filename: str, unsafe: bool=False):
    """Disabling garbage collection is probably too dangerous for most use cases"""
    if unsafe:
        gc.disable()
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if unsafe:
        gc.enable()

def save_json(x: Iterable[dict[int, np.ndarray]], fname):
    fname = fname.replace(".pkl", ".json")
    with open(fname, 'w', encoding="ascii") as f:
        json.dump(x, f, skipkeys=True, ensure_ascii=True, check_circular=False, indent=None,
                  separators=(',', ':'), cls=TTDSIndexEncoder)

def save_in_batches(batch_size: int, index_type: str, index: Iterable[dict], prefix: str, 
                    index_size: int=None, unsafe_pickle: bool=False, use_json: bool=False, 
                    pool: concurrent.futures.ProcessPoolExecutor=None):
    gc.collect()
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    if batch_size <= 0:
        if use_json:
            save_json(index, os.path.join(index_dir, f"{prefix}_{index_type}_index.json"))
        else:
            save_pickle(index, os.path.join(index_dir, f"{prefix}_{index_type}_index.pkl"), unsafe_pickle)
    else:
        if index_size == None:
            index_size = len(index)
        num_batches = index_size // batch_size
        if index_size % batch_size:
            num_batches += 1
        filename = os.path.join(index_dir, f"{prefix}_{index_type}_%0{len(str(num_batches))}d.pkl")
        end = 0
        i = -1

        save_func = save_json if use_json else save_pickle
        save_type = "json" if use_json else "pickle"

        if pool is None:
            for i in tqdm(range(num_batches), desc=f"Dumping ({save_type})", total=num_batches):
                start = i * batch_size
                end = min(start + batch_size, index_size)
                save_func(index[start:end], filename %(i))
        else:
            jobs = dict()
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, index_size)
                jobs[pool.submit(save_func, index[start:end], filename %(i))] = i
            
            for job in tqdm(concurrent.futures.as_completed(jobs), total=num_batches,
                            desc=f"Dumping ({save_type})"):
                segment_index = jobs[job]
                try:
                    job.result()
                except Exception as e:
                    with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                        f.write(f"Dumping {save_type} failure at segment {segment_index}:\n{''.join(traceback.format_exception(e))}\n")
            
            del jobs
        gc.collect()

def fetch_sizes(parts):
    sizes = dict()
    for part in parts:
        with open(part, "rb") as f:
            tmp = pickle.load(f)
        sizes[part] = len(tmp)
        gc.collect()
    return sizes

def measure_sizes(dir: str=index_dir, naming_rule: str=r"part([0-9]+)_inverted_([0-9]+).pkl"):
    naming_regex = re.compile(naming_rule)
    index_segments = glob.glob(naming_rule.replace("([0-9]+)", '*'), root_dir=dir)
    lookup_table = defaultdict(lambda : list())
    for segment_name in index_segments:
        match = naming_regex.fullmatch(segment_name)
        if match:
            segment_index = match.group(2)
            lookup_table[segment_index].append(os.path.join(dir, segment_name))

    print(f"{len(lookup_table)} segments to merge")

    complete_counter = 0
    results = dict()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        jobs = {pool.submit(fetch_sizes, segments) : segment_index
                for segment_index, segments in lookup_table.items()}
        for job in concurrent.futures.as_completed(jobs):
            segment_index = jobs[job]
            results[segment_index] = job.result()
            complete_counter += 1
            print(f"Finished measuring size for {complete_counter} segments...", end="\r", flush=True)
    print("\nAll done")
    return results

def convert_to_json(path: str):
    json_path = path.replace(".pkl", ".json")
    with open(path, "rb") as f:
        segment = pickle.load(f)
    gc.collect()
    save_json(segment, json_path)

def pickle_to_json(naming_pattern: str=r"([0-9]+)_merged.pkl", dir_: str=index_dir):
    regex_naming = re.compile(naming_pattern)
    glob_pattern = naming_pattern.replace("([0-9]+)", '*')
    segments = [[int(regex_naming.fullmatch(filename).group(1)), filename] 
                for filename in glob.glob(glob_pattern, root_dir=dir_)]
    segments.sort(key=lambda x: x[0])

    print(f"{len(segments)} segments to load")
    complete_counter = 0
    with concurrent.futures.ProcessPoolExecutor() as pool:
        jobs = {pool.submit(convert_to_json, os.path.join(dir_, filename)) : segment_index
                for segment_index, filename in segments}
        for job in tqdm(concurrent.futures.as_completed(jobs), total=len(jobs), desc="Saving json segments"):
            segment_index = jobs[job]
            try:
                job.result()
            except Exception as e:
                with open(LOG_PATH, 'a', encoding="UTF-8") as f:
                    f.write(f"Save json index failure at segment {segment_index}:\n{''.join(traceback.format_exception(e))}\n")
            complete_counter += 1
            if (complete_counter % 5 is 0):
                gc.collect()
    
class ZeroDict(dict):
    def __missing__(self, _):
        return 0

def dict2arr(data: dict[int, int|float], dtype: np.dtype) -> np.ndarray:
    keys = list(data.keys())
    arr = np.zeros(max(keys) + 1, dtype=np.float32)
    arr[keys] = list(data.values())
    return arr