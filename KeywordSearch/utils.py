from typing import Iterable

import pickle
import gc
import re
import os
import glob
import traceback
import concurrent.futures
from collections import defaultdict

import numpy as np
import scipy.sparse

try:
    from tqdm import tqdm
    USE_TQDM = True
except:
    USE_TQDM = False

def construct_bool_table(index: Iterable[dict], all_tokens, valid_books, save_path):
    table = scipy.sparse.dok_matrix((len(all_tokens), max(valid_books) + 1), dtype=np.bool_)
    length = len(all_tokens)
    tqdm_iter = enumerate(index[:length])
    if USE_TQDM:
        tqdm_iter = tqdm(tqdm_iter, total=length)
    for token_id, token_dict in tqdm_iter:
        if token_dict:
            table[token_id, tuple(token_dict.keys())] = True
    gc.collect()
    table = table.tocsr()
    if save_path is None:
        return table
    else:
        gc.collect()
        scipy.sparse.save_npz("lookup_table.npz", table, compressed=True)
        return table

# import h5py

# def save_inv_index_HDF5(filename: str, index: Iterable[dict], **kwargs):
#     with h5py.File(filename, 'w') as f:
#         for i, entry in enumerate(index):
#             group = f.create_group(str(i))
#             for book_id, occurrences in entry.items():
#                 group.create_dataset(str(book_id), data=occurrences, **kwargs)

def cast2intarr(x: Iterable, delta_encode: bool=True, *args, **kwargs):
    delta = np.min(x) if delta_encode else 0
    data_range = np.max(x) - delta
    bit_depth = np.uint8 if data_range < 256 else np.uint16 if data_range < 65536 else np.uint32 if data_range < 4294967296 else np.uint64
    return np.array(x - delta, *args, dtype=bit_depth, **kwargs), delta

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

def pickle_save(filename: str, obj: object, unsafe: bool=False):
    if unsafe:
        gc.disable()
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if unsafe:
        gc.enable()

def save_in_batches(batch_size: int, index_type: str, index: Iterable[dict], prefix: str, 
                    index_size: int=None, unsafe_pickle: bool=False):
    gc.collect()
    if batch_size <= 0:
        pickle_save(f"index/{prefix}_{index_type}_index.pkl", index, unsafe_pickle)
    else:
        if index_size == None:
            index_size = len(index)
        batches = index_size // batch_size
        filename = f"index/{prefix}_{index_type}_%0{len(str(batches))}d.pkl"
        end = 0
        i = -1

        for i in range(batches):
            start = i * batch_size
            end = min(start + batch_size, index_size)
            pickle_save(filename %(i), index[start:end], unsafe_pickle)
            gc.collect()
        
        pickle_save(filename %(i+1), index[end:], unsafe_pickle)

def fetch_sizes(parts):
    sizes = dict()
    for part in parts:
        with open(part, "rb") as f:
            tmp = pickle.load(f)
        sizes[part] = len(tmp)
        gc.collect()
    return sizes

def measure_sizes(dir: str="index", naming_rule: str=r"part([0-9]+)_inverted_([0-9]+).pkl"):
    naming_regex = re.compile(naming_rule)
    index_segments = glob.glob(naming_rule.replace("([0-9]+)", '*'), root_dir=dir)
    lookup_table = defaultdict(lambda : list())
    for segment_name in index_segments:
        match = naming_regex.fullmatch(segment_name)
        if match:
            segment_index = match.group(2)
            lookup_table[segment_index].append(os.path.join(dir, segment_name))

    print(f"{len(lookup_table)} segments to merge")
    failed_jobs = []
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
    
class ZeroDict(dict):
    def __missing__(self, _):
        return 0