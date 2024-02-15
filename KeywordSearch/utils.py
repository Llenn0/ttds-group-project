from typing import Iterable
import pickle
import gc

import h5py
import numpy as np

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

def save_inv_index_HDF5(filename: str, index: Iterable[dict], **kwargs):
    with h5py.File(filename, 'w') as f:
        for i, entry in enumerate(index):
            group = f.create_group(str(i))
            for book_id, occurrences in entry.items():
                group.create_dataset(str(book_id), data=occurrences, **kwargs)

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
        
        pickle_save(filename %(i+1), index[start:end], unsafe_pickle)
