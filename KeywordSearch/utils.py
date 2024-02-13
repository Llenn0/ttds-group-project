from typing import Iterable
import pickle

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

def save_in_batches(batch_size: int, index_type: str, index: Iterable[dict], index_size: int=None):
    if batch_size <= 0:
        with open(f"index/{index_type}_index.pkl", "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if index_size == None:
            index_size = len(index)
        batches = index_size // batch_size
        filename = f"index/{index_type}_%0{len(str(batches))}d.pkl"
        end = 0
        i = -1

        for i in range(batches):
            start = i * batch_size
            end = min(start + batch_size, index_size)
            with open(filename %(i), "wb") as f:
                pickle.dump(index[start:end], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(filename %(i+1), "wb") as f:
            pickle.dump(index[end:], f, protocol=pickle.HIGHEST_PROTOCOL)