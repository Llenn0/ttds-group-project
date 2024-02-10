from typing import Iterable

import numpy as np

def cast2intarr(x: Iterable, delta_encode: bool=True, *args, **kwargs):
    tmp = np.array(x)
    delta = np.min(x) if delta_encode else 0
    data_range = np.max(x) - delta
    bit_depth = np.uint8 if data_range < 256 else np.uint16 if data_range < 65536 else np.uint32 if data_range < 4294967296 else np.uint64
    return np.array(x - delta, *args, dtype=bit_depth, **kwargs), delta