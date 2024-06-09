# -*- coding: utf-8 -*-
import cudf
import math
from numba import cuda
from cudf.core import Series
from pandas_ta.utils import get_offset, verify_series
import numpy as np

@cuda.jit
def calculate_ebsw(close, length, bars, offset, result):
    i = cuda.grid(1)
    m = len(close)
    if i >= m - length + 1:
        return
    alpha1 = math.sin(math.pi * 360 / length)
    HP = 0
    lastHP = 0
    lastClose = close[i + length - 1]
    FilterHist = cuda.local.array((2,), dtype=float)
    FilterHist[0] = 0
    FilterHist[1] = 0
    for j in range(length - 1, m):
        alpha1 = (1 - math.sin(math.pi * 360 / length)) / math.cos(math.pi * 360 / length)
        HP = 0.5 * (1 + alpha1) * (close[j] - lastClose) + alpha1 * lastHP
        a1 = math.exp(-math.sqrt(2) * math.pi / bars)
        b1 = 2 * a1 * math.cos(math.sqrt(2) * math.pi * 180 / bars)
        c2 = b1
        c3 = -1 * a1 * a1
        c1 = 1 - c2 - c3
        Filt = c1 * (HP + lastHP) / 2 + c2 * FilterHist[1] + c3 * FilterHist[0]
        Wave = (Filt + FilterHist[1] + FilterHist[0]) / 3
        Pwr = (Filt * Filt + FilterHist[1] * FilterHist[1] + FilterHist[0] * FilterHist[0]) / 3
        Wave = Wave / math.sqrt(Pwr)
        FilterHist[0] = FilterHist[1]
        FilterHist[1] = Filt
        lastHP = HP
        lastClose = close[j]
        result[i + length - 1] = Wave

def ebsw(close, length=None, bars=None, offset=None, **kwargs):
    length = int(length) if length and length > 38 else 40
    bars = int(bars) if bars and bars > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)

    if close is None: return

    m = len(close)
    result = [np.nan for _ in range(0, length - 1)] + [0] * (m - length + 1)
    threadsperblock = 256
    blockspergrid = (m + threadsperblock - 1) // threadsperblock
    calculate_ebsw[blockspergrid, threadsperblock](close.values, length, bars, offset, result)
    ebsw = Series(result, index=close.index)

    if offset != 0:
        ebsw = ebsw.shift(offset)

    if "fillna" in kwargs:
        ebsw.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ebsw.fillna(method=kwargs["fill_method"], inplace=True)

    ebsw.name = f"EBSW_{length}_{bars}"
    ebsw.category = "cycles"

    return ebsw