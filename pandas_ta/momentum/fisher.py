# -*- coding: utf-8 -*-
import cupy as cp
import cudf
from cuml.preprocessing.data import ArrayListener
from cuml.utils import get_cudflocales
from pandas_ta.overlap import hl2
from pandas_ta.utils import get_offset, high_low_range, verify_series


def fisher(high, low, length=None, signal=None, offset=None, **kwargs):
    """Indicator: Fisher Transform (FISHT)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 9
    signal = int(signal) if signal and signal > 0 else 1
    _length = max(length, signal)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)

    if high is None or low is None: return

    # Calculate Result
    hl2_ = hl2(high, low)
    highest_hl2 = hl2_.rolling(window=length).max()
    lowest_hl2 = hl2_.rolling(window=length).min()

    hlr = high_low_range(highest_hl2, lowest_hl2)
    hlr[hlr < 0.001] = 0.001

    position = ((hl2_ - lowest_hl2) / hlr) - 0.5

    v = 0
    m = high.size
    result = [cp.nan for _ in range(0, length - 1)] + [0]
    for i in range(length, m):
        v = 0.66 * position.iloc[i] + 0.67 * v
        if v < -0.99: v = -0.999
        if v > 0.99: v = 0.999
        result.append(0.5 * (cp.log((1 + v) / (1 - v)) + result[i - 1]))
    fisher = cudf.Series(result, index=high.index)
    signalma = fisher.shift(signal)

    # Offset
    if offset != 0:
        fisher = fisher.shift(offset)
        signalma = signalma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        fisher.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        fisher.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _props = f"_{length}_{signal}"
    fisher.name = f"FISHERT{_props}"
    signalma.name = f"FISHERTs{_props}"
    fisher.category = signalma.category = "momentum"

    # Prepare DataFrame to return
    data = {fisher.name: fisher, signalma.name: signalma}
    df = cudf.DataFrame(data)
    df.name = f"FISHERT{_props}"
    df.category = fisher.category

    return df