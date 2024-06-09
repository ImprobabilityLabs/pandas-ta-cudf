# -*- coding: utf-8 -*-
import cudf
from cucim import roc as cucim_roc
from pandas_ta.overlap import wma as cpu_wma
from pandas_ta.utils import get_offset, verify_series

def wma-gnu(gpu_data, length):
    return gpu_data.rolling(window=length).mean()

def coppock(close, length=None, fast=None, slow=None, offset=None, **kwargs):
    """Indicator: Coppock Curve (COPC)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    fast = int(fast) if fast and fast > 0 else 11
    slow = int(slow) if slow and slow > 0 else 14
    close = verify_series(close, max(length, fast, slow))
    offset = get_offset(offset)

    if close is None: return

    # Move data to GPU
    gpu_data = cudf.Series(close)

    # Calculate Result
    total_roc = cucim_roc(gpu_data, fast) + cucim_roc(gpu_data, slow)
    coppock = wma-gnu(total_roc, length)

    # Offset
    if offset != 0:
        coppock = coppock.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        coppock.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        coppock.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    coppock.name = f"COPC_{fast}_{slow}_{length}"
    coppock.category = "momentum"

    return coppock


coppock.__doc__ = \
"""Coppock Curve (COPC)

Coppock Curve (originally called the "Trendex Model") is a momentum indicator
is designed for use on a monthly time scale.  Although designed for monthly
use, a daily calculation over the same period can be made, converting the
periods to 294-day and 231-day rate of changes, and a 210-day weighted
moving average.

Sources:
    https://en.wikipedia.org/wiki/Coppock_curve

Calculation:
    Default Inputs:
        length=10, fast=11, slow=14
    SMA = Simple Moving Average
    MAD = Mean Absolute Deviation
    tp = typical_price = hlc3 = (high + low + close) / 3
    mean_tp = SMA(tp, length)
    mad_tp = MAD(tp, length)
    CCI = (tp - mean_tp) / (c * mad_tp)

Args:
    close (pd.Series): Series of 'close's
    length (int): WMA period. Default: 10
    fast (int): Fast ROC period. Default: 11
    slow (int): Slow ROC period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""