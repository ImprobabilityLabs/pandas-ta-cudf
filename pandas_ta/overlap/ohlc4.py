# -*- coding: utf-8 -*-
import cudf
from pandas_ta.utils import get_offset, verify_series_cudf

def ohlc4(open_, high, low, close, offset=None, **kwargs):
    """Indicator: OHLC4"""
    # Validate Arguments
    open_ = verify_series_cudf(open_)
    high = verify_series_cudf(high)
    low = verify_series_cudf(low)
    close = verify_series_cudf(close)
    offset = get_offset(offset)

    # Calculate Result
    ohlc4 = 0.25 * (open_ + high + low + close)

    # Offset
    if offset != 0:
        ohlc4 = ohlc4.roll(offset, axis=0)

    # Name & Category
    ohlc4.name = "OHLC4"
    ohlc4.attrs["category"] = "overlap"

    return ohlc4