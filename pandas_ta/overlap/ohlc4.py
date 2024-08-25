# -*- coding: utf-8 -*-
import cudf
from pandas_ta.utils import get_offset, verify_series

def ohlc4(open_, high, low, close, offset=None, **kwargs):
    """Indicator: OHLC4"""
    # Validate Arguments
    open_ = cudf.Series(verify_series(open_))
    high = cudf.Series(verify_series(high))
    low = cudf.Series(verify_series(low))
    close = cudf.Series(verify_series(close))
    offset = get_offset(offset)

    # Calculate Result
    ohlc4 = 0.25 * (open_ + high + low + close)

    # Offset
    if offset != 0:
        ohlc4 = ohlc4.shift(offset)

    # Name & Category
    ohlc4.name = "OHLC4"
    ohlc4.attrs['category'] = "overlap"  # cudf doesn't support category attribute like pandas

    return ohlc4