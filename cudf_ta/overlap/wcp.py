# -*- coding: utf-8 -*-
import cudf
from cudf_ta import Imports
from cudf_ta.utils import get_offset, verify_series

def wcp(high, low, close, talib=None, offset=None, **kwargs):
    """Indicator: Weighted Closing Price (WCP)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # Calculate Result
    if Imports["cupy"] and mode_tal:
        import cupy as cp
        high = cudf.Series(high).values
        low = cudf.Series(low).values
        close = cudf.Series(close).values
        wcp = cp.ElementwiseKernel('float64 x, float64 y, float64 z', 'float64 out', '(x + y + 2 * z) / 4')(high, low, close)
    else:
        wcp = (cudf.Series(high) + cudf.Series(low) + 2 * cudf.Series(close)) / 4

    # Offset
    if offset != 0:
        wcp = wcp.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        wcp.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        wcp.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    wcp.name = "WCP"
    wcp.category = "overlap"

    return wcp


wcp.__doc__ = \
"""Weighted Closing Price (WCP)

Weighted Closing Price is the weighted price given: high, low
and double the close.

Sources:
    https://www.fmlabs.com/reference/default.htm?url=WeightedCloses.htm

Calculation:
    WCP = (2 * close + high + low) / 4

Args:
    high (cudf.Series): Series of 'high's
    low (cudf.Series): Series of 'low's
    close (cudf.Series): Series of 'close's
    talib (bool): If cuPY is installed and talib is True, Returns the cuPY
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): cudf.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    cudf.Series: New feature generated.
"""