# -*- coding: utf-8 -*-
import cupy as cp
import cudf
from cudf_ta import Imports
from cudf_ta.utils import get_offset, verify_series

def midprice(high, low, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Midprice"""
    # Validate arguments
    length = int(length) if length and length > 0 else 2
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    _length = max(length, min_periods)
    high = verify_series(high, _length)
    low = verify_series(low, _length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None: return

    # Convert cudf.DataFrame to cuDF
    high = cudf.from_pandas(high)
    low = cudf.from_pandas(low)

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MIDPRICE
        midprice = MIDPRICE(high, low, length)
    else:
        lowest_low = low.rolling(window=length, min_periods=min_periods).min()
        highest_high = high.rolling(window=length, min_periods=min_periods).max()
        midprice = 0.5 * (lowest_low + highest_high)

    # Offset
    if offset != 0:
        midprice = midprice.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        midprice.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        midprice.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    midprice.name = f"MIDPRICE_{length}"
    midprice.category = "overlap"

    # Convert back to pandas
    midprice = midprice.to_pandas()

    return midprice