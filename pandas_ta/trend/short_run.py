# -*- coding: utf-8 -*-
import cudf
from .decreasing import decreasing
from .increasing import increasing
from pandas_ta.utils import get_offset, verify_series

def short_run(fast, slow, length=None, offset=None, **kwargs):
    """Indicator: Short Run"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 2
    fast = verify_series(fast, length)
    slow = verify_series(slow, length)
    offset = get_offset(offset)

    if fast is None or slow is None: return

    # Convert to CuDF
    fast_cu = cudf.DataFrame({'fast': fast.values})
    slow_cu = cudf.DataFrame({'slow': slow.values})

    # Calculate Result
    pt = decreasing(fast_cu, length) & increasing(slow_cu, length)  # potential top or top
    bd = decreasing(fast_cu, length) & decreasing(slow_cu, length)  # fast and slow are decreasing
    short_run_cu = pt & bd | pt | bd

    # Offset
    if offset != 0:
        short_run_cu = short_run_cu.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        short_run_cu.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        short_run_cu.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    short_run_cu.name = f"SR_{length}"
    short_run_cu.category = "trend"

    return short_run_cu