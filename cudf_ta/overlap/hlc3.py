# -*- coding: utf-8 -*-
import cudf
from cudf.core.column import string
from cudf.utils import cuda
from cupy import get_array_module
import numpy as np

def hlc3(high, low, close, talib=None, offset=None, **kwargs):
    """Indicator: HLC3"""
    # Validate Arguments
    high = cudf.Series(high._column)
    low = cudf.Series(low._column)
    close = cudf.Series(close._column)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # Calculate Result
    if mode_tal:
        hlc3 = (high + low + close) / 3.0
    else:
        hlc3 = (high + low + close) / 3.0

    # Offset
    if offset != 0:
        hlc3 = hlc3.shift(offset)

    # Name & Category
    hlc3.name = "HLC3"
    hlc3.category = "overlap"

    return hlc3