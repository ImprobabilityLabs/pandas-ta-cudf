```
# -*- coding: utf-8 -*-
import cudf
from pandas_ta.utils import get_offset, verify_series

def hlc3(high, low, close, talib=None, offset=None, **kwargs):
    """Indicator: HLC3"""
    # Validate Arguments
    high = cudf.Series(verify_series(high))
    low = cudf.Series(verify_series(low))
    close = cudf.Series(verify_series(close))
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
    hlc3.metadata = {"category": "overlap"}

    return hlc3
```