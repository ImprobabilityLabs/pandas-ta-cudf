Here is the refactored code to work with CuDF and GPU processing:

```python
# -*- coding: utf-8 -*-
import cudf
from cudf import DataFrame as GCudfDataFrame
from .tsignals import tsignals
from pandas_ta.utils._signals import cross_value
from pandas_ta.utils import get_offset, verify_series


def xsignals(signal, xa, xb, above:bool=True, long:bool=True, asbool:bool=None, trend_reset:int=0, trade_offset:int=None, offset:int=None, **kwargs):
    """Indicator: Cross Signals"""
    # Validate Arguments
    signal = verify_series(signal)
    offset = get_offset(offset)

    # Calculate Result
    if above:
        entries = cross_value(signal, xa)
        exits = -cross_value(signal, xb, above=False)
    else:
        entries = cross_value(signal, xa, above=False)
        exits = -cross_value(signal, xb)
    trades = entries + exits

    # Modify trades to fill gaps for trends
    trades.replace({0: float('nan')}, inplace=True)
    trades.interpolate(method="pad", inplace=True)
    trades.fillna(0, inplace=True)

    trends = (trades > 0).astype(int)
    if not long:
        trends = 1 - trends

    tskwargs = {
        "asbool":asbool,
        "trade_offset":trade_offset,
        "trend_reset":trend_reset,
        "offset":offset
    }
    df = tsignals(trends, **tskwargs)

    # Offset handled by tsignals
    result = GCudfDataFrame({
        f"XS_LONG": df.TS_Trends,
        f"XS_SHORT": 1 - df.TS_Trends
    })


    # Handle fills
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        result.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    result.name = f"XS"
    result.category = "trend"

    return result
```