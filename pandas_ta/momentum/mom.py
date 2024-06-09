Here is the refactored code to work with CuDF:

```python
# -*- coding: utf-8 -*-
import cudf
from pandas_ta import Imports
from pandas_ta.utils import get_offset, verify_series


def mom(close, length=None, talib=None, offset=None, **kwargs):
    """Indicator: Momentum (MOM)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from taslib import MOM  # replaced talib with taslib
        mom = MOM(close, length)
    else:
        close = cudf.DataFrame({'close': close})  # convert to CuDF DataFrame
        mom = close['close'].diff(length)  # use CuDF's diff method

    # Offset
    if offset != 0:
        mom = mom.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        mom.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        mom.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    mom.name = f"MOM_{length}"
    mom.category = "momentum"

    return mom


mom.__doc__ = \
"""Momentum (MOM)

Momentum is an indicator used to measure a security's speed (or strength) of
movement.  Or simply the change in price.

Sources:
    http://www.onlinetradingconcepts.com/TechnicalAnalysis/Momentum.html

Calculation:
    Default Inputs:
        length=1
    MOM = close.diff(length)

Args:
    close (pd.Series or cudf.Series): Series of 'close's
    length (int): It's period. Default: 1
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series or cudf.Series: New feature generated.
"""
```