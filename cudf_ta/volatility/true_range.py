# -*- coding: utf-8 -*-
import cudf
from cudf.core.column import numeric
from cudf.core.series import Series
from cuml.utils.import_utils import has_cuml
from cudf_ta.utils import get_drift, get_offset, non_zero_range, verify_series
import numpy as np

def true_range(high, low, close, talib=None, drift=None, offset=None, **kwargs):
    """Indicator: True Range"""
    # Validate arguments
    high = cudf.Series(high) if isinstance(high, list) else high
    low = cudf.Series(low) if isinstance(low, list) else low
    close = cudf.Series(close) if isinstance(close, list) else close
    drift = get_drift(drift)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # Calculate Result
    if has_cuml() and mode_tal:
        from cuml.dask.common import cuml_memcpy
        true_range = cuml_memcpy(cudf.concat([high, low, close], axis=1)).TrueRange()
    else:
        high_low_range = non_zero_range(high, low)
        prev_close = close.shift(drift)
        ranges = [high_low_range, high - prev_close, prev_close - low]
        true_range = cudf.concat(ranges, axis=1)
        true_range = true_range.abs().max(axis=1)
        true_range.iloc[:drift] = np.nan

    # Offset
    if offset != 0:
        true_range = true_range.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        true_range.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        true_range.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    true_range.name = f"TRUERANGE_{drift}"
    true_range.category = "volatility"

    return true_range


true_range.__doc__ = \
"""True Range

An method to expand a classical range (high minus low) to include
possible gap scenarios.

Sources:
    https://www.macroption.com/true-range/

Calculation:
    Default Inputs:
        drift=1
    ABS = Absolute Value
    prev_close = close.shift(drift)
    TRUE_RANGE = ABS([high - low, high - prev_close, low - prev_close])

Args:
    high (cudf.Series or list): Series of 'high's
    low (cudf.Series or list): Series of 'low's
    close (cudf.Series or list): Series of 'close's
    talib (bool): If CuML is installed and talib is True, Returns the CuML
        version. Default: True
    drift (int): The shift period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): cudf.Series.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    cudf.Series: New feature
"""