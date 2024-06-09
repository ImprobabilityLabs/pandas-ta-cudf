import cudf
from cuml.utils import get_cuml_array
import cupy as cp
import numpy as np
from cuml.metrics import mean_absolute_error

def mad(close, length=None, offset=None, **kwargs):
    """Indicator: Mean Absolute Deviation"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 30
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    close = cudf.Series(get_cuml_array(close, dtype=cp.float64), index=cudf.RangeIndex(len(close)))
    offset = get_offset(offset)

    if close is None: return

    # Calculate Result
    def mad_(series):
        """Mean Absolute Deviation"""
        mean = cp.mean(series)
        return mean_absolute_error(series, mean)

    mad = close.rolling(window=length, min_periods=min_periods).apply(mad_)

    # Offset
    if offset != 0:
        mad = mad.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        mad.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        mad.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    mad.name = f"MAD_{length}"
    mad.category = "statistics"

    return mad

mad.__doc__ = \
"""Rolling Mean Absolute Deviation

Sources:

Calculation:
    Default Inputs:
        length=30
    mad = close.rolling(length).mad()

Args:
    close (cudf.Series): Series of 'close's
    length (int): It's period. Default: 30
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    cudf.Series: New feature generated.
"""