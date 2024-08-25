# -*- coding: utf-8 -*-
import cudf
from cudf_ta.utils import get_drift, get_offset, non_zero_range, verify_series

def pdist(open_, high, low, close, drift=None, offset=None, **kwargs):
    """Indicator: Price Distance (PDIST)"""
    # Validate Arguments
    open_ = verify_series(open_.to_cudf())
    high = verify_series(high.to_cudf())
    low = verify_series(low.to_cudf())
    close = verify_series(close.to_cudf())
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    pdist = 2 * non_zero_range(high, low)
    pdist += non_zero_range(open_, close.shift(drift)).abs()
    pdist -= non_zero_range(close, open_).abs()

    # Offset
    if offset != 0:
        pdist = pdist.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        pdist.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        pdist.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    pdist.name = "PDIST"
    pdist.category = "volatility"

    return pdist


pdist.__doc__ = \
"""Price Distance (PDIST)

Measures the "distance" covered by price movements.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/pricedistance/

Calculation:
    Default Inputs:
        drift=1

    PDIST = 2(high - low) - ABS(close - open) + ABS(open - close[drift])

Args:
    open_ (cudf.Series): Series of 'opens's
    high (cudf.Series): Series of 'high's
    low (cudf.Series): Series of 'low's
    close (cudf.Series): Series of 'close's
    drift (int): The difference period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): cudf.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    cudf.Series: New feature generated.
"""