# -*- coding: utf-8 -*-
import cucim
from cudf import Series
from cuml.tensor.tensor_algo import cumsum
from pandas_ta.utils import get_offset, non_zero_range, verify_series
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def ad(high, low, close, volume, open_=None, talib=None, offset=None, **kwargs):
    """Indicator: Accumulation/Distribution (AD)"""
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import AD
        ad = AD(high, low, close, volume)
    else:
        if open_ is not None:
            open_ = verify_series(open_)
            ad = non_zero_range(close, open_)  # AD with Open
        else:
            ad = 2 * close - (high + low)  # AD with High, Low, Close

        high_low_range = non_zero_range(high, low)
        ad *= volume / high_low_range
        ad = cumsum(ad)

    # Offset
    if offset != 0:
        ad = ad.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        ad.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        ad.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    ad.name = "AD" if open_ is None else "ADo"
    ad.category = "volume"

    return ad


ad.__doc__ = \
"""Accumulation/Distribution (AD)

Accumulation/Distribution indicator utilizes the relative position
of the close to it's High-Low range with volume.  Then it is cumulated.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/accumulationdistribution-ad/

Calculation:
    CUM = Cumulative Sum
    if 'open':
        AD = close - open
    else:
        AD = 2 * close - high - low

    hl_range = high - low
    AD = AD * volume / hl_range
    AD = CUM(AD)

Args:
    high (pd.Series or cudf.Series): Series of 'high's
    low (pd.Series or cudf.Series): Series of 'low's
    close (pd.Series or cudf.Series): Series of 'close's
    volume (pd.Series or cudf.Series): Series of 'volume's
    open (pd.Series or cudf.Series): Series of 'open's
    talib (bool): If TA Lib is installed and talib is True, Returns the TA Lib
        version. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series or cudf.Series: New feature generated.
"""