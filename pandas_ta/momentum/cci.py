import cucim
from cuml.metrics import mean_absolute_error
from cuml.preprocessing import RobustScaler
import cudf
from cuml.stats import mad as cupy_mad

def cci(high, low, close, length=None, c=None, talib=None, offset=None, **kwargs):
    """Indicator: Commodity Channel Index (CCI)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    c = float(c) if c and c > 0 else 0.015
    high = cudf.Series(high)
    low = cudf.Series(low)
    close = cudf.Series(close)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if high is None or low is None or close is None: return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import CCI
        cci = CCI(high, low, close, length)
    else:
        typical_price = (high + low + close) / 3
        mean_typical_price = typical_price.rolling(window=length).mean()
        mad_typical_price = cupy_mad(typical_price, length)

        cci = typical_price - mean_typical_price
        cci /= c * mad_typical_price

    # Offset
    if offset != 0:
        cci = cci.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        cci.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cci.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    cci.name = f"CCI_{length}_{c}"
    cci.category = "momentum"

    return cci