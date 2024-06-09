import cudf
from pandas_ta.overlap import ma
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series

def accbands(high, low, close, length=None, c=None, drift=None, mamode=None, offset=None, **kwargs):
    """Indicator: Acceleration Bands (ACCBANDS)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 20
    c = float(c) if c and c > 0 else 4
    mamode = mamode if isinstance(mamode, str) else "sma"
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None: return

    # Calculate Result
    high_low_range = non_zero_range(high, low)
    hl_ratio = high_low_range / (high + low)
    hl_ratio *= c
    _lower = low * (1 - hl_ratio)
    _upper = high * (1 + hl_ratio)

    lower = ma(mamode, _lower, length=length)
    mid = ma(mamode, close, length=length)
    upper = ma(mamode, _upper, length=length)

    # Offset
    if offset != 0:
        lower = lower.shift(offset)
        mid = mid.shift(offset)
        upper = upper.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        mid.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        mid.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    lower.name = f"ACCBL_{length}"
    mid.name = f"ACCBM_{length}"
    upper.name = f"ACCBU_{length}"
    mid.category = upper.category = lower.category = "volatility"

    # Prepare DataFrame to return
    data = {lower.name: lower, mid.name: mid, upper.name: upper}
    accbandsdf = cudf.DataFrame(data)
    accbandsdf.name = f"ACCBANDS_{length}"
    accbandsdf.category = mid.category

    return accbandsdf