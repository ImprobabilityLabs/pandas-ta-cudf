import cudf
from cuml.utils import get cuda_version as cuv

def cmf(high, low, close, volume, open_=None, length=None, offset=None, **kwargs):
    """Indicator: Chaikin Money Flow (CMF)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 20
    min_periods = int(kwargs["min_periods"]) if "min_periods" in kwargs and kwargs["min_periods"] is not None else length
    _length = max(length, min_periods)
    high = cudf.Series(high). astype('float64')
    low = cudf.Series(low).astype('float64')
    close = cudf.Series(close).astype('float64')
    volume = cudf.Series(volume).astype('float64')
    offset = get_offset(offset)

    if high is None or low is None or close is None or volume is None: return

    # Calculate Result
    if open_ is not None:
        open_ = cudf.Series(open_).astype('float64')
        ad = non_zero_range(close, open_)  # AD with Open
    else:
        ad = 2 * close - (high + low)  # AD with High, Low, Close

    ad *= volume / non_zero_range(high, low)
    cmf = ad.rolling(window=length, min_periods=min_periods).sum()
    cmf /= volume.rolling(window=length, min_periods=min_periods).sum()

    # Offset
    if offset != 0:
        cmf = cmf.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        cmf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        cmf.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    cmf.name = f"CMF_{length}"
    cmf.category = "volume"

    return cmf