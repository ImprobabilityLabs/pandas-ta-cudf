import cudf
from cuml.integrations.pycuda import rs
from culibsOverlap import ma
from culibsUtils import get_offset, non_zero_range, verify_series

def stochrsi(close, length=None, rsi_length=None, k=None, d=None, mamode=None, offset=None, **kwargs):
    length = length if length and length > 0 else 14
    rsi_length = rsi_length if rsi_length and rsi_length > 0 else 14
    k = k if k and k > 0 else 3
    d = d if d and d > 0 else 3
    close = verify_series(close, max(length, rsi_length, k, d))
    offset = get_offset(offset)
    mamode = mamode if isinstance(mamode, str) else "sma"

    if close is None: return

    rsi_ = rs.rsi(close, length=rsi_length)
    lowest_rsi = rsi_.rolling(window=length).min()
    highest_rsi = rsi_.rolling(window=length).max()

    stoch = 100 * (rsi_ - lowest_rsi)
    stoch /= non_zero_range(highest_rsi, lowest_rsi)

    stochrsi_k = ma(mamode, stoch, length=k)
    stochrsi_d = ma(mamode, stochrsi_k, length=d)

    if offset != 0:
        stochrsi_k = stochrsi_k.shift(offset)
        stochrsi_d = stochrsi_d.shift(offset)

    if "fillna" in kwargs:
        stochrsi_k.fillna(kwargs["fillna"], inplace=True)
        stochrsi_d.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stochrsi_k.fillna(method=kwargs["fill_method"], inplace=True)
        stochrsi_d.fillna(method=kwargs["fill_method"], inplace=True)

    _name = "STOCHRSI"
    _props = f"_{length}_{rsi_length}_{k}_{d}"
    stochrsi_k.name = f"{_name}k{_props}"
    stochrsi_d.name = f"{_name}d{_props}"
    stochrsi_k.category = stochrsi_d.category = "momentum"

    data = {stochrsi_k.name: stochrsi_k, stochrsi_d.name: stochrsi_d}
    df = cudf.DataFrame(data)
    df.name = f"{_name}{_props}"
    df.category = stochrsi_k.category

    return df