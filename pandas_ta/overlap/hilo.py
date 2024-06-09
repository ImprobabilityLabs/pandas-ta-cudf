import cupy as cp
import cudf
from cuml.metrics import moving_average as ma
from .ma import ma as gpu_ma
import numpy as np

def hilo(high, low, close, high_length=None, low_length=None, mamode=None, offset=None, **kwargs):
    """Indicator: Gann HiLo (HiLo)"""
    # Validate Arguments
    high_length = int(high_length) if high_length and high_length > 0 else 13
    low_length = int(low_length) if low_length and low_length > 0 else 21
    mamode = mamode.lower() if isinstance(mamode, str) else "sma"
    _length = max(high_length, low_length)
    high = cudf.Series(high).iloc[-_length:]
    low = cudf.Series(low).iloc[-_length:]
    close = cudf.Series(close).iloc[-_length:]
    offset = get_offset(offset)

    if high.empty or low.empty or close.empty: return

    # Calculate Result
    m = close.size
    hilo = cudf.Series([np.nan]*m, index=close.index)
    long = cudf.Series([np.nan]*m, index=close.index)
    short = cudf.Series([np.nan]*m, index=close.index)

    high_ma = gpu_ma(mamode, high, length=high_length)
    low_ma = gpu_ma(mamode, low, length=low_length)

    for i in range(1, m):
        if close.iloc[i] > high_ma.iloc[i - 1]:
            hilo.iloc[i] = long.iloc[i] = low_ma.iloc[i]
        elif close.iloc[i] < low_ma.iloc[i - 1]:
            hilo.iloc[i] = short.iloc[i] = high_ma.iloc[i]
        else:
            hilo.iloc[i] = hilo.iloc[i - 1]
            long.iloc[i] = short.iloc[i] = hilo.iloc[i - 1]

    # Offset
    if offset != 0:
        hilo = hilo.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        hilo.fillna(kwargs["fillna"], inplace=True)
        long.fillna(kwargs["fillna"], inplace=True)
        short.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        hilo.fillna(method=kwargs["fill_method"], inplace=True)
        long.fillna(method=kwargs["fill_method"], inplace=True)
        short.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    _props = f"_{high_length}_{low_length}"
    data = {f"HILO{_props}": hilo, f"HILOl{_props}": long, f"HILOs{_props}": short}
    df = cudf.DataFrame(data, index=close.index)

    df.name = f"HILO{_props}"
    df.category = "overlap"

    return df