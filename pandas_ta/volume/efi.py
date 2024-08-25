# -*- coding: utf-8 -*-
import cudf
from pandas_ta.overlap import ma
from pandas_ta.utils import get_drift, get_offset, verify_series

@cuda.jit
def efi_cuda(close, volume, length, drift, offset, mamode):
    i = cuda.grid(1)
    if i < close.size:
        pv_diff = close[i] * volume[i] - close[i - drift] * volume[i - drift]
        efi = 0
        for j in range(1, length + 1):
            efi += pv_diff / j
        if mamode == "ema":
            efi /= 2.71828 ** (1 / length)
        efi = efi if offset == 0 else efi[i - offset]
    return efi

def efi(close, volume, length=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Elder's Force Index (EFI)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 13
    mamode = mamode if isinstance(mamode, str) else "ema"
    close = verify_series(close, length)
    volume = verify_series(volume, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None or volume is None: return

    # Calculate Result
    close_cuda = cudf.Series(close)
    volume_cuda = cudf.Series(volume)
    efi = cudf.Series(efi_cuda[1, close_cuda.size](close_cuda, volume_cuda, length, drift, offset, mamode))

    # Handle fills
    if "fillna" in kwargs:
        efi.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        efi.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    efi.name = f"EFI_{length}"
    efi.category = "volume"

    return efi

efi.__doc__ = \
"""Elder's Force Index (EFI)

Elder's Force Index measures the power behind a price movement using price
and volume as well as potential reversals and price corrections.

Sources:
    https://www.tradingview.com/wiki/Elder%27s_Force_Index_(EFI)
    https://www.motivewave.com/studies/elders_force_index.htm

Calculation:
    Default Inputs:
        length=20, drift=1, mamode=None
    EMA = Exponential Moving Average
    SMA = Simple Moving Average

    pv_diff = close.diff(drift) * volume
    if mamode == 'sma':
        EFI = SMA(pv_diff, length)
    else:
        EFI = EMA(pv_diff, length)

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The short period. Default: 13
    drift (int): The diff period. Default: 1
    mamode (str): See ```help(ta.ma)```. Default: 'ema'
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""