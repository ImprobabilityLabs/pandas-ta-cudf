# -*- coding: utf-8 -*-
import cudf
from cudf_ta.overlap import sma as cpu_sma
from cudf_ta.utils import get_offset, high_low_range, is_percent
from cudf_ta.utils import real_body, verify_series

def sma(cuda-ser, length):
    return cuda-ser.rolling(window=length).mean()

def cdl_doji(open_, high, low, close, length=None, factor=None, scalar=None, asint=True, offset=None, **kwargs):
    """Candle Type: Doji"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    factor = float(factor) if is_percent(factor) else 10
    scalar = float(scalar) if scalar else 100
    open_ = verify_series(open_, length)
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    offset = get_offset(offset)
    naive = kwargs.pop("naive", False)

    if open_ is None or high is None or low is None or close is None: return

    # Calculate Result
    open_cuda = cudf.Series(open_.values)
    high_cuda = cudf.Series(high.values)
    low_cuda = cudf.Series(low.values)
    close_cuda = cudf.Series(close.values)

    body = real_body(open_cuda, close_cuda).abs()
    hl_range = high_low_range(high_cuda, low_cuda).abs()
    hl_range_avg = sma(hl_range, length)
    doji = body < 0.01 * factor * hl_range_avg

    if naive:
        doji.iloc[:length] = body < 0.01 * factor * hl_range
    if asint:
        doji = scalar * doji.astype(int)

    # Offset
    if offset != 0:
        doji = doji.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        doji.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        doji.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    doji.name = f"CDL_DOJI_{length}_{0.01 * factor}"
    doji.category = "candles"

    return doji.to_pandas()
