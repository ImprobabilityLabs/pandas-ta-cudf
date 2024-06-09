import cudf
from cuml.preprocessing import StandardScaler
import numpy as np

def bbands(close, length=None, std=None, ddof=0, mamode=None, talib=None, offset=None, **kwargs):
    """Indicator: Bollinger Bands (BBANDS)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 5
    std = float(std) if std and std > 0 else 2.0
    mamode = mamode if isinstance(mamode, str) else "sma"
    ddof = int(ddof) if ddof >= 0 and ddof < length else 1
    close = cudf.Series(close)
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import BBANDS
        upper, mid, lower = BBANDS(close, length, std, std, tal_ma(mamode))
    else:
        standard_deviation = close.rolling(window=length).std(ddof=ddof)
        deviations = std * standard_deviation
        # deviations = std * standard_deviation.loc[standard_deviation.first_valid_index():,]

        mid = close.rolling(window=length).mean()
        lower = mid - deviations
        upper = mid + deviations

    ulr = non_zero_range(upper, lower)
    bandwidth = 100 * ulr / mid
    percent = non_zero_range(close, lower) / ulr

    # Offset
    if offset != 0:
        lower = lower.shift(offset)
        mid = mid.shift(offset)
        upper = upper.shift(offset)
        bandwidth = bandwidth.shift(offset)
        percent = bandwidth.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        lower.fillna(kwargs["fillna"], inplace=True)
        mid.fillna(kwargs["fillna"], inplace=True)
        upper.fillna(kwargs["fillna"], inplace=True)
        bandwidth.fillna(kwargs["fillna"], inplace=True)
        percent.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        lower.fillna(method=kwargs["fill_method"], inplace=True)
        mid.fillna(method=kwargs["fill_method"], inplace=True)
        upper.fillna(method=kwargs["fill_method"], inplace=True)
        bandwidth.fillna(method=kwargs["fill_method"], inplace=True)
        percent.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    lower.name = f"BBL_{length}_{std}"
    mid.name = f"BBM_{length}_{std}"
    upper.name = f"BBU_{length}_{std}"
    bandwidth.name = f"BBB_{length}_{std}"
    percent.name = f"BBP_{length}_{std}"
    upper.category = lower.category = "volatility"
    mid.category = bandwidth.category = upper.category

    # Prepare DataFrame to return
    data = {
        lower.name: lower, mid.name: mid, upper.name: upper,
        bandwidth.name: bandwidth, percent.name: percent
    }
    bbandsdf = cudf.DataFrame(data)
    bbandsdf.name = f"BBANDS_{length}_{std}"
    bbandsdf.category = mid.category

    return bbandsdf