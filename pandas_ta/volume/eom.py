```
# -*- coding: utf-8 -*-
import cudf
from pandas_ta.overlap import hl2, sma
from pandas_ta.utils import get_drift, get_offset, non_zero_range, verify_series

def eom(high, low, close, volume, length=None, divisor=None, drift=None, offset=None, **kwargs):
    """Indicator: Ease of Movement (EOM)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 14
    divisor = divisor if divisor and divisor > 0 else 100000000
    high = cudf.Series(high).astype('float64')
    low = cudf.Series(low).astype('float64')
    close = cudf.Series(close).astype('float64')
    volume = cudf.Series(volume).astype('float64')
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high.has_nulls or low.has_nulls or close.has_nulls or volume.has_nulls: return

    # Calculate Result
    high_low_range = non_zero_range(high, low)
    distance  = hl2(high=high, low=low)
    distance -= hl2(high=high.shift(drift), low=low.shift(drift))
    box_ratio = volume / divisor
    box_ratio /= high_low_range
    eom = distance / box_ratio
    eom = sma(eom, length=length)

    # Offset
    if offset != 0:
        eom = eom.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        eom.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        eom.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    eom.name = f"EOM_{length}_{divisor}"
    eom.category = "volume"

    return eom


eom.__doc__ = \
"""Ease of Movement (EOM)

Ease of Movement is a volume based oscillator that is designed to measure the
relationship between price and volume flucuating across a zero line.

Sources:
    https://www.tradingview.com/wiki/Ease_of_Movement_(EOM)
    https://www.motivewave.com/studies/ease_of_movement.htm
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ease_of_movement_emv

Calculation:
    Default Inputs:
        length=14, divisor=100000000, drift=1
    SMA = Simple Moving Average
    hl_range = high - low
    distance = 0.5 * (high - high.shift(drift) + low - low.shift(drift))
    box_ratio = (volume / divisor) / hl_range
    eom = distance / box_ratio
    EOM = SMA(eom, length)

Args:
    high (cudf.Series): Series of 'high's
    low (cudf.Series): Series of 'low's
    close (cudf.Series): Series of 'close's
    volume (cudf.Series): Series of 'volume's
    length (int): The short period. Default: 14
    drift (int): The diff period. Default: 1
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): cudf.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    cudf.Series: New feature generated.
"""