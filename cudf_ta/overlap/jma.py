# -*- coding: utf-8 -*-
import cudf
import numpy as np
from cuml.preprocessing import CumlScaler

def jma(close, length=None, phase=None, offset=None, **kwargs):
    """Indicator: Jurik Moving Average (JMA)"""
    # Validate Arguments
    _length = int(length) if length and length > 0 else 7
    phase = float(phase) if phase and phase != 0 else 0
    close_gdf = cudf.DataFrame({'close': close})
    close_gdf = close_gdf.iloc[-_length:]
    offset = get_offset(offset)
    if close_gdf is None: return

    # Define base variables
    jma = cudf.Series(np.zeros_like(close_gdf['close']), index=close_gdf.index)
    volty = cudf.Series(np.zeros_like(close_gdf['close']), index=close_gdf.index)
    v_sum = cudf.Series(np.zeros_like(close_gdf['close']), index=close_gdf.index)

    kv = det0 = det1 = ma2 = 0.0
    jma[0] = ma1 = uBand = lBand = close_gdf['close'].iloc[0]

    # Static variables
    sum_length = 10
    length = 0.5 * (_length - 1)
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else 1.5 + phase * 0.01
    length1 = max((np.log(np.sqrt(length)) / np.log(2.0)) + 2.0, 0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(length)
    bet = length2 / (length2 + 1)
    beta = 0.45 * (_length - 1) / (0.45 * (_length - 1) + 2.0)

    m = len(close_gdf)
    for i in range(1, m):
        price = close_gdf['close'].iloc[i]

        # Price volatility
        del1 = price - uBand
        del2 = price - lBand
        volty.iloc[i] = max(abs(del1),abs(del2)) if abs(del1)!=abs(del2) else 0

        # Relative price volatility factor
        v_sum.iloc[i] = v_sum.iloc[i - 1] + (volty.iloc[i] - volty.iloc[max(i - sum_length, 0)]) / sum_length
        avg_volty = np.average(v_sum.iloc[max(i - 65, 0):i + 1])
        d_volty = 0 if avg_volty ==0 else volty.iloc[i] / avg_volty
        r_volty = max(1.0, min(np.power(length1, 1 / pow1), d_volty))

        # Jurik volatility bands
        pow2 = np.power(r_volty, pow1)
        kv = np.power(bet, np.sqrt(pow2))
        uBand = price if (del1 > 0) else price - (kv * del1)
        lBand = price if (del2 < 0) else price - (kv * del2)

        # Jurik Dynamic Factor
        power = np.power(r_volty, pow1)
        alpha = np.power(beta, power)

        # 1st stage - prelimimary smoothing by adaptive EMA
        ma1 = ((1 - alpha) * price) + (alpha * ma1)

        # 2nd stage - one more prelimimary smoothing by Kalman filter
        det0 = ((price - ma1) * (1 - beta)) + (beta * det0)
        ma2 = ma1 + pr * det0

        # 3rd stage - final smoothing by unique Jurik adaptive filter
        det1 = ((ma2 - jma.iloc[i - 1]) * (1 - alpha) * (1 - alpha)) + (alpha * alpha * det1)
        jma.iloc[i] = jma.iloc[i-1] + det1

    # Remove initial lookback data
    jma.iloc[:_length - 1] = np.nan

    # Offset
    if offset != 0:
        jma = jma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        jma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        jma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    jma.name = f"JMA_{_length}_{phase}"
    jma.category = "overlap"

    return jma


jma.__doc__ = \
"""Jurik Moving Average Average (JMA)

Mark Jurik's Moving Average (JMA) attempts to eliminate noise to see the "true"
underlying activity. It has extremely low lag, is very smooth and is responsive
to market gaps.

Sources:
    https://c.mql5.com/forextsd/forum/164/jurik_1.pdf
    https://www.prorealcode.com/prorealtime-indicators/jurik-volatility-bands/

Calculation:
    Default Inputs:
        length=7, phase=0

Args:
    close (pd.Series): Series of 'close's
    length (int): Period of calculation. Default: 7
    phase (float): How heavy/light the average is [-100, 100]. Default: 0
    offset (int): How many lengths to offset the result. Default: 0

Kwargs:
    fillna (value, optional): cudf.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""
