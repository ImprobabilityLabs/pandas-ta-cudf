Here is the refactored code to work with CuDF and CUDA:
```
# -*- coding: utf-8 -*-
import cucim
import cupy
import cudf
from numba import cuda
from cucim.utils import get_drift, get_offset, verify_series

def vidya(close, length=None, drift=None, offset=None, **kwargs):
    """Indicator: Variable Index Dynamic Average (VIDYA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 14
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if close is None: return

    def _cmo(source: cudf.Series, n: int, d: int):
        """Chande Momentum Oscillator (CMO) Patch"""
        mom = source.diff(d)
        positive = mom.clip(lower=0)
        negative = mom.clip(upper=0).abs()
        pos_sum = positive.rolling(n).sum()
        neg_sum = negative.rolling(n).sum()
        return (pos_sum - neg_sum) / (pos_sum + neg_sum)

    # Calculate Result
    m = close.size
    alpha = 2 / (length + 1)
    abs_cmo = _cmo(close, length, drift).abs()
    vidya = cudf.Series(0, index=close.index)
    for i in range(length, m):
        vidya.iloc[i] = alpha * abs_cmo.iloc[i] * close.iloc[i] + vidya.iloc[i - 1] * (1 - alpha * abs_cmo.iloc[i])
    vidya.replace({0: cupy.nan}, inplace=True)

    # Offset
    if offset != 0:
        vidya = vidya.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        vidya.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vidya.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    vidya.name = f"VIDYA_{length}"
    vidya.category = "overlap"

    return vidya

vidya.__doc__ = \
"""Variable Index Dynamic Average (VIDYA)

Variable Index Dynamic Average (VIDYA) was developed by Tushar Chande. It is
similar to an Exponential Moving Average but it has a dynamically adjusted
lookback period dependent on relative price volatility as measured by Chande
Momentum Oscillator (CMO). When volatility is high, VIDYA reacts faster to
price changes. It is often used as moving average or trend identifier.

Sources:
    https://www.tradingview.com/script/hdrf0fXV-Variable-Index-Dynamic-Average-VIDYA/
    https://www.perfecttrendsystem.com/blog_mt4_2/en/vidya-indicator-for-mt4

Calculation:
    Default Inputs:
        length=10, adjust=False, sma=True
    if sma:
        sma_nth = close[0:length].sum() / length
        close[:length - 1] = cupy.nan
        close.iloc[length - 1] = sma_nth
    EMA = close.ewm(span=length, adjust=adjust).mean()

Args:
    close (cudf.Series): Series of 'close's
    length (int): It's period. Default: 14
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    adjust (bool, optional): Use adjust option for EMA calculation. Default: False
    sma (bool, optional): If True, uses SMA for initial value for EMA calculation. Default: True
    talib (bool): If True, uses TA-Libs implementation for CMO. Otherwise uses EMA version. Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    cudf.Series: New feature generated.
"""