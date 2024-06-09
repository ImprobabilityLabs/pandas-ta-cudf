from cudf import DataFrame
from cupy import zeros_like
from cuml.tsa.overlap import ma as cu_ma
from cuml.tsa.volatility import atr
from cuml.tsa.utils import get_drift, get_offset, verify_series, zero
import cupy as cp

def adx(high, low, close, length=None, lensig=None, scalar=None, mamode=None, drift=None, offset=None, **kwargs):
    length = length if length and length > 0 else 14
    lensig = lensig if lensig and lensig > 0 else length
    mamode = mamode if isinstance(mamode, str) else "rma"
    scalar = float(scalar) if scalar else 100
    high = verify_series(high, length)
    low = verify_series(low, length)
    close = verify_series(close, length)
    drift = get_drift(drift)
    offset = get_offset(offset)

    if high is None or low is None or close is None: return

    atr_ = atr(high=high, low=low, close=close, length=length)

    up = high - high.shift(drift)  # high.diff(drift)
    dn = low.shift(drift) - low    # low.diff(-drift).shift(drift)

    pos = ((up > dn) & (up > 0)) * up
    neg = ((dn > up) & (dn > 0)) * dn

    pos = pos.apply(zero)
    neg = neg.apply(zero)

    k = scalar / atr_
    dmp = k * cu_ma(mamode, pos, length=length)
    dmn = k * cu_ma(mamode, neg, length=length)

    dx = scalar * (dmp - dmn).abs() / (dmp + dmn)
    adx = cu_ma(mamode, dx, length=lensig)

    if offset != 0:
        dmp = dmp.shift(offset)
        dmn = dmn.shift(offset)
        adx = adx.shift(offset)

    if "fillna" in kwargs:
        adx.fillna(kwargs["fillna"], inplace=True)
        dmp.fillna(kwargs["fillna"], inplace=True)
        dmn.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        adx.fillna(method=kwargs["fill_method"], inplace=True)
        dmp.fillna(method=kwargs["fill_method"], inplace=True)
        dmn.fillna(method=kwargs["fill_method"], inplace=True)

    adx.name = f"ADX_{lensig}"
    dmp.name = f"DMP_{length}"
    dmn.name = f"DMN_{length}"

    adx.category = dmp.category = dmn.category = "trend"

    data = {adx.name: adx, dmp.name: dmp, dmn.name: dmn}
    adxdf = DataFrame(data)
    adxdf.name = f"ADX_{lensig}"
    adxdf.category = "trend"

    return adxdf