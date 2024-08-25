```python
# -*- coding: utf-8 -*-
from cudf import DataFrame, Series
from cudf.core.series import Series as verify_series
import cupy as cp
from numba import cuda

# CUDA kernel function to speed up Schaff Trend Cycle calculations
@cuda.jit
def schaff_tc_kernel(xmacd, lowest_xmacd, xmacd_range, factor, pf, m):
    stoch1 = cuda.local.array(1024, dtype=cp.float32)
    stoch2 = cuda.local.array(1024, dtype=cp.float32)
    pff = cuda.local.array(1024, dtype=cp.float32)

    i = cuda.grid(1)
    if i < m:
        stoch1[0], pf[0] = 0.0, 0.0
        lowest_pf = pf.rolling(tclength).min()
        pf_range = non_zero_range(pf.rolling(tclength).max(), lowest_pf)

        if lowest_xmacd[i] > 0:
            stoch1[i] = 100 * ((xmacd[i] - lowest_xmacd[i]) / xmacd_range[i])
        else:
            stoch1[i] = stoch1[i - 1]

        pf[i] = pf[i - 1] + factor * (stoch1[i] - pf[i - 1])
        pf[i] = round(pf[i], 8)

        if pf_range[i] > 0:
            stoch2[i] = 100 * ((pf[i] - lowest_pf[i]) / pf_range[i])
        else:
            stoch2[i] = stoch2[i - 1]

        pff[i] = pff[i - 1] + factor * (stoch2[i] - pff[i - 1])
        pff[i] = round(pff[i], 8)

@cuda.jit    
def schaff_tc(close, xmacd, tclength, factor):
    m = len(xmacd)
    pf = cp.zeros(m, dtype=cp.float32)
    lowest_xmacd = xmacd.rolling(tclength).min()
    xmacd_range = non_zero_range(xmacd.rolling(tclength).max(), lowest_xmacd)

    # Allocate on GPU and execute kernel
    schaff_tc_kernel[1, m](xmacd, lowest_xmacd, xmacd_range, cp.float32(factor), pf, m)
    
    pf = Series(pf, index=close.index)
 
    # Calculate 2nd Stochastic
    pff = cp.zeros(m, dtype=cp.float32)
    lowest_pf = pf.rolling(tclength).min()
    pf_range = non_zero_range(pf.rolling(tclength).max(), lowest_pf)

    schaff_tc_kernel[1, m](pf, lowest_pf, pf_range, cp.float32(factor), pff, m)

    return [pff, pf]


def stc(close, tclength=None, fast=None, slow=None, factor=None, offset=None, **kwargs):
    """Indicator: Schaff Trend Cycle (STC)"""
    # Validate arguments
    tclength = int(tclength) if tclength and tclength > 0 else 10
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    factor = float(factor) if factor and factor > 0 else 0.5
    if slow < fast:                # mandatory condition, but might be confusing
        fast, slow = slow, fast
    _length = max(tclength, fast, slow)
    close = verify_series(close, _length)
    offset = get_offset(offset)

    if close is None: return

    # kwargs allows for three more series (ma1, ma2 and osc) which can be passed
    # here ma1 and ma2 input negate internal ema calculations, osc substitutes
    # both ma's.
    ma1 = kwargs.pop("ma1", False)
    ma2 = kwargs.pop("ma2", False)
    osc = kwargs.pop("osc", False)

    # 3 different modes of calculation..
    if isinstance(ma1, Series) and isinstance(ma2, Series) and not osc:
        ma1 = verify_series(ma1, _length)
        ma2 = verify_series(ma2, _length)

        if ma1 is None or ma2 is None: return
        # Calculate Result based on external feeded series
        xmacd = ma1 - ma2
        # invoke shared calculation
        pff, pf = schaff_tc(close, xmacd, tclength, factor)

    elif isinstance(osc, Series):
        osc = verify_series(osc, _length)
        if osc is None: return
        # Calculate Result based on feeded oscillator
        # (should be ranging around 0 x-axis)
        xmacd = osc
        # invoke shared calculation
        pff, pf = schaff_tc(close, xmacd, tclength, factor)

    else:
        # Calculate Result .. (traditionel/full)
        # MACD line
        fastma = ema(close, length=fast)
        slowma = ema(close, length=slow)
        xmacd = fastma - slowma
        # invoke shared calculation
        pff, pf = schaff_tc(close, xmacd, tclength, factor)

    # Resulting Series
    stc = Series(pff, index=close.index)
    macd = Series(xmacd, index=close.index)
    stoch = Series(pf, index=close.index)

    # Offset
    if offset != 0:
        stc = stc.shift(offset)
        macd = macd.shift(offset)
        stoch = stoch.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        stc.fillna(kwargs["fillna"], inplace=True)
        macd.fillna(kwargs["fillna"], inplace=True)
        stoch.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        stc.fillna(method=kwargs["fill_method"], inplace=True)
        macd.fillna(method=kwargs["fill_method"], inplace=True)
        stoch.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _props = f"_{tclength}_{fast}_{slow}_{factor}"
    stc.name = f"STC{_props}"
    macd.name = f"STCmacd{_props}"
    stoch.name = f"STCstoch{_props}"
    stc.category = macd.category = stoch.category ="momentum"

    # Prepare DataFrame to return
    data = {stc.name: stc, macd.name: macd, stoch.name: stoch}
    df = DataFrame(data)
    df.name = f"STC{_props}"
    df.category = stc.category

    return df
```