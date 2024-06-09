import cudf
from numba import cuda
import numpy as np
import pandas as pd

def psar(high, low, close=None, af0=None, af=None, max_af=None, offset=None, **kwargs):
    # Validate Arguments
    high = cudf.Series(high) if not isinstance(high, cudf.Series) else high
    low = cudf.Series(low) if not isinstance(low, cudf.Series) else low
    af = float(af) if af and af > 0 else 0.02
    af0 = float(af0) if af0 and af0 > 0 else af
    max_af = float(max_af) if max_af and max_af > 0 else 0.2
    offset = get_offset(offset)

    @cuda.jit
    def _falling(high, low, drift):
        up = high - high.shift(drift)
        dn = low.shift(drift) - low
        _dmn = (((dn > up) & (dn > 0)) * dn).apply(zero).iloc[-1]
        return _dmn > 0

    # Falling if the first NaN -DM is positive
    falling = _falling(high.iloc[:2], low.iloc[:2])
    if falling:
        sar = high.iloc[0]
        ep = low.iloc[0]
    else:
        sar = low.iloc[0]
        ep = high.iloc[0]

    if close is not None:
        close = cudf.Series(close)
        sar = close.iloc[0]

    long = cudf.Series(index=high.index)
    short = long.copy()
    reversal = cudf.Series(0, index=high.index)
    _af = long.copy()
    _af.iloc[0:2] = af0

    # Calculate Result
    m = high.shape[0]
    for row in range(1, m):
        high_ = high.iloc[row]
        low_ = low.iloc[row]

        if falling:
            _sar = sar + af * (ep - sar)
            reverse = high_ > _sar

            if low_ < ep:
                ep = low_
                af = min(af + af0, max_af)

            _sar = max(high.iloc[row - 1], high.iloc[row - 2], _sar)
        else:
            _sar = sar + af * (ep - sar)
            reverse = low_ < _sar

            if high_ > ep:
                ep = high_
                af = min(af + af0, max_af)

            _sar = min(low.iloc[row - 1], low.iloc[row - 2], _sar)

        if reverse:
            _sar = ep
            af = af0
            falling = not falling  # Must come before next line
            ep = low_ if falling else high_

        sar = _sar  # Update SAR

        # Seperate long/short sar based on falling
        if falling:
            short.iloc[row] = sar
        else:
            long.iloc[row] = sar

        _af.iloc[row] = af
        reversal.iloc[row] = int(reverse)

    # Offset
    if offset != 0:
        _af = _af.shift(offset)
        long = long.shift(offset)
        short = short.shift(offset)
        reversal = reversal.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        _af.fillna(kwargs["fillna"], inplace=True)
        long.fillna(kwargs["fillna"], inplace=True)
        short.fillna(kwargs["fillna"], inplace=True)
        reversal.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        _af.fillna(method=kwargs["fill_method"], inplace=True)
        long.fillna(method=kwargs["fill_method"], inplace=True)
        short.fillna(method=kwargs["fill_method"], inplace=True)
        reversal.fillna(method=kwargs["fill_method"], inplace=True)

    # Prepare DataFrame to return
    _params = f"_{af0}_{max_af}"
    data = {
        f"PSARl{_params}": long,
        f"PSARs{_params}": short,
        f"PSARaf{_params}": _af,
        f"PSARr{_params}": reversal,
    }
    psardf = cudf.DataFrame(data)
    psardf.name = f"PSAR{_params}"
    psardf.category = long.category = short.category = "trend"

    return psardf