# -*- coding: utf-8 -*-
import cudf
from numba import cuda

@cuda.jit
def pvr_kernel(close, volume, pvr):
    idx = cuda.grid(1)
    if idx < close.size:
        close_diff = close[idx] - close[idx - 1] if idx > 0 else 0
        volume_diff = volume[idx] - volume[idx - 1] if idx > 0 else 0
        if close_diff >= 0 and volume_diff >= 0:
            pvr[idx] = 1
        elif close_diff >= 0 and volume_diff < 0:
            pvr[idx] = 2
        elif close_diff < 0 and volume_diff >= 0:
            pvr[idx] = 3
        else:
            pvr[idx] = 4

def pvr(close, volume):
    """ Indicator: Price Volume Rank"""
    # Validate arguments
    close = cudf.Series(close)
    volume = cudf.Series(volume)

    # Calculate Result
    pvr_ser = cudf.Series(index=close.index)
    pvr_ary = cuda.to_device(pvr_ser.values)

    blockdim = 256
    griddim = (len(close) + blockdim - 1) // blockdim
    pvr_kernel[griddim, blockdim](close.values, volume.values, pvr_ary)
    pvr_ser.values = pvr_ary.copy_to_host()

    # Name and Categorize it
    pvr_ser.name = "PVR"
    pvr_ser.category = "volume"

    return pvr_ser


pvr.__doc__ = \
"""Price Volume Rank

The Price Volume Rank was developed by Anthony J. Macek and is described in his
article in the June, 1994 issue of Technical Analysis of Stocks & Commodities
Magazine. It was developed as a simple indicator that could be calculated even
without a computer. The basic interpretation is to buy when the PV Rank is below
2.5 and sell when it is above 2.5.

Sources:
    https://www.fmlabs.com/reference/default.htm?url=PVrank.htm

Calculation:
    return 1 if 'close change' >= 0 and 'volume change' >= 0
    return 2 if 'close change' >= 0 and 'volume change' < 0
    return 3 if 'close change' < 0 and 'volume change' >= 0
    return 4 if 'close change' < 0 and 'volume change' < 0

Args:
    close (cudf.Series): Series of 'close's
    volume (cudf.Series): Series of 'volume's

Returns:
    cudf.Series: New feature generated.
"""