# -*- coding: utf-8 -*-
import cudf
from cuml.utils import get_cuml_supported_sklearn_versions
from cudf.utils import cuda

from pandas_ta.utils import verify_series
from numba import cuda
import cusignal
import cuml

def pvr(close, volume):
    """ Indicator: Price Volume Rank"""
    # Validate arguments
    close = cudf.Series(close)
    volume = cudf.Series(volume)

    # Calculate Result
    close_diff = close.diff().fillna(0)
    volume_diff = volume.diff().fillna(0)
    pvr_ = cudf.Series(cuda.to_device(close.index), dtype=cudf.float64)
    pvr_[((close_diff >= 0) & (volume_diff >= 0))] = 1
    pvr_[((close_diff >= 0) & (volume_diff < 0))] = 2
    pvr_[((close_diff < 0) & (volume_diff >= 0))] = 3
    pvr_[((close_diff < 0) & (volume_diff < 0))] = 4

    # Name and Categorize it
    pvr_.name = f"PVR"
    pvr_.category = "volume"

    return pvr_

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
    close (cuDF.Series): Series of 'close's
    volume (cuDF.Series): Series of 'volume's

Returns:
    cuDF.Series: New feature generated.
"""