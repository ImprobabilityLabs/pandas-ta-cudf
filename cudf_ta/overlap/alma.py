# -*- coding: utf-8 -*-
import cudf
from numba import cuda
import math
from typing import Tuple

@cuda.jit
def alma_kernel(wtd, close, result, cum_sum, length, sigma, m, s):
    i = cuda.grid(1)
    if i < close.size - length + 1:
        window_sum = 0
        cum_sum = 0
        for j in range(length):
            window_sum += wtd[j] * close[i + j]
            cum_sum += wtd[j]
        almean = window_sum / cum_sum
        result[i + length - 1] = almean

def alma(close: cudf.Series, length: int = 10, sigma: float = 6.0, distribution_offset: float = 0.85, offset: int = 0, **kwargs) -> cudf.Series:
    """Indicator: Arnaud Legoux Moving Average (ALMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    sigma = float(sigma) if sigma and sigma > 0 else 6.0
    distribution_offset = float(distribution_offset) if distribution_offset and distribution_offset > 0 else 0.85
    close = cudf.Series(close)
    offset = offset

    if close is None: return

    # Pre-Calculations
    m = distribution_offset * (length - 1)
    s = length / sigma
    wtd = cudf.Series([math.exp(-1 * ((i - m) * (i - m)) / (2 * s * s)) for i in range(length)])
    result = cudf.Series([float('nan') for _ in range(length - 1)] + [0], index=close.index)

    # Calculate Result
    threadsperblock = 256
    blockspergrid = (close.size + threadsperblock - 1) // threadsperblock
    alma_kernel[blockspergrid, threadsperblock](cuda.to_device(wtd.values), cuda.to_device(close.values), cuda.to_device(result.values), cuda.device_array((close.size - length + 1,), dtype=cudf.Series.dtype), length, sigma, m, s)
    result = cudf.Series(result, index=close.index)

    # Offset
    if offset != 0:
        result = result.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        result.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        result.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    result.name = f"ALMA_{length}_{sigma}_{distribution_offset}"
    result.category = "overlap"

    return result


alma.__doc__ = \
"""Arnaud Legoux Moving Average (ALMA)

The ALMA moving average uses the curve of the Normal (Gauss) distribution, which
can be shifted from 0 to 1. This allows regulating the smoothness and high
sensitivity of the indicator. Sigma is another parameter that is responsible for
the shape of the curve coefficients. This moving average reduces lag of the data
in conjunction with smoothing to reduce noise.

Implemented for CuDF TA by rengel8 based on the source provided below.

Sources:
    https://www.prorealcode.com/prorealtime-indicators/alma-arnaud-legoux-moving-average/

Calculation:
    refer to provided source

Args:
    close (CuDF.Series): Series of 'close's
    length (int): It's period, window size. Default: 10
    sigma (float): Smoothing value. Default 6.0
    distribution_offset (float): Value to offset the distribution min 0
        (smoother), max 1 (more responsive). Default 0.85
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): cudf.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    CuDF.Series: New feature generated.
"""