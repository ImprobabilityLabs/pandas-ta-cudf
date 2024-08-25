# -*- coding: utf-8 -*-
import cusignal
import cupy as cp
import cudf

from cudf_ta.utils import fibonacci, get_offset, verify_series, weights

def fwma(close, length=None, asc=None, offset=None, **kwargs):
    """Indicator: Fibonacci's Weighted Moving Average (FWMA)"""
    # Validate Arguments
    length = int(length) if length and length > 0 else 10
    asc = asc if asc else True
    close = verify_series(cudf.Series(close), length)
    offset = get_offset(offset)

    if close is None: return

    # Calculate Result
    fibs = fibonacci(n=length, weighted=True)
    fwma = cusignal.window.weighted-moving-average(close, window=fibs)

    # Offset
    if offset != 0:
        fwma = fwma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        fwma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        fwma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    fwma.name = f"FWMA_{length}"
    fwma.category = "overlap"

    return fwma


fwma.__doc__ = \
"""Fibonacci's Weighted Moving Average (FWMA)

Fibonacci's Weighted Moving Average is similar to a Weighted Moving Average
(WMA) where the weights are based on the Fibonacci Sequence.

Source: Kevin Johnson

Calculation:
    Default Inputs:
        length=10,

    def weights(w):
        def _compute(x):
            return cp.dot(w * x)
        return _compute

    fibs = utils.fibonacci(length - 1)
    FWMA = close.rolling(length)_.apply(weights(fibs), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period. Default: 10
    asc (bool): Recent values weigh more. Default: True
    offset (int): How many periods to offset the result. Default: 0

Kwargs:
    fillna (value, optional): cudf.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""