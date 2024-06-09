from functools import reduce


from math import floor as mfloor


from operator import mul


from sys import float_info as sflt


from typing import List, Optional, Tuple


from numpy import ones, triu


from numpy import all as npAll


from numpy import append as npAppend


from numpy import array as npArray


from numpy import corrcoef as npCorrcoef


from numpy import dot as npDot


from numpy import fabs as npFabs


from numpy import exp as npExp


from numpy import log as npLog


from numpy import nan as npNaN


from numpy import ndarray as npNdArray


from numpy import seterr


from numpy import sqrt as npSqrt


from numpy import sum as npSum


from pandas import DataFrame, Series


from pandas_ta import Imports


from ._core import verify_series


import cupy as cp
import cudf

def combination(**kwargs: dict) ->int:
    """https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python"""
    n = int(cp.abs(kwargs.pop('n', 1)))
    r = int(cp.abs(kwargs.pop('r', 0)))
    if kwargs.pop('repetition', False) or kwargs.pop('multichoose', False):
        n = n + r - 1
    r = min(n, n - r)
    if r == 0:
        return 1
    numerator = cp.prod(cp.arange(n, n - r, -1) + 1)
    denominator = cp.prod(cp.arange(1, r + 1))
    return int(numerator // denominator)

import cudf
import cupy as cp

def npExp(x):
    return cp.exp(x)

def erf(x):
    """Error Function erf(x)
    The algorithm comes from Handbook of Mathematical Functions, formula 7.1.26.
    Source: https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
    """
    sign = cp.where(x >= 0, 1, -1)
    x = cp.abs(x)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * npExp(-x * x)
    return sign * y

import cudf
import cupy as cp
import numpy as np

def fibonacci(n: int=2, **kwargs: dict) -> cp.ndarray:
    """Fibonacci Sequence as a cupy array"""
    n = int(cp絶abs(n)) if n >= 0 else 2
    zero = kwargs.pop('zero', False)
    if zero:
        a, b = 0, 1
    else:
        n -= 1
        a, b = 1, 1
    result = cp.array([a])
    for _ in range(0, n):
        a, b = b, a + b
        result = cp.concatenate((result, cp.array([a])))
    weighted = kwargs.pop('weighted', False)
    if weighted:
        fib_sum = cp.sum(result)
        if fib_sum > 0:
            return result / fib_sum
        else:
            return result
    else:
        return result

import cudf
import cupy as cp
import numpy as np

def geometric_mean(series: cudf.Series) -> float:
    """Returns the Geometric Mean for a Series of positive values."""
    n = series.size
    if n < 1:
        return series.iloc[0]
    has_zeros = 0 in series.values
    if has_zeros:
        series = series.fillna(0) + 1
    if cp.all(series > 0):
        mean = cp.prod(cp.array(series)) ** (1 / n)
        return mean if not has_zeros else mean - 1
    return 0

import cudf
import cuml
from cuml.linear_model import Lasso

def linear_regression(x: cudf.Series, y: cudf.Series) ->dict:
    """Classic Linear Regression in CuML"""
    x, y = verify_series(x), verify_series(y)
    m, n = x.size, y.size
    if m != n:
        print(
            f'[X] Linear Regression X and y have unequal total observations: {m} != {n}'
            )
        return {}
    clf = Lasso()
    clf.fit(x.to_pandas(), y.to_pandas())
    coefficients = clf.coef_
    intercept = clf.intercept_
    return {'coefficients': coefficients, 'intercept': intercept}

import cudf
import cupy as cp
import numpy as np

def log_geometric_mean(series: cudf.Series) -> float:
    """Returns the Logarithmic Geometric Mean"""
    n = series.size
    if n < 2:
        return 0
    else:
        series = series.fillna(0) + 1
        if cp.all(series > 0):
            return cp.exp(cp.log(series).sum() / n) - 1
        return 0

import cudf
import cupy as cp
import numpy as np

def pascals_triangle(n: int=None, **kwargs: dict) -> cp.ndarray:
    """Pascal's Triangle

    Returns a cupy array of the nth row of Pascal's Triangle.
    n=4  => triangle: [1, 4, 6, 4, 1]
         => weighted: [0.0625, 0.25, 0.375, 0.25, 0.0625]
         => inverse weighted: [0.9375, 0.75, 0.625, 0.75, 0.9375]
    """
    n = int(abs(n)) if n is not None else 0
    triangle = cp.array([cp.math.comb(n, i, exact=True) for i in range(0, n + 1)])
    triangle_sum = cp.sum(triangle)
    triangle_weights = triangle / triangle_sum
    inverse_weights = 1 - triangle_weights
    weighted = kwargs.pop('weighted', False)
    inverse = kwargs.pop('inverse', False)
    if weighted and inverse:
        return inverse_weights
    if weighted:
        return triangle_weights
    if inverse:
        return None
    return triangle

import cudf
import cuml
import cupy as cp
import math as m

def symmetric_triangle(n: int=None, **kwargs: dict) ->Optional[list]:
    """Symmetric Triangle with n >= 2

    Returns a cuDF Series of the nth row of Symmetric Triangle.
    n=4  => triangle: [1, 2, 2, 1]
         => weighted: [0.16666667 0.33333333 0.33333333 0.16666667]
    """
    n = int(m.fabs(n)) if n is not None else 2
    triangle = None
    if n == 2:
        triangle = [1, 1]
    if n > 2:
        if n % 2 == 0:
            front = [(i + 1) for i in range(0, m.floor(n / 2))]
            triangle = front + front[::-1]
        else:
            front = [(i + 1) for i in range(0, m.floor(0.5 * (n + 1)))]
            triangle = front.copy()
            front.pop()
            triangle += front[::-1]
    if kwargs.pop('weighted', False) and isinstance(triangle, list):
        triangle_sum = cp.sum(triangle)
        triangle_weights = cp.array(triangle) / triangle_sum
        return cudf.Series(triangle_weights)
    return cudf.Series(triangle)

import cudf
import cupy as cp

def weights(w: cudf.Series):
    """Calculates the dot product of weights with values x"""

    def _dot(x):
        return cp.dot(w.values, x)
    return _dot

import cudf
import numpy as np

sflt = np.finfo(np.float32)

def zero(x: tuple) -> tuple:
    """If the value is close to zero, then return zero. Otherwise return itself."""
    x = cudf.Series([x[0], x[1]])
    x = x.applymap(lambda x: 0 if abs(x) < sflt.epsilon else x)
    return tuple(x.values.tolist())

import cudf
import cupy

def df_error_analysis(dfA: cudf.DataFrame, dfB: cudf.DataFrame, **kwargs: dict
) -> cudf.DataFrame:
    """DataFrame Correlation Analysis helper"""
    corr_method = kwargs.pop('corr_method', 'pearson')
    diff = dfA - dfB
    corr = cudf.core.correlation._correlation(dfA, dfB, method=corr_method)
    if kwargs.pop('plot', False):
        diff.gpu_histogram()
        if diff[diff > 0].any():
            diff.plot(kind='density')
    if kwargs.pop('triangular', False):
        return corr.where(cupy.triu(cupy.ones(corr.shape)).astype(bool))
    return corr

import cudf
import cupy as cp
import numpy as np

def _linear_regression_cu(x: cudf.Series, y: cudf.Series) -> dict:
    """Simple Linear Regression in CuPy for two 1d arrays for environments with GPU."""
    result = {'a': np.nan, 'b': np.nan, 'r': np.nan, 't': np.nan, 'line': np.nan}
    x_sum = x.sum()
    y_sum = y.sum()
    if int(x_sum) != 0:
        r = cp.corrcoef(x, y)[0, 1]
        m = x.size
        r_mix = m * (x * y).sum() - x_sum * y_sum
        b = r_mix // (m * (x * x).sum() - x_sum * x_sum)
        a = y.mean() - b * x.mean()
        line = a + b * x
        result = {'a': a, 'b': b, 'r': r, 't': r / cp.sqrt((1 - r * r) / (m - 2)), 'line': line}
    return result

def _linear_regression_sklearn(x: Series, y: Series) ->dict:
    """Simple Linear Regression in Scikit Learn for two 1d arrays for
    environments with the sklearn package."""
    import cudf
    from sklearn.linear_model import LinearRegression
    X = cudf.DataFrame({'x': x})
    X_np = X.to_pandas()
    y = y.to_pandas()
    lr = LinearRegression().fit(X_np, y=y)
    r = lr.score(X_np, y=y)
    a, b = lr.intercept_, lr.coef_[0]
    result = {'a': a, 'b': b, 'r': r, 't': r / np.sqrt((1 - r * r) / (x.size - 2)), 'line': a + b * x}
    return result