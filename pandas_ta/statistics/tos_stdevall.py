Here is the refactored code that incorporates CuDF and other necessary CUDA-related changes:

```python
# -*- coding: utf-8 -*-
import cudf
from cuml.preprocessing.poly_regression import PolyRegression
from cupy import sqrt
from pandas import DatetimeIndex, Series

def tos_stdevall(close, length=None, stds=None, ddof=None, offset=None, **kwargs):
    # Validate Arguments
    stds = stds if isinstance(stds, list) and len(stds) > 0 else [1, 2, 3]
    if min(stds) <= 0: return
    if not all(i < j for i, j in zip(stds, stds[1:])):
        stds = stds[::-1]
    ddof = int(ddof) if ddof and ddof >= 0 and ddof < length else 1
    offset = get_offset(offset)

    _props = f"TOS_STDEVALL"
    if length is None:
        length = close.size
    else:
        length = int(length) if isinstance(length, int) and length > 2 else 30
        close = close.iloc[-length:]
        _props = f"{_props}_{length}"

    close = verify_series(close, length)

    if close is None: return

    # Calculate Result
    X = src_index = close.index
    if isinstance(close.index, DatetimeIndex):
        X = cudf.Series(arange(length))
        close = cudf.Series(close)

    pr = PolyRegression(degree=1)
    pr.fit(cudf.DataFrame({'X': X}, index=X), cudf.Series(close, index=X))
    m, b = pr.coef_
    lr = cudf.Series(m * X + b, index=src_index)
    stdev = sqrt(((close - lr)**2).mean())

    # Name and Categorize it
    df = cudf.DataFrame({f"{_props}_LR": lr}, index=src_index)
    for i in stds:
        df[f"{_props}_L_{i}"] = lr - i * stdev
        df[f"{_props}_U_{i}"] = lr + i * stdev
        df[f"{_props}_L_{i}"].name = df[f"{_props}_U_{i}"].name = f"{_props}"
        df[f"{_props}_L_{i}"].category = df[f"{_props}_U_{i}"].category = "statistics"

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # Prepare DataFrame to return
    df.name = f"{_props}"
    df.category = "statistics"

    return df
```