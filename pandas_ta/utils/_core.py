import re as re_
from pathlib import Path
from sys import float_info as sflt
import cudf
from cuml.common.input_utils import get_input_device_type
from cuml.dataloader import Dataset
from cuml.metrics import get_metric
from cudf.dataframe import DataFrame as cuDF
from cudf.series import Series as cuSeries
from cudf.core.index import DatetimeIndex as cuDatetimeIndex
from cudf.core.index import RangeIndex as cuRangeIndex

cudf.import_dask()


def _camelCase2Title(x: str):
    return re_.sub("([a-z])([A-Z])","\g<1> \g<2>", x).title()


def category_files(category: str) -> list:
    files = [
        x.stem
        for x in list(Path(f"pandas_ta/{category}/").glob("*.py"))
        if x.stem != "__init__"
    ]
    return files


def get_drift(x: int) -> int:
    return int(x) if isinstance(x, int) and x != 0 else 1


def get_offset(x: int) -> int:
    return int(x) if isinstance(x, int) else 0


def is_datetime_ordered(df: cuDF or cuSeries) -> bool:
    index_is_datetime = isinstance(df.index, cuDatetimeIndex)
    try:
        ordered = df.index[0] < df.index[-1]
    except RuntimeWarning:
        pass
    finally:
        return True if index_is_datetime and ordered else False


def is_percent(x: int or float) -> bool:
    if isinstance(x, (int, float)):
        return x is not None and x >= 0 and x <= 100
    return False


def non_zero_range(high: cuSeries, low: cuSeries) -> cuSeries:
    diff = high - low
    if diff.eq(0).any().any():
        diff += sflt.epsilon
    return diff


def recent_maximum_index(x):
    return int(argmax(x[::-1]))


def recent_minimum_index(x):
    return int(argmin(x[::-1]))


def signed_series(series: cuSeries, initial: int = None) -> cuSeries:
    series = verify_series(series)
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


def tal_ma(name: str) -> int:
    if Imports["talib"] and isinstance(name, str) and len(name) > 1:
        from talib import MA_Type
        name = name.lower()
        if   name == "sma":   return MA_Type.SMA   # 0
        elif name == "ema":   return MA_Type.EMA   # 1
        elif name == "wma":   return MA_Type.WMA   # 2
        elif name == "dema":  return MA_Type.DEMA  # 3
        elif name == "tema":  return MA_Type.TEMA  # 4
        elif name == "trima": return MA_Type.TRIMA # 5
        elif name == "kama":  return MA_Type.KAMA  # 6
        elif name == "mama":  return MA_Type.MAMA  # 7
        elif name == "t3":    return MA_Type.T3    # 8
    return 0 # Default: SMA -> 0


def unsigned_differences(series: cuSeries, amount: int = None, **kwargs) -> (cuSeries, cuSeries):
    amount = int(amount) if amount is not None else 1
    negative = series.diff(amount)
    negative.fillna(0, inplace=True)
    positive = negative.copy()

    positive[positive <= 0] = 0
    positive[positive > 0] = 1

    negative[negative >= 0] = 0
    negative[negative < 0] = 1

    if kwargs.pop("asint", False):
        positive = positive.astype(int)
        negative = negative.astype(int)

    return positive, negative


def verify_series(series: cuSeries, min_length: int = None) -> cuSeries:
    has_length = min_length is not None and isinstance(min_length, int)
    if series is not None and isinstance(series, cuSeries):
        return None if has_length and series.size < min_length else series
