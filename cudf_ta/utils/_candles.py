# -*- coding: utf-8 -*-
import cudf
from ._core import non_zero_range

def candle_color(open_: cudf.Series, close: cudf.Series) -> cudf.Series:
    color = close.copy().astype(int)
    color[close >= open_] = 1
    color[close < open_] = -1
    return color


def high_low_range(high: cudf.Series, low: cudf.Series) -> cudf.Series:
    return non_zero_range(high, low)


def real_body(open_: cudf.Series, close: cudf.Series) -> cudf.Series:
    return non_zero_range(close, open_)