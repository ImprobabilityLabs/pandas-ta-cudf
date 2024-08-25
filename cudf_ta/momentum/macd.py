# -*- coding: utf-8 -*-
import cudf
from cuml.tsa import ExponentialMovingAverage
from cuml.utils import get_dtype
from cuml.utils.time import to_datetime
import numpy as np
from typing import Optional, Union

def ema(cudf_series: cudf.Series, length: int) -> cudf.Series:
    ema = ExponentialMovingAverage(window=length)
    return ema.fit_transform(cudf_series)

def macd(close: cudf.Series, fast: Optional[Union[int, str]] = 12, slow: Optional[Union[int, str]] = 26, 
         signal: Optional[Union[int, str]] = 9, talib: Optional[bool] = True, offset: Optional[int] = 0, 
         asmode: Optional[bool] = False, **kwargs) -> cudf.DataFrame:
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    if slow < fast:
        fast, slow = slow, fast
    
    close = close.rolling(window=max(fast, slow, signal)).mean().fillna(method='bfill')
    
    if close is None: return

    as_mode = bool(asmode) if isinstance(asmode, bool) else False

    fastma = ema(close, length=fast)
    slowma = ema(close, length=slow)

    macd = fastma - slowma
    signalma = ema(macd, length=signal)
    histogram = macd - signalma

    if as_mode:
        macd = macd - signalma
        signalma = ema(macd, length=signal)
        histogram = macd - signalma

    if offset != 0:
        macd = macd.shift(offset)
        histogram = histogram.shift(offset)
        signalma = signalma.shift(offset)

    if 'fillna' in kwargs:
        macd.fillna(kwargs['fillna'], inplace=True)
        histogram.fillna(kwargs['fillna'], inplace=True)
        signalma.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        macd.fillna(method=kwargs['fill_method'], inplace=True)
        histogram.fillna(method=kwargs['fill_method'], inplace=True)
        signalma.fillna(method=kwargs['fill_method'], inplace=True)

    _asmode = 'AS' if as_mode else ''
    _props = f'_{fast}_{slow}_{signal}'
    macd.name = f'MACD{_asmode}{_props}'
    histogram.name = f'MACD{_asmode}h{_props}'
    signalma.name = f'MACD{_asmode}s{_props}'
    macd.__setattr__('category', 'momentum')
    histogram.__setattr__('category', 'momentum')
    signalma.__setattr__('category', 'momentum')

    data = {macd.name: macd, histogram.name: histogram, signalma.name: signalma}
    df = cudf.DataFrame(data)
    df.name = f'MACD{_asmode}{_props}'
    df.__setattr__('category', 'momentum')

    signal_indicators = kwargs.pop('signal_indicators', False)
    if signal_indicators:
        signalsdf = cudf.concat(
            [
                df,
                signals(
                    indicator=histogram,
                    xa=kwargs.pop('xa', 0),
                    xb=kwargs.pop('xb', None),
                    xserie=kwargs.pop('xserie', None),
                    xserie_a=kwargs.pop('xserie_a', None),
                    xserie_b=kwargs.pop('xserie_b', None),
                    cross_values=kwargs.pop('cross_values', True),
                    cross_series=kwargs.pop('cross_series', True),
                    offset=offset,
                ),
                signals(
                    indicator=macd,
                    xa=kwargs.pop('xa', 0),
                    xb=kwargs.pop('xb', None),
                    xserie=kwargs.pop('xserie', None),
                    xserie_a=kwargs.pop('xserie_a', None),
                    xserie_b=kwargs.pop('xserie_b', None),
                    cross_values=kwargs.pop('cross_values', False),
                    cross_series=kwargs.pop('cross_series', True),
                    offset=offset,
                ),
            ],
            axis=1,
        )

        return signalsdf
    else:
        return df

def signals(indicator, xa=0, xb=None, xserie=None, xserie_a=None, xserie_b=None, cross_values=True, 
            cross_series=True, offset=0):
    pass
