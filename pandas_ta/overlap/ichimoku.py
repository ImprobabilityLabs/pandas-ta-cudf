import cudf
from cupy import get_default_memory_pool
pool = get_default_memory_pool()

def midprice(high, low, length):
    return (high.rolling(window=length).max() + low.rolling(window=length).min()) / 2

def ichimoku(high, low, close, tenkan=None, kijun=None, senkou=None, include_chikou=True, offset=None, **kwargs):
    """Indicator: Ichimoku Kinkō Hyō (Ichimoku)"""
    tenkan = int(tenkan) if tenkan and tenkan > 0 else 9
    kijun = int(kijun) if kijun and kijun > 0 else 26
    senkou = int(senkou) if senkou and senkou > 0 else 52
    _length = max(tenkan, kijun, senkou)
    high = high.rolling(window=_length).max()
    low = low.rolling(window=_length).min()
    close = close
    offset = get_offset(offset)
    if not kwargs.get("lookahead", True):
        include_chikou = False

    if high is None or low is None or close is None: return None, None

    # Calculate Result
    tenkan_sen = midprice(high, low, length=tenkan)
    kijun_sen = midprice(high, low, length=kijun)
    span_a = 0.5 * (tenkan_sen + kijun_sen)
    span_b = midprice(high, low, length=senkou)

    # Copy Span A and B values before their shift
    _span_a = span_a[-kijun:].copy()
    _span_b = span_b[-kijun:].copy()

    span_a = span_a.shift(kijun)
    span_b = span_b.shift(kijun)
    chikou_span = close.shift(-kijun)

    # Offset
    if offset != 0:
        tenkan_sen = tenkan_sen.shift(offset)
        kijun_sen = kijun_sen.shift(offset)
        span_a = span_a.shift(offset)
        span_b = span_b.shift(offset)
        chikou_span = chikou_span.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        span_a.fillna(kwargs["fillna"], inplace=True)
        span_b.fillna(kwargs["fillna"], inplace=True)
        chikou_span.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        span_a.fillna(method=kwargs["fill_method"], inplace=True)
        span_b.fillna(method=kwargs["fill_method"], inplace=True)
        chikou_span.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    span_a.name = f"ISA_{tenkan}"
    span_b.name = f"ISB_{kijun}"
    tenkan_sen.name = f"ITS_{tenkan}"
    kijun_sen.name = f"IKS_{kijun}"
    chikou_span.name = f"ICS_{kijun}"

    chikou_span.category = kijun_sen.category = tenkan_sen.category = "trend"
    span_b.category = span_a.category = chikou_span

    # Prepare Ichimoku DataFrame
    data = {
        span_a.name: span_a,
        span_b.name: span_b,
        tenkan_sen.name: tenkan_sen,
        kijun_sen.name: kijun_sen,
    }
    if include_chikou:
        data[chikou_span.name] = chikou_span

    ichimokudf = cudf.DataFrame(data)
    ichimokudf.name = f"ICHIMOKU_{tenkan}_{kijun}_{senkou}"
    ichimokudf.category = "overlap"

    # Prepare Span DataFrame
    last = close.index[-1]
    if close.index.dtype == "int64":
        ext_index = cudf.RangeIndex(start=last + 1, stop=last + kijun + 1)
        spandf = cudf.DataFrame(index=ext_index, columns=[span_a.name, span_b.name])
        _span_a.index = _span_b.index = ext_index
    else:
        df_freq = close.index.value_counts().mode()[0]
        tdelta = cudf.Timedelta(df_freq, unit="d")
        new_dt = cudf.date_range(start=last + tdelta, periods=kijun, freq="B")
        spandf = cudf.DataFrame(index=new_dt, columns=[span_a.name, span_b.name])
        _span_a.index = _span_b.index = new_dt

    spandf[span_a.name] = _span_a
    spandf[span_b.name] = _span_b
    spandf.name = f"ICHISPAN_{tenkan}_{kijun}"
    spandf.category = "overlap"

    return ichimokudf, spandf