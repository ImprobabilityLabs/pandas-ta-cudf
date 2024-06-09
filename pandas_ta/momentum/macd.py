import cudf
from cudf.core._compat import PandasCompat
from cudf_ta.overlap import ema
from cudf_ta.utils import get_offset, verify_series, signals

def macd(close, fast=None, slow=None, signal=None, talib=None, offset=None, **kwargs):
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    if slow < fast:
        fast, slow = slow, fast
    close = verify_series(close, max(fast, slow, signal))
    offset = get_offset(offset)
    mode_tal = bool(talib) if isinstance(talib, bool) else True

    if close is None: return

    as_mode = kwargs.setdefault("asmode", False)

    # Calculate Result
    if Imports["talib"] and mode_tal:
        from talib import MACD
        macd, signalma, histogram = MACD(close, fast, slow, signal)
    else:
        fastma = ema(close, length=fast)
        slowma = ema(close, length=slow)

        macd = fastma - slowma
        signalma = ema(close=macd.loc[macd.first_valid_index():,], length=signal)
        histogram = macd - signalma

    if as_mode:
        macd = macd - signalma
        signalma = ema(close=macd.loc[macd.first_valid_index():,], length=signal)
        histogram = macd - signalma

    # Offset
    if offset != 0:
        macd = macd.shift(offset)
        histogram = histogram.shift(offset)
        signalma = signalma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        macd.fillna(kwargs["fillna"], inplace=True)
        histogram.fillna(kwargs["fillna"], inplace=True)
        signalma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        macd.fillna(method=kwargs["fill_method"], inplace=True)
        histogram.fillna(method=kwargs["fill_method"], inplace=True)
        signalma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    _asmode = "AS" if as_mode else ""
    _props = f"_{fast}_{slow}_{signal}"
    macd.name = f"MACD{_asmode}{_props}"
    histogram.name = f"MACD{_asmode}h{_props}"
    signalma.name = f"MACD{_asmode}s{_props}"
    macd.category = histogram.category = signalma.category = "momentum"

    # Prepare DataFrame to return
    data = {macd.name: macd, histogram.name: histogram, signalma.name: signalma}
    df = PandasCompat.cudf_to_pandas(cudf.DataFrame(data))
    df.name = f"MACD{_asmode}{_props}"
    df.category = macd.category

    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        signalsdf = PandasCompat.cudf_to_pandas(cudf.concat(
            [
                df,
                signals(
                    indicator=histogram,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", True),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
                signals(
                    indicator=macd,
                    xa=kwargs.pop("xa", 0),
                    xb=kwargs.pop("xb", None),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        ))

        return signalsdf
    else:
        return df