import cupy as cp
import cudf
from cuml.metrics import getcuml
from cuml.preprocessing.data import CumlArray
from .tsignals import tsignals

def xsignals(signal, xa, xb, above:bool=True, long:bool=True, asbool:bool=None, trend_reset:int=0, trade_offset:int=None, offset:int=None, **kwargs):
    signal = cudf.from_pandas(verify_series(signal))
    offset = get_offset(offset)

    if above:
        entries = cross_value(signal, xa)
        exits = -cross_value(signal, xb, above=False)
    else:
        entries = cross_value(signal, xa, above=False)
        exits = -cross_value(signal, xb)
    trades = entries + exits

    trades.replace({0: cp.nan}, inplace=True)
    trades.interpolate(method="pad", inplace=True)
    trades.fillna(0, inplace=True)

    trends = (trades > 0).astype(int)
    if not long:
        trends = 1 - trends

    tskwargs = {
        "asbool":asbool,
        "trade_offset":trade_offset,
        "trend_reset":trend_reset,
        "offset":offset
    }
    df = tsignals(trends, **tskwargs)

    cuml_array = CumlArray(df.values, dtype=cp.float32)
    df = cudf.DataFrame(cuml_array, columns=[f"XS_LONG", f"XS_SHORT"])

    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    df.name = f"XS"
    df.category = "trend"

    return df