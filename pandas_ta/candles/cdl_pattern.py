import cudf
from numba import cuda

from . import cdl_doji, cdl_inside
from cudf.utils import get_offset, verify_series
from ....imports import Imports

ALL_PATTERNS = [
    "2crows", "3blackcrows", "3inside", "3linestrike", "3outside", "3starsinsouth",
    "3whitesoldiers", "abandonedbaby", "advanceblock", "belthold", "breakaway",
    "closingmarubozu", "concealbabyswall", "counterattack", "darkcloudcover", "doji",
    "dojistar", "dragonflydoji", "engulfing", "eveningdojistar", "eveningstar",
    "gapsidesidewhite", "gravestonedoji", "hammer", "hangingman", "harami",
    "haramicross", "highwave", "hikkake", "hikkakemod", "homingpigeon",
    "identical3crows", "inneck", "inside", "invertedhammer", "kicking", "kickingbylength",
    "ladderbottom", "longleggeddoji", "longline", "marubozu", "matchinglow", "mathold",
    "morningdojistar", "morningstar", "onneck", "piercing", "rickshawman",
    "risefall3methods", "separatinglines", "shootingstar", "shortline", "spinningtop",
    "stalledpattern", "sticksandwich", "takuri", "tasukigap", "thrusting", "tristar",
    "unique3river", "upsidegap2crows", "xsidegap3methods"
]

def cdl_pattern(open_, high, low, close, name: str="all", scalar=100, offset=0, **kwargs) -> cudf.DataFrame:
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)

    pta_patterns = {
        "doji": cdl_doji, "inside": cdl_inside
    }

    if name == "all":
        name = ALL_PATTERNS
    if type(name) is str:
        name = [name]

    if Imports["talib"]:
        import talib.abstract as tala

    result = {}
    for n in name:
        if n not in ALL_PATTERNS:
            print(f"[X] There is no candle pattern named {n} available!")
            continue

        if n in pta_patterns:
            pattern_result = pta_patterns[n](open_, high, low, close, offset=offset, scalar=scalar, **kwargs)
            result[pattern_result.name] = pattern_result
        else:
            if not Imports["talib"]:
                print(f"[X] Please install TA-Lib to use {n}. (pip install TA-Lib)")
                continue

            pattern_func = tala.Function(f"CDL{n.upper()}")
            pattern_result = cudf.Series(pattern_func(open_, high, low, close, **kwargs) / 100 * scalar)
            pattern_result.index = close.index

            if offset != 0:
                pattern_result = pattern_result.shift(offset)

            if "fillna" in kwargs:
                pattern_result.fillna(kwargs["fillna"], inplace=True)
            if "fill_method" in kwargs:
                pattern_result.fillna(method=kwargs["fill_method"], inplace=True)

            result[f"CDL_{n.upper()}"] = pattern_result

    if len(result) == 0: return

    df = cudf.DataFrame(result)
    df.name = "CDL_PATTERN"
    df.category = "candles"
    return df