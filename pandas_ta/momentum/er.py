import cudf
from cuml.utils import get_cuda_version
from pandas_ta.utils import get_drift, get_offset, verify_series, signals

def er(close, length=None, drift=None, offset=None, **kwargs):
    """Indicator: Efficiency Ratio (ER)"""
    # Validate arguments
    length = int(length) if length and length > 0 else 10
    close = verify_series(close, length)
    offset = get_offset(offset)
    drift = get_drift(drift)

    if close is None: return

    # Calculate Result
    abs_diff = close.diff(length).abs()
    abs_volatility = close.diff(drift).abs()

    er = abs_diff
    er /= abs_volatility.rolling(window=length).sum()

    # Offset
    if offset != 0:
        er = er.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        er.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        er.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    er.name = f"ER_{length}"
    er.category = "momentum"

    signal_indicators = kwargs.pop("signal_indicators", False)
    if signal_indicators:
        signalsdf = cudf.concat(
            [
                cudf.DataFrame({er.name: er}),
                signals(
                    indicator=er,
                    xa=kwargs.pop("xa", 80),
                    xb=kwargs.pop("xb", 20),
                    xserie=kwargs.pop("xserie", None),
                    xserie_a=kwargs.pop("xserie_a", None),
                    xserie_b=kwargs.pop("xserie_b", None),
                    cross_values=kwargs.pop("cross_values", False),
                    cross_series=kwargs.pop("cross_series", True),
                    offset=offset,
                ),
            ],
            axis=1,
        )

        return signalsdf
    else:
        return er