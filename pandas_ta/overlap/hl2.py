from cudf import Series
import cudf
from pandas_ta.utils import get_offset, verify_series

def hl2(high, low, offset=None, **kwargs):
    """Indicator: HL2 """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    offset = get_offset(offset)

    # Calculate Result
    if isinstance(high, cudf.Series) and isinstance(low, cudf.Series):
        hl2 = 0.5 * (high + low)
    else:
        hl2 = 0.5 * (high.to_cudf() + low.to_cudf())

    # Offset
    if offset != 0:
        hl2 = hl2.shift(offset)

    # Name & Category
    hl2.name = "HL2"
    hl2.category = "overlap"

    return hl2