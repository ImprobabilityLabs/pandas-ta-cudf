import cudf
from cuml.metrics import sqrt as npSqrt
from cudf_ta.utils import get_offset, verify_series

def hwc(close, na=None, nb=None, nc=None, nd=None, scalar=None, channel_eval=None, offset=None, **kwargs):
    # Validate Arguments
    na = float(na) if na and na > 0 else 0.2
    nb = float(nb) if nb and nb > 0 else 0.1
    nc = float(nc) if nc and nc > 0 else 0.1
    nd = float(nd) if nd and nd > 0 else 0.1
    scalar = float(scalar) if scalar and scalar > 0 else 1
    channel_eval = bool(channel_eval) if channel_eval and channel_eval else False
    close = verify_series(close)
    offset = get_offset(offset)

    # Calculate Result
    last_a = last_v = last_var = 0
    last_f = last_price = last_result = close[0]
    lower, result, upper = [], [], []
    chan_pct_width, chan_width = [], []

    m = len(close)
    for i in range(m):
        F = (1.0 - na) * (last_f + last_v + 0.5 * last_a) + na * close[i]
        V = (1.0 - nb) * (last_v + last_a) + nb * (F - last_f)
        A = (1.0 - nc) * last_a + nc * (V - last_v)
        result.append((F + V + 0.5 * A))

        var = (1.0 - nd) * last_var + nd * (last_price - last_result) * (last_price - last_result)
        stddev = npSqrt(last_var)
        upper.append(result[i] + scalar * stddev)
        lower.append(result[i] - scalar * stddev)

        if channel_eval:
            # channel width
            chan_width.append(upper[i] - lower[i])
            # channel percentage price position
            chan_pct_width.append((close[i] - lower[i]) / (upper[i] - lower[i]))
            # print('channel_eval (width|percentageWidth):', chan_width[i], chan_pct_width[i])

        # update values
        last_price = close[i]
        last_a = A
        last_f = F
        last_v = V
        last_var = var
        last_result = result[i]

    # Aggregate
    hwc_df = cudf.DataFrame(
        {
            "result": result,
            "upper": upper,
            "lower": lower
        },
        index=close.index
    )

    if channel_eval:
        hwc_df['chan_width'] = chan_width
        hwc_df['chan_pct_width'] = chan_pct_width

    # Offset
    if offset != 0:
        hwc_df = hwc_df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        hwc_df.fillna(kwargs["fillna"], inplace=True)

    if "fill_method" in kwargs:
        hwc_df.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    hwc_df.name = "HWC"
    hwc_df.columns = ["HWM", "HWU", "HWL"]
    if channel_eval:
        hwc_df.columns = ["HWM", "HWU", "HWL", "HWW", "HWPCT"]

    return hwc_df