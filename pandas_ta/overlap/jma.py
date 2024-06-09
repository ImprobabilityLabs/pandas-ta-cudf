from cudf import Series
import cusignal 
from cucim import cpu_enabled

@cpu_enabled()
def jma(close, length=None, phase=None, offset=None, **kwargs):
    """Indicator: Jurik Moving Average (JMA)"""
    # Validate Arguments
    _length = int(length) if length and length > 0 else 7
    phase = float(phase) if phase and phase != 0 else 0
    close = verify_series(close, _length)
    offset = get_offset(offset)
    if close is None: return

    # Define base variables
    jma = cusignal.kernels.zeros_like(close)
    volty = cusignal.kernels.zeros_like(close)
    v_sum = cusignal.kernels.zeros_like(close)

    kv = det0 = det1 = ma2 = 0.0
    jma[0] = ma1 = uBand = lBand = close[0]

    # Static variables
    sum_length = 10
    length = 0.5 * (_length - 1)
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else 1.5 + phase * 0.01
    length1 = max((cusignal.kernels.log(cusignal.kernels.sqrt(length)) / cusignal.kernels.log(2.0)) + 2.0, 0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * cusignal.kernels.sqrt(length)
    bet = length2 / (length2 + 1)
    beta = 0.45 * (_length - 1) / (0.45 * (_length - 1) + 2.0)

    m = close.shape[0]
    for i in range(1, m):
        price = close[i]

        # Price volatility
        del1 = price - uBand
        del2 = price - lBand
        volty[i] = cusignal.kernels.maximum(cusignal.kernels.abs(del1), cusignal.kernels.abs(del2)) if cusignal.kernels.abs(del1)!=cusignal.kernels.abs(del2) else 0

        # Relative price volatility factor
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[max(i - sum_length, 0)]) / sum_length
        avg_volty = cusignal.kernels.average(v_sum[max(i - 65, 0):i + 1])
        d_volty = 0 if avg_volty ==0 else volty[i] / avg_volty
        r_volty = cusignal.kernels.maximum(1.0, cusignal.kernels.minimum(cusignal.kernels.power(length1, 1 / pow1), d_volty))

        # Jurik volatility bands
        pow2 = cusignal.kernels.power(r_volty, pow1)
        kv = cusignal.kernels.power(bet, cusignal.kernels.sqrt(pow2))
        uBand = price if (del1 > 0) else price - (kv * del1)
        lBand = price if (del2 < 0) else price - (kv * del2)

        # Jurik Dynamic Factor
        power = cusignal.kernels.power(r_volty, pow1)
        alpha = cusignal.kernels.power(beta, power)

        # 1st stage - prelimimary smoothing by adaptive EMA
        ma1 = ((1 - alpha) * price) + (alpha * ma1)

        # 2nd stage - one more prelimimary smoothing by Kalman filter
        det0 = ((price - ma1) * (1 - beta)) + (beta * det0)
        ma2 = ma1 + pr * det0

        # 3rd stage - final smoothing by unique Jurik adaptive filter
        det1 = ((ma2 - jma[i - 1]) * (1 - alpha) * (1 - alpha)) + (alpha * alpha * det1)
        jma[i] = jma[i-1] + det1

    # Remove initial lookback data and convert to pandas frame
    jma[0:_length - 1] = cusignal.kernels.nan
    jma = Series(jma)

    # Offset
    if offset != 0:
        jma = jma.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        jma.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        jma.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    jma.name = f"JMA_{_length}_{phase}"
    jma.category = "overlap"

    return jma