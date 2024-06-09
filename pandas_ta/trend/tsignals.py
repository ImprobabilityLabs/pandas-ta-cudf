import cudf
from cuml.utils import get_drift, get_offset, verify_series

def tsignals(trend, asbool=None, trend_reset=0, trade_offset=None, drift=None, offset=None, **kwargs):
    """Indicator: Trend Signals"""
    # Validate Arguments
    trend = verify_series(trend)
    asbool = bool(asbool) if isinstance(asbool, bool) else False
    trend_reset = int(trend_reset) if trend_reset and isinstance(trend_reset, int) else 0
    if trade_offset !=0:
        trade_offset = int(trade_offset) if trade_offset and isinstance(trade_offset, int) else 0
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    trends = trend.astype(int)
    trades = trends.diff(drift).shift(trade_offset).fillna(0).astype(int)
    entries = (trades > 0).astype(int)
    exits = (trades < 0).abs().astype(int)

    if asbool:
        trends = trends.astype(bool)
        entries = entries.astype(bool)
        exits = exits.astype(bool)

    data = {
        f"TS_Trends": trends,
        f"TS_Trades": trades,
        f"TS_Entries": entries,
        f"TS_Exits": exits,
    }
    df = cudf.DataFrame(data, index=trends.index)

    # Offset
    if offset != 0:
        df = df.shift(offset)

    # Handle fills
    if "fillna" in kwargs:
        df.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        df.fillna(method=kwargs["fill_method"], inplace=True)

    # Name & Category
    df.name = f"TS"
    df.category = "trend"

    return df