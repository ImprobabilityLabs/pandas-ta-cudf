# -*- coding: utf-8 -*-
import cudf
from cudf.utils import cuda
from numba import cuda
from numpy import mean
import cupy

def vp(close, volume, width=None, **kwargs):
    """Indicator: Volume Profile (VP)"""
    # Validate arguments
    width = int(width) if width and width > 0 else 10
    close = cudf.Series(if close is None else cudf.Series(close))
    volume = cudf.Series(if volume is None else cudf.Series(volume))
    sort_close = kwargs.pop("sort_close", False)

    if close is None or volume is None: return

    # Setup
    signed_price = cupy.sign(close)
    pos_volume = volume * signed_price * (signed_price > 0)
    pos_volume.name = volume.name
    neg_volume = -volume * signed_price * (signed_price < 0)
    neg_volume.name = volume.name
    vp = cudf.concat([close, pos_volume, neg_volume], axis=1)

    close_col = f"{vp.columns[0]}"
    high_price_col = f"high_{close_col}"
    low_price_col = f"low_{close_col}"
    mean_price_col = f"mean_{close_col}"

    volume_col = f"{vp.columns[1]}"
    pos_volume_col = f"pos_{volume_col}"
    neg_volume_col = f"neg_{volume_col}"
    total_volume_col = f"total_{volume_col}"
    vp.columns = [close_col, pos_volume_col, neg_volume_col]

    # sort_close: Sort by close before splitting into ranges. Default: False
    # If False, it sorts by date index or chronological versus by price

    if sort_close:
        vp[mean_price_col] = vp[close_col]
        vpdf = vp.groupby(cudf.cut(vp[close_col], width, include_lowest=True, precision=2)).agg({
            mean_price_col: mean,
            pos_volume_col: 'sum',
            neg_volume_col: 'sum',
        })
        vpdf[low_price_col] = [x.left for x in vpdf.index]
        vpdf[high_price_col] = [x.right for x in vpdf.index]
        vpdf = vpdf.reset_index(drop=True)
        vpdf = vpdf[[low_price_col, mean_price_col, high_price_col, pos_volume_col, neg_volume_col]]
    else:
        vp_ranges = cudf.core.reshape.split(vp, width)
        result = ({
            low_price_col: r[close_col].min(),
            mean_price_col: r[close_col].mean(),
            high_price_col: r[close_col].max(),
            pos_volume_col: r[pos_volume_col].sum(),
            neg_volume_col: r[neg_volume_col].sum(),
        } for r in vp_ranges)
        vpdf = cudf.DataFrame(result)
    vpdf[total_volume_col] = vpdf[pos_volume_col] + vpdf[neg_volume_col]

    # Handle fills
    if "fillna" in kwargs:
        vpdf.fillna(kwargs["fillna"], inplace=True)
    if "fill_method" in kwargs:
        vpdf.fillna(method=kwargs["fill_method"], inplace=True)

    # Name and Categorize it
    vpdf.name = f"VP_{width}"
    vpdf.category = "volume"

    return vpdf