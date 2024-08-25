import os
import cudf
from cudf import DatetimeIndex

VERBOSE = True

ALERT = f"[!]"
INFO = f"[i]"

CORRELATION = "corr"  # "sem"
CORRELATION_THRESHOLD = 0.99  # Less than 0.99 is undesirable

sample_data = cudf.read_csv(
    f"data/SPY_D.csv",
    parse_dates=["date"],
    keep_default_na=False,
)
sample_data["date"] = sample_data["date"].astype("datetime64[ns]")
sample_data.set_index("date", inplace=True, drop=True)

def error_analysis(df, kind, msg, icon=INFO, newline=True):
    if VERBOSE:
        s = f"{icon} {df.name}['{kind}']: {msg}"
        if newline:
            s = f"\n{s}"
        print(s)