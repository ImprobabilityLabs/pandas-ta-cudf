import os
import cudf
from cudf.io.parsers import read_csv

VERBOSE = True

ALERT = f"[!]"
INFO = f"[i]"

CORRELATION = "corr"  # "sem"
CORRELATION_THRESHOLD = 0.99  # Less than 0.99 is undesirable

sample_data = read_csv(
    f"data/SPY_D.csv",
    parse_dates=["date"],
    dayfirst=True,
)
sample_data.set_index("date", inplace=True)

def error_analysis(gdf, kind, msg, icon=INFO, newline=True):
    if VERBOSE:
        s = f"{icon} {gdf.name}['{kind}']: {msg}"
        if newline:
            s = f"\n{s}"
        print(s)