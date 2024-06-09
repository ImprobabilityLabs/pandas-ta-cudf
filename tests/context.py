import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cudf
import pandas_taoverrides = pandas_ta.OVERRIDES
pandas_ta.OVERRIDES = overrides
cudf.core.DataFrame.ta = lambda df, *args, **kwargs: pandas_ta_compute(df, *args, **kwargs)
def pandas_ta_compute(df, *args, **kwargs):
    pdf = df.to_pandas()
    result = getattr(pandas_ta, df.ta.__name__)(*args, **kwargs)(pdf)
    return cudf.from_pandas(result)