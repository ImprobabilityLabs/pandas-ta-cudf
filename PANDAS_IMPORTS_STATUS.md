# Pandas Imports Conversion Status

## Summary
There are **58 files** that still have pandas imports that need to be converted to cudf.

## Files Fixed ✅

### Recently Fixed:
- ✅ `pandas_ta/volume/pvr.py` - Changed to `from cudf import Series`
- ✅ `pandas_ta/volume/mfi.py` - Changed to `from cudf import DataFrame` + fixed fillna
- ✅ `pandas_ta/volume/aobv.py` - Changed to `from cudf import DataFrame` + fixed fillna
- ✅ `pandas_ta/momentum/er.py` - Changed to `from cudf import DataFrame, concat` + fixed fillna
- ✅ `pandas_ta/momentum/rsx.py` - Changed to `from cudf import concat, DataFrame, Series`
- ✅ `pandas_ta/overlap/ema.py` - Fixed fillna issues

### Previously Fixed:
- ✅ `pandas_ta/core.py`
- ✅ `pandas_ta/volume/vp.py` (uses pandas cut - intentional)
- ✅ `pandas_ta/volume/pvt.py`
- ✅ `pandas_ta/volume/pvol.py`
- ✅ `pandas_ta/volume/nvi.py`
- ✅ `pandas_ta/volume/kvo.py`
- ✅ `pandas_ta/volatility/true_range.py`
- ✅ `pandas_ta/momentum/macd.py`
- ✅ `pandas_ta/momentum/rsi.py`
- ✅ `pandas_ta/overlap/sma.py`
- ✅ `pandas_ta/volatility/bbands.py`

## Files That Should Keep Pandas ⚠️

These files intentionally use pandas:
- `pandas_ta/utils/data/yahoofinance.py` - Data source API
- `pandas_ta/utils/data/alphavantage.py` - Data source API
- `pandas_ta/utils/_time.py` - Uses pandas Timestamp for compatibility
- `pandas_ta/volume/vp.py` - Uses pandas `cut()` function (not in cudf)

## Remaining Files to Fix (58 files)

### Volatility (6 files):
- `pandas_ta/volatility/thermo.py` - `from pandas import DataFrame`
- `pandas_ta/volatility/kc.py` - `from pandas import DataFrame`
- `pandas_ta/volatility/hwc.py` - `from pandas import DataFrame, Series`
- `pandas_ta/volatility/donchian.py` - `from pandas import DataFrame`
- `pandas_ta/volatility/accbands.py` - `from pandas import DataFrame`
- `pandas_ta/volatility/aberration.py` - `from pandas import DataFrame`

### Trend (10 files):
- `pandas_ta/trend/xsignals.py` - `from pandas import DataFrame`
- `pandas_ta/trend/ttm_trend.py` - `from pandas import DataFrame`
- `pandas_ta/trend/vortex.py` - `from pandas import DataFrame`
- `pandas_ta/trend/tsignals.py` - `from pandas import DataFrame`
- `pandas_ta/trend/psar.py` - `from pandas import DataFrame, Series`
- `pandas_ta/trend/decay.py` - `from pandas import DataFrame`
- `pandas_ta/trend/cksp.py` - `from pandas import DataFrame`
- `pandas_ta/trend/aroon.py` - `from pandas import DataFrame`
- `pandas_ta/trend/adx.py` - `from pandas import DataFrame`
- `pandas_ta/trend/amat.py` - `from pandas import DataFrame`

### Overlap (12 files):
- `pandas_ta/overlap/wma.py` - `from pandas import Series`
- `pandas_ta/overlap/vidya.py` - `from pandas import Series`
- `pandas_ta/overlap/supertrend.py` - `from pandas import DataFrame`
- `pandas_ta/overlap/sinwma.py` - `from pandas import Series`
- `pandas_ta/overlap/ma.py` - `from pandas import Series`
- `pandas_ta/overlap/linreg.py` - `from pandas import Series`
- `pandas_ta/overlap/jma.py` - `from pandas import Series`
- `pandas_ta/overlap/kama.py` - `from pandas import Series`
- `pandas_ta/overlap/hwma.py` - `from pandas import Series`
- `pandas_ta/overlap/hilo.py` - `from pandas import DataFrame, Series`
- `pandas_ta/overlap/alma.py` - `from pandas import Series`
- `pandas_ta/overlap/ichimoku.py` - `from pandas import date_range, DataFrame, RangeIndex, Timedelta` (special case)

### Momentum (18 files):
- `pandas_ta/momentum/uo.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/tsi.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/trix.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/td_seq.py` - `from pandas import DataFrame, Series`
- `pandas_ta/momentum/stc.py` - `from pandas import DataFrame, Series`
- `pandas_ta/momentum/stochrsi.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/stoch.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/squeeze_pro.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/squeeze.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/rvgi.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/smi.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/qqe.py` - `from pandas import DataFrame, Series`
- `pandas_ta/momentum/pvo.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/ppo.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/kdj.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/kst.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/fisher.py` - `from pandas import DataFrame, Series`
- `pandas_ta/momentum/cti.py` - `from pandas import Series`
- `pandas_ta/momentum/brar.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/dm.py` - `from pandas import DataFrame`
- `pandas_ta/momentum/eri.py` - `from pandas import DataFrame`

### Other Categories:
- `pandas_ta/statistics/tos_stdevall.py` - `from pandas import DataFrame, DatetimeIndex, Series` (special case)
- `pandas_ta/performance/drawdown.py` - `from pandas import DataFrame`
- `pandas_ta/cycles/ebsw.py` - `from pandas import Series`
- `pandas_ta/candles/ha.py` - `from pandas import DataFrame`
- `pandas_ta/candles/cdl_z.py` - `from pandas import DataFrame`
- `pandas_ta/candles/cdl_pattern.py` - `from pandas import Series, DataFrame`
- `pandas_ta/utils/_metrics.py` - `from pandas import Series, Timedelta` (special case)
- `pandas_ta/utils/_candles.py` - `from pandas import Series`

## How to Fix

### Option 1: Run the Simple Fix Script
```bash
python fix_pandas_imports_simple.py
```

This will automatically fix most simple cases like:
- `from pandas import DataFrame` → `from cudf import DataFrame`
- `from pandas import Series` → `from cudf import Series`
- `from pandas import concat` → `from cudf import concat`
- `from pandas import DataFrame, Series` → `from cudf import DataFrame, Series`

### Option 2: Run the Comprehensive Fix Script
```bash
python fix_all_cudf_issues.py
```

This will fix:
- Pandas imports
- fillna inplace issues
- fillna method parameter issues
- Docstring updates

### Option 3: Manual Fix
For each file, replace:
```python
# Before
from pandas import DataFrame
from pandas import Series
from pandas import DataFrame, Series
from pandas import concat

# After
from cudf import DataFrame
from cudf import Series
from cudf import DataFrame, Series
from cudf import concat
```

## Special Cases

### Files with Complex Imports:
1. **`pandas_ta/overlap/ichimoku.py`**
   - Has: `from pandas import date_range, DataFrame, RangeIndex, Timedelta`
   - Needs: Keep pandas for `date_range`, `RangeIndex`, `Timedelta` (not in cudf)
   - Fix: `from cudf import DataFrame` + `from pandas import date_range, RangeIndex, Timedelta`

2. **`pandas_ta/statistics/tos_stdevall.py`**
   - Has: `from pandas import DataFrame, DatetimeIndex, Series`
   - Needs: Keep pandas for `DatetimeIndex` (not in cudf)
   - Fix: `from cudf import DataFrame, Series` + `from pandas import DatetimeIndex`

3. **`pandas_ta/utils/_metrics.py`**
   - Has: `from pandas import Series, Timedelta`
   - Needs: Keep pandas for `Timedelta` (not in cudf)
   - Fix: `from cudf import Series` + `from pandas import Timedelta`

## After Fixing

1. Run the test script:
   ```bash
   python test_cudf_comprehensive.py
   ```

2. Check for remaining issues:
   ```bash
   python -c "from pathlib import Path; import re; files = [f for f in Path('pandas_ta').rglob('*.py') if '__pycache__' not in str(f) and not any(x in str(f) for x in ['yahoofinance', 'alphavantage', '_time.py', 'vp.py']); imports = [f for f in files if re.search(r'^from pandas import (DataFrame|Series|concat)', open(f).read(), re.MULTILINE)]; print(f'Remaining: {len(imports)} files')"
   ```

## Progress

- **Fixed:** ~15 files
- **Remaining:** ~58 files (excluding intentional pandas usage)
- **Progress:** ~20% complete

Most remaining files are simple cases that can be fixed automatically with the scripts.
