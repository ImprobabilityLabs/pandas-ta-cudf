# Testing Guide for pandas-ta-cudf

This guide explains how to test the cuDF conversion of pandas-ta.

## Test Scripts

### 1. `test_cudf_comprehensive.py` (Recommended)
**Comprehensive test script that checks everything:**
- Basic imports and setup
- DataFrame creation and accessor
- Multiple indicators across all categories
- Append functionality
- Strategy functionality
- Remaining conversion issues
- Sample data testing

**Usage:**
```bash
python test_cudf_comprehensive.py
```

### 2. `test_cudf_conversion.py` (Basic)
**Simpler test script for quick checks:**
- Basic imports
- DataFrame accessor
- A few basic indicators
- Append functionality
- Issue detection

**Usage:**
```bash
python test_cudf_conversion.py
```

## Batch Fix Script

### `fix_all_cudf_issues.py`
**Automatically fixes common cuDF conversion issues:**
- Replaces pandas imports with cudf
- Fixes fillna inplace issues
- Fixes fillna method parameter issues
- Updates docstrings

**Usage:**
```bash
python fix_all_cudf_issues.py
```

**Note:** This script will modify files. Make sure to:
1. Commit your current work
2. Review changes after running
3. Test thoroughly

## What to Test

### 1. Basic Functionality
```python
import cudf
import pandas_ta as ta

# Create a cuDF DataFrame
df = cudf.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [99, 100, 101, 102, 103],
    'close': [104, 105, 106, 107, 108],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Test accessor
df.ta  # Should work

# Test indicators
df.ta.sma(length=10)
df.ta.rsi(length=14)
df.ta.macd()
```

### 2. Append Functionality
```python
# Append indicators to DataFrame
df.ta.sma(length=10, append=True)
df.ta.rsi(length=14, append=True)

# Check new columns
print(df.columns)
```

### 3. Strategy
```python
# Custom strategy
strategy = ta.Strategy(
    name="My Strategy",
    ta=[
        {"kind": "sma", "length": 10},
        {"kind": "rsi", "length": 14},
    ]
)

df.ta.strategy(strategy)
```

### 4. With Real Data
```python
# Load CSV with cuDF
df = cudf.read_csv('your_data.csv')

# Lowercase column names if needed
df.columns = df.columns.str.lower()

# Use indicators
df.ta.sma(length=20, append=True)
```

## Common Issues

### Issue: "cuDF not installed"
**Solution:** Install cuDF
```bash
conda install -c rapidsai -c conda-forge cudf
```

### Issue: "fillna inplace error"
**Solution:** Run the batch fix script
```bash
python fix_all_cudf_issues.py
```

### Issue: "pandas import error"
**Solution:** Check if file should use pandas (data sources) or run batch fix
```bash
python fix_all_cudf_issues.py
```

## Expected Results

### Successful Test Output
- ✓ All imports work
- ✓ DataFrame accessor works
- ✓ Indicators return results
- ✓ Append adds columns
- ✓ No fillna inplace issues
- ✓ No unexpected pandas imports

### Warnings (OK)
- Some files may legitimately use pandas (data sources)
- Some fill_method parameters may be commented out (cudf doesn't support)

## Next Steps After Testing

1. **If tests pass:**
   - Test with your actual data
   - Verify GPU memory usage
   - Run performance benchmarks

2. **If tests fail:**
   - Review error messages
   - Run batch fix script if needed
   - Check for remaining pandas imports
   - Fix fillna issues manually if needed

## Performance Testing

```python
import time
import cudf
import pandas_ta as ta

# Create large dataset
n = 100000
df = cudf.DataFrame({
    'open': range(n),
    'high': range(n),
    'low': range(n),
    'close': range(n),
    'volume': range(n)
})

# Time indicator calculation
start = time.time()
result = df.ta.sma(length=20)
elapsed = time.time() - start
print(f"SMA calculation: {elapsed:.4f} seconds")
```

## GPU Memory Check

```python
import cupy as cp

# Check GPU memory
mempool = cp.get_default_memory_pool()
print(f"GPU memory used: {mempool.used_bytes() / 1024**2:.2f} MB")
print(f"GPU memory total: {mempool.total_bytes() / 1024**2:.2f} MB")
```
