#!/bin/bash
# Test script for pandas-ta-cudf conversion
# This script verifies that the cudf conversion is working correctly

set -e  # Exit on error

echo "=========================================="
echo "Pandas TA cuDF Conversion Test Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Checking Python version...${NC}"
python --version
echo ""

echo -e "${YELLOW}Step 2: Checking for cudf installation...${NC}"
python -c "import cudf; print(f'cuDF version: {cudf.__version__}')" 2>/dev/null || {
    echo -e "${RED}ERROR: cuDF not installed${NC}"
    echo "Install cuDF via: conda install -c rapidsai -c conda-forge cudf"
    exit 1
}
echo -e "${GREEN}✓ cuDF found${NC}"
echo ""

echo -e "${YELLOW}Step 3: Testing basic imports...${NC}"
python -c "
import sys
try:
    import pandas_ta
    print('✓ pandas_ta imported successfully')
    print(f'  Version: {pandas_ta.version}')
except Exception as e:
    print(f'✗ Failed to import pandas_ta: {e}')
    sys.exit(1)
" || exit 1
echo ""

echo -e "${YELLOW}Step 4: Testing cudf DataFrame creation...${NC}"
python -c "
import cudf
import pandas_ta as ta

# Create a simple cudf DataFrame
df = cudf.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [99, 100, 101, 102, 103],
    'close': [104, 105, 106, 107, 108],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

print('✓ Created cudf DataFrame')
print(f'  Shape: {df.shape}')
print(f'  Columns: {list(df.columns)}')
" || exit 1
echo ""

echo -e "${YELLOW}Step 5: Testing DataFrame accessor...${NC}"
python -c "
import cudf
import pandas_ta as ta

df = cudf.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    'close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})

try:
    # Test accessor
    ta_accessor = df.ta
    print('✓ DataFrame.ta accessor works')
    print(f'  Accessor type: {type(ta_accessor)}')
except Exception as e:
    print(f'✗ DataFrame.ta accessor failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" || exit 1
echo ""

echo -e "${YELLOW}Step 6: Testing basic indicators...${NC}"
python -c "
import cudf
import pandas_ta as ta

df = cudf.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
    'close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
})

indicators_tested = 0
indicators_passed = 0

# Test SMA
try:
    sma_result = df.ta.sma(length=5)
    print(f'✓ SMA: {sma_result.name if hasattr(sma_result, \"name\") else \"OK\"}')
    indicators_passed += 1
except Exception as e:
    print(f'✗ SMA failed: {e}')
indicators_tested += 1

# Test RSI
try:
    rsi_result = df.ta.rsi(length=14)
    print(f'✓ RSI: {rsi_result.name if hasattr(rsi_result, \"name\") else \"OK\"}')
    indicators_passed += 1
except Exception as e:
    print(f'✗ RSI failed: {e}')
indicators_tested += 1

# Test MACD
try:
    macd_result = df.ta.macd()
    print(f'✓ MACD: {macd_result.name if hasattr(macd_result, \"name\") else \"OK\"}')
    indicators_passed += 1
except Exception as e:
    print(f'✗ MACD failed: {e}')
indicators_tested += 1

# Test BBANDS
try:
    bbands_result = df.ta.bbands(length=5)
    print(f'✓ BBANDS: {bbands_result.name if hasattr(bbands_result, \"name\") else \"OK\"}')
    indicators_passed += 1
except Exception as e:
    print(f'✗ BBANDS failed: {e}')
indicators_tested += 1

print(f'\\nIndicator Test Results: {indicators_passed}/{indicators_tested} passed')

if indicators_passed < indicators_tested:
    print('⚠ Some indicators failed - check errors above')
    exit(1)
" || exit 1
echo ""

echo -e "${YELLOW}Step 7: Testing append functionality...${NC}"
python -c "
import cudf
import pandas_ta as ta

df = cudf.DataFrame({
    'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
    'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    'close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
    'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})

initial_cols = len(df.columns)
df.ta.sma(length=5, append=True)
final_cols = len(df.columns)

if final_cols > initial_cols:
    print(f'✓ Append works: {initial_cols} -> {final_cols} columns')
    print(f'  New columns: {[c for c in df.columns if c not in [\"open\", \"high\", \"low\", \"close\", \"volume\"]]}')
else:
    print(f'✗ Append failed: columns unchanged')
    exit(1)
" || exit 1
echo ""

echo -e "${YELLOW}Step 8: Checking for remaining pandas imports...${NC}"
python -c "
import os
import re
from pathlib import Path

pandas_imports = []
base_dir = Path('pandas_ta')

for py_file in base_dir.rglob('*.py'):
    if '__pycache__' in str(py_file):
        continue
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if re.search(r'from pandas import|import pandas|pd\.DataFrame|pd\.Series', content):
                pandas_imports.append(str(py_file))
    except:
        pass

if pandas_imports:
    print(f'⚠ Found {len(pandas_imports)} files with pandas imports:')
    for f in pandas_imports[:10]:  # Show first 10
        print(f'  - {f}')
    if len(pandas_imports) > 10:
        print(f'  ... and {len(pandas_imports) - 10} more')
    print('\\nNote: Some files may legitimately use pandas (e.g., data sources)')
else:
    print('✓ No pandas imports found in pandas_ta directory')
" || exit 1
echo ""

echo -e "${YELLOW}Step 9: Checking for fillna inplace issues...${NC}"
python -c "
import os
import re
from pathlib import Path

fillna_issues = []
base_dir = Path('pandas_ta')

for py_file in base_dir.rglob('*.py'):
    if '__pycache__' in str(py_file):
        continue
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if re.search(r'\.fillna\([^)]*inplace=True\)|\.fillna\([^)]*method=', content):
                fillna_issues.append(str(py_file))
    except:
        pass

if fillna_issues:
    print(f'⚠ Found {len(fillna_issues)} files with fillna inplace/method issues:')
    for f in fillna_issues:
        print(f'  - {f}')
    print('\\nThese need to be fixed for cudf compatibility')
    exit(1)
else:
    print('✓ No fillna inplace/method issues found')
" || exit 1
echo ""

echo "=========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the batch update script if needed: python update_cudf_imports.py"
echo "2. Test with your actual data"
echo "3. Verify GPU memory usage"
echo ""

