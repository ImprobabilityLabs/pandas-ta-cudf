#!/usr/bin/env python3
"""
Python test script for pandas-ta-cudf conversion
This script verifies that the cudf conversion is working correctly
"""

import sys
import os
from pathlib import Path

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_step(step_num, message):
    """Print a test step header"""
    print(f"\n{Colors.YELLOW}Step {step_num}: {message}{Colors.NC}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.NC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.NC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.NC}")

def test_imports():
    """Test 1: Check Python and cudf"""
    print_step(1, "Checking Python and cuDF installation...")
    try:
        import sys
        print(f"Python version: {sys.version.split()[0]}")
        
        import cudf
        print_success(f"cuDF version: {cudf.__version__}")
        return True
    except ImportError as e:
        print_error(f"cuDF not installed: {e}")
        print("Install cuDF via: conda install -c rapidsai -c conda-forge cudf")
        return False

def test_pandas_ta_import():
    """Test 2: Import pandas_ta"""
    print_step(2, "Testing pandas_ta import...")
    try:
        import pandas_ta as ta
        print_success(f"pandas_ta imported successfully")
        print(f"  Version: {ta.version}")
        return True
    except Exception as e:
        print_error(f"Failed to import pandas_ta: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cudf_dataframe():
    """Test 3: Create cudf DataFrame"""
    print_step(3, "Testing cudf DataFrame creation...")
    try:
        import cudf
        
        df = cudf.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [99, 100, 101, 102, 103],
            'close': [104, 105, 106, 107, 108],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        print_success(f"Created cudf DataFrame")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return True, df
    except Exception as e:
        print_error(f"Failed to create cudf DataFrame: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_accessor(df):
    """Test 4: Test DataFrame accessor"""
    print_step(4, "Testing DataFrame.ta accessor...")
    try:
        ta_accessor = df.ta
        print_success("DataFrame.ta accessor works")
        print(f"  Accessor type: {type(ta_accessor).__name__}")
        return True
    except Exception as e:
        print_error(f"DataFrame.ta accessor failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indicators():
    """Test 5: Test basic indicators"""
    print_step(5, "Testing basic indicators...")
    import cudf
    import pandas_ta as ta
    
    df = cudf.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'close': [104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
    })
    
    indicators = [
        ('SMA', lambda: df.ta.sma(length=5)),
        ('RSI', lambda: df.ta.rsi(length=14)),
        ('MACD', lambda: df.ta.macd()),
        ('BBANDS', lambda: df.ta.bbands(length=5)),
        ('EMA', lambda: df.ta.ema(length=10)),
    ]
    
    passed = 0
    failed = 0
    
    for name, func in indicators:
        try:
            result = func()
            result_name = result.name if hasattr(result, 'name') else 'OK'
            print_success(f"{name}: {result_name}")
            passed += 1
        except Exception as e:
            print_error(f"{name} failed: {e}")
            failed += 1
    
    print(f"\nIndicator Test Results: {passed}/{len(indicators)} passed")
    return failed == 0

def test_append():
    """Test 6: Test append functionality"""
    print_step(6, "Testing append functionality...")
    try:
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
            new_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
            print_success(f"Append works: {initial_cols} -> {final_cols} columns")
            print(f"  New columns: {new_cols}")
            return True
        else:
            print_error("Append failed: columns unchanged")
            return False
    except Exception as e:
        print_error(f"Append test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_pandas_imports():
    """Test 7: Check for remaining pandas imports"""
    print_step(7, "Checking for remaining pandas imports...")
    import re
    
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
        print_warning(f"Found {len(pandas_imports)} files with pandas imports:")
        for f in pandas_imports[:10]:
            print(f"  - {f}")
        if len(pandas_imports) > 10:
            print(f"  ... and {len(pandas_imports) - 10} more")
        print("\nNote: Some files may legitimately use pandas (e.g., data sources)")
        return False
    else:
        print_success("No pandas imports found in pandas_ta directory")
        return True

def check_fillna_issues():
    """Test 8: Check for fillna inplace issues"""
    print_step(8, "Checking for fillna inplace/method issues...")
    import re
    
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
        print_warning(f"Found {len(fillna_issues)} files with fillna inplace/method issues:")
        for f in fillna_issues:
            print(f"  - {f}")
        print("\nThese need to be fixed for cudf compatibility")
        return False
    else:
        print_success("No fillna inplace/method issues found")
        return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Pandas TA cuDF Conversion Test Script")
    print("=" * 50)
    
    results = []
    
    # Test 1: Imports
    if not test_imports():
        print("\n" + Colors.RED + "CRITICAL: cuDF not installed. Cannot continue." + Colors.NC)
        sys.exit(1)
    
    # Test 2: pandas_ta import
    if not test_pandas_ta_import():
        sys.exit(1)
    
    # Test 3: cudf DataFrame
    success, df = test_cudf_dataframe()
    if not success:
        sys.exit(1)
    
    # Test 4: Accessor
    if not test_accessor(df):
        sys.exit(1)
    
    # Test 5: Indicators
    results.append(("Indicators", test_indicators()))
    
    # Test 6: Append
    results.append(("Append", test_append()))
    
    # Test 7: Pandas imports check
    results.append(("Pandas Imports Check", check_pandas_imports()))
    
    # Test 8: Fillna issues check
    results.append(("Fillna Issues Check", check_fillna_issues()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        status = Colors.GREEN + "PASS" + Colors.NC if result else Colors.RED + "FAIL" + Colors.NC
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n" + Colors.GREEN + "All tests passed!" + Colors.NC)
        print("\nNext steps:")
        print("1. Test with your actual data")
        print("2. Verify GPU memory usage")
        print("3. Run performance benchmarks")
        return 0
    else:
        print("\n" + Colors.YELLOW + "Some tests failed. Review errors above." + Colors.NC)
        return 1

if __name__ == '__main__':
    sys.exit(main())

