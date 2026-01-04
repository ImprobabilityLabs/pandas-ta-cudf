#!/usr/bin/env python3
"""
Comprehensive test script for pandas-ta-cudf conversion
This script tests:
1. Basic imports and setup
2. DataFrame creation and accessor
3. Multiple indicators across all categories
4. Append functionality
5. Strategy functionality
6. Error handling
7. Performance check
"""

import sys
import time
from pathlib import Path

# Colors for output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header(text):
    """Print a section header"""
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.NC}")
    print(f"{Colors.CYAN}{text}{Colors.NC}")
    print(f"{Colors.CYAN}{'=' * 60}{Colors.NC}")

def print_step(step_num, message):
    """Print a test step header"""
    print(f"\n{Colors.YELLOW}[Step {step_num}] {message}{Colors.NC}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.NC}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.NC}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.NC}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.NC}")

# Test results tracking
test_results = {
    'passed': 0,
    'failed': 0,
    'warnings': 0
}

def test_imports():
    """Test 1: Check Python and cudf installation"""
    print_step(1, "Checking Python and cuDF installation...")
    try:
        import sys
        print_info(f"Python version: {sys.version.split()[0]}")
        
        import cudf
        print_success(f"cuDF version: {cudf.__version__}")
        
        # Check GPU availability
        try:
            import cupy as cp
            print_info(f"GPU available: {cp.cuda.is_available()}")
            if cp.cuda.is_available():
                print_info(f"GPU device: {cp.cuda.Device().id}")
        except ImportError:
            print_warning("cupy not available, cannot check GPU status")
        
        test_results['passed'] += 1
        return True
    except ImportError as e:
        print_error(f"cuDF not installed: {e}")
        print("Install cuDF via: conda install -c rapidsai -c conda-forge cudf")
        test_results['failed'] += 1
        return False

def test_pandas_ta_import():
    """Test 2: Import pandas_ta"""
    print_step(2, "Testing pandas_ta import...")
    try:
        import pandas_ta as ta
        print_success(f"pandas_ta imported successfully")
        print_info(f"Version: {ta.version}")
        print_info(f"cuDF available: {ta.Imports.get('cudf', False)}")
        test_results['passed'] += 1
        return True
    except Exception as e:
        print_error(f"Failed to import pandas_ta: {e}")
        import traceback
        traceback.print_exc()
        test_results['failed'] += 1
        return False

def test_cudf_dataframe():
    """Test 3: Create cudf DataFrame"""
    print_step(3, "Testing cudf DataFrame creation...")
    try:
        import cudf
        
        # Create sample data
        import numpy as np
        n = 100
        df = cudf.DataFrame({
            'open': np.random.uniform(100, 110, n),
            'high': np.random.uniform(110, 120, n),
            'low': np.random.uniform(90, 100, n),
            'close': np.random.uniform(100, 110, n),
            'volume': np.random.uniform(1000, 2000, n)
        })
        
        print_success(f"Created cudf DataFrame")
        print_info(f"Shape: {df.shape}")
        print_info(f"Columns: {list(df.columns)}")
        print_info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        test_results['passed'] += 1
        return True, df
    except Exception as e:
        print_error(f"Failed to create cudf DataFrame: {e}")
        import traceback
        traceback.print_exc()
        test_results['failed'] += 1
        return False, None

def test_accessor(df):
    """Test 4: Test DataFrame accessor"""
    print_step(4, "Testing DataFrame.ta accessor...")
    try:
        ta_accessor = df.ta
        print_success("DataFrame.ta accessor works")
        print_info(f"Accessor type: {type(ta_accessor).__name__}")
        
        # Test some properties
        print_info(f"Categories: {ta_accessor.categories}")
        print_info(f"Version: {ta_accessor.version}")
        test_results['passed'] += 1
        return True
    except Exception as e:
        print_error(f"DataFrame.ta accessor failed: {e}")
        import traceback
        traceback.print_exc()
        test_results['failed'] += 1
        return False

def test_indicators_by_category():
    """Test 5: Test indicators from each category"""
    print_step(5, "Testing indicators from each category...")
    import cudf
    import numpy as np
    
    # Create larger dataset
    n = 200
    df = cudf.DataFrame({
        'open': np.random.uniform(100, 110, n),
        'high': np.random.uniform(110, 120, n),
        'low': np.random.uniform(90, 100, n),
        'close': np.random.uniform(100, 110, n),
        'volume': np.random.uniform(1000, 2000, n)
    })
    
    # Test indicators from each category
    indicators = {
        'Overlap': [
            ('SMA', lambda: df.ta.sma(length=20)),
            ('EMA', lambda: df.ta.ema(length=20)),
            ('WMA', lambda: df.ta.wma(length=20)),
            ('BBANDS', lambda: df.ta.bbands(length=20)),
        ],
        'Momentum': [
            ('RSI', lambda: df.ta.rsi(length=14)),
            ('MACD', lambda: df.ta.macd()),
            ('STOCH', lambda: df.ta.stoch()),
            ('CCI', lambda: df.ta.cci(length=20)),
        ],
        'Trend': [
            ('ADX', lambda: df.ta.adx(length=14)),
            ('AROON', lambda: df.ta.aroon(length=25)),
            ('PSAR', lambda: df.ta.psar()),
        ],
        'Volatility': [
            ('ATR', lambda: df.ta.atr(length=14)),
            ('NATR', lambda: df.ta.natr(length=14)),
            ('KC', lambda: df.ta.kc(length=20)),
        ],
        'Volume': [
            ('OBV', lambda: df.ta.obv()),
            ('CMF', lambda: df.ta.cmf(length=20)),
            ('MFI', lambda: df.ta.mfi(length=14)),
        ],
        'Statistics': [
            ('ZSCORE', lambda: df.ta.zscore(length=30)),
            ('STDEV', lambda: df.ta.stdev(length=30)),
        ],
    }
    
    passed = 0
    failed = 0
    
    for category, inds in indicators.items():
        print(f"\n  Testing {category} indicators:")
        for name, func in inds:
            try:
                start_time = time.time()
                result = func()
                elapsed = (time.time() - start_time) * 1000
                
                if result is not None:
                    result_type = type(result).__name__
                    if hasattr(result, 'shape'):
                        shape_info = f"shape={result.shape}"
                    elif hasattr(result, '__len__'):
                        shape_info = f"length={len(result)}"
                    else:
                        shape_info = "OK"
                    
                    print_success(f"  {name}: {result_type} ({shape_info}) - {elapsed:.2f}ms")
                    passed += 1
                else:
                    print_warning(f"  {name}: returned None")
                    failed += 1
            except Exception as e:
                print_error(f"  {name} failed: {str(e)[:50]}")
                failed += 1
    
    print(f"\n  Indicator Test Results: {passed} passed, {failed} failed")
    if failed == 0:
        test_results['passed'] += 1
    else:
        test_results['failed'] += 1
        test_results['warnings'] += failed
    
    return failed == 0

def test_append():
    """Test 6: Test append functionality"""
    print_step(6, "Testing append functionality...")
    try:
        import cudf
        import numpy as np
        
        df = cudf.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.uniform(1000, 2000, 50)
        })
        
        initial_cols = len(df.columns)
        
        # Append multiple indicators
        df.ta.sma(length=10, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        
        final_cols = len(df.columns)
        new_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
        
        if final_cols > initial_cols:
            print_success(f"Append works: {initial_cols} -> {final_cols} columns")
            print_info(f"New columns: {', '.join(new_cols[:5])}{'...' if len(new_cols) > 5 else ''}")
            test_results['passed'] += 1
            return True
        else:
            print_error("Append failed: columns unchanged")
            test_results['failed'] += 1
            return False
    except Exception as e:
        print_error(f"Append test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results['failed'] += 1
        return False

def test_strategy():
    """Test 7: Test strategy functionality"""
    print_step(7, "Testing strategy functionality...")
    try:
        import cudf
        import numpy as np
        import pandas_ta as ta
        
        df = cudf.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 2000, 100)
        })
        
        initial_cols = len(df.columns)
        
        # Test custom strategy
        custom_strategy = ta.Strategy(
            name="Test Strategy",
            ta=[
                {"kind": "sma", "length": 10},
                {"kind": "rsi", "length": 14},
            ]
        )
        
        df.ta.strategy(custom_strategy)
        
        final_cols = len(df.columns)
        
        if final_cols > initial_cols:
            print_success(f"Strategy works: {initial_cols} -> {final_cols} columns")
            test_results['passed'] += 1
            return True
        else:
            print_warning("Strategy may not have added columns")
            test_results['warnings'] += 1
            return False
    except Exception as e:
        print_error(f"Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        test_results['failed'] += 1
        return False

def check_remaining_issues():
    """Test 8: Check for remaining pandas imports and fillna issues"""
    print_step(8, "Checking for remaining conversion issues...")
    import re
    
    pandas_imports = []
    fillna_issues = []
    base_dir = Path('pandas_ta')
    
    exclude_patterns = [
        '__pycache__',
        'utils/data/yahoofinance.py',
        'utils/data/alphavantage.py',
        'utils/_time.py',  # Uses pandas Timestamp for compatibility
    ]
    
    for py_file in base_dir.rglob('*.py'):
        file_str = str(py_file)
        
        # Skip excluded files
        if any(pattern in file_str for pattern in exclude_patterns):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for pandas imports (excluding docstrings and comments)
                if re.search(r'^from pandas import|^import pandas', content, re.MULTILINE):
                    # Check if it's in a comment or docstring
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped.startswith('from pandas import') or stripped.startswith('import pandas'):
                            # Check if it's a comment
                            if not stripped.startswith('#'):
                                pandas_imports.append((file_str, i+1))
                                break
                
                # Check for fillna issues
                if re.search(r'\.fillna\([^)]*inplace=True\)|\.fillna\([^)]*method=', content):
                    fillna_issues.append(file_str)
        except:
            pass
    
    if pandas_imports:
        print_warning(f"Found {len(pandas_imports)} files with pandas imports:")
        for f, line in pandas_imports[:10]:
            print(f"  - {f}:{line}")
        if len(pandas_imports) > 10:
            print(f"  ... and {len(pandas_imports) - 10} more")
        test_results['warnings'] += len(pandas_imports)
    else:
        print_success("No pandas imports found (excluding data sources)")
    
    if fillna_issues:
        print_warning(f"Found {len(fillna_issues)} files with fillna inplace/method issues:")
        for f in fillna_issues[:10]:
            print(f"  - {f}")
        if len(fillna_issues) > 10:
            print(f"  ... and {len(fillna_issues) - 10} more")
        test_results['warnings'] += len(fillna_issues)
    else:
        print_success("No fillna inplace/method issues found")
    
    return len(pandas_imports) == 0 and len(fillna_issues) == 0

def test_with_sample_data():
    """Test 9: Test with sample CSV data if available"""
    print_step(9, "Testing with sample data...")
    
    sample_files = [
        Path('data/sample.csv'),
        Path('data/SPY_D.csv'),
    ]
    
    for sample_file in sample_files:
        if sample_file.exists():
            try:
                import cudf
                df = cudf.read_csv(sample_file)
                
                # Try to rename columns if needed
                if 'Close' in df.columns:
                    df.columns = df.columns.str.lower()
                
                print_info(f"Loaded {sample_file}: {df.shape}")
                
                # Test a few indicators
                try:
                    result = df.ta.sma(length=10)
                    print_success(f"✓ SMA works with {sample_file.name}")
                    test_results['passed'] += 1
                    return True
                except Exception as e:
                    print_warning(f"Indicator test failed with {sample_file.name}: {e}")
                    test_results['warnings'] += 1
                    return False
            except Exception as e:
                print_warning(f"Could not load {sample_file}: {e}")
                test_results['warnings'] += 1
    
    print_info("No sample data files found, skipping")
    return True

def main():
    """Run all tests"""
    print_header("Pandas TA cuDF Conversion - Comprehensive Test")
    
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
    test_indicators_by_category()
    
    # Test 6: Append
    test_append()
    
    # Test 7: Strategy
    test_strategy()
    
    # Test 8: Check remaining issues
    check_remaining_issues()
    
    # Test 9: Sample data
    test_with_sample_data()
    
    # Summary
    print_header("Test Summary")
    print(f"{Colors.GREEN}Passed: {test_results['passed']}{Colors.NC}")
    print(f"{Colors.RED}Failed: {test_results['failed']}{Colors.NC}")
    print(f"{Colors.YELLOW}Warnings: {test_results['warnings']}{Colors.NC}")
    
    total = test_results['passed'] + test_results['failed']
    if total > 0:
        success_rate = (test_results['passed'] / total) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if test_results['failed'] == 0:
        print(f"\n{Colors.GREEN}✓ All critical tests passed!{Colors.NC}")
        if test_results['warnings'] > 0:
            print(f"{Colors.YELLOW}⚠ Some warnings found - review above{Colors.NC}")
        return 0
    else:
        print(f"\n{Colors.RED}✗ Some tests failed. Review errors above.{Colors.NC}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
