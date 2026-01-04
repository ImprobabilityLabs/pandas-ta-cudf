#!/usr/bin/env python3
"""
Batch script to fix all cuDF conversion issues in pandas_ta
This script will:
1. Replace pandas imports with cudf imports
2. Fix fillna inplace issues
3. Fix fillna method parameter issues
4. Update docstrings
"""

import os
import re
from pathlib import Path

# Files that should keep pandas (data sources, etc.)
EXCLUDE_FILES = {
    'pandas_ta/utils/data/yahoofinance.py',
    'pandas_ta/utils/data/alphavantage.py',
    'pandas_ta/utils/_time.py',  # Uses pandas Timestamp for compatibility
}

def fix_file(filepath):
    """Fix a single file for cuDF compatibility"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes = []
        
        # 1. Replace pandas imports with cudf
        # Handle "from pandas import DataFrame, Series"
        if re.search(r'from pandas import.*DataFrame.*Series', content):
            content = re.sub(
                r'from pandas import (DataFrame, Series)',
                r'from cudf import DataFrame, Series',
                content
            )
            changes.append("Updated DataFrame, Series import")
        
        # Handle "from pandas import DataFrame"
        if re.search(r'from pandas import.*DataFrame', content) and 'Series' not in content:
            content = re.sub(
                r'from pandas import (DataFrame)',
                r'from cudf import \1',
                content
            )
            changes.append("Updated DataFrame import")
        
        # Handle "from pandas import concat"
        if re.search(r'from pandas import.*concat', content):
            content = re.sub(
                r'from pandas import (concat)',
                r'from cudf import \1',
                content
            )
            changes.append("Updated concat import")
        
        # Handle "import pandas as pd"
        if re.search(r'import pandas as pd', content):
            content = re.sub(
                r'import pandas as pd',
                r'import cudf',
                content
            )
            changes.append("Updated pandas import")
            # Also replace pd.DataFrame and pd.Series
            content = re.sub(r'pd\.DataFrame', 'DataFrame', content)
            content = re.sub(r'pd\.Series', 'Series', content)
            changes.append("Updated pd.DataFrame/pd.Series references")
        
        # 2. Fix fillna inplace issues
        # Pattern: series.fillna(value, inplace=True)
        if re.search(r'\.fillna\([^)]*inplace=True\)', content):
            # Replace fillna with inplace=True
            def replace_fillna_inplace(match):
                line = match.group(0)
                # Extract variable name and value
                var_match = re.search(r'(\w+)\.fillna\(([^,]+),\s*inplace=True\)', line)
                if var_match:
                    var_name = var_match.group(1)
                    value = var_match.group(2)
                    return f'{var_name} = {var_name}.fillna({value})'
                return line
            
            content = re.sub(
                r'(\w+)\.fillna\(([^,]+),\s*inplace=True\)',
                r'\1 = \1.fillna(\2)',
                content
            )
            changes.append("Fixed fillna inplace=True")
        
        # 3. Fix fillna method parameter (cudf doesn't support it)
        # Pattern: series.fillna(method='ffill', inplace=True)
        if re.search(r'\.fillna\([^)]*method=', content):
            # Comment out fill_method lines
            def replace_fillna_method(match):
                line = match.group(0)
                indent = len(line) - len(line.lstrip())
                return ' ' * indent + '# Note: cudf doesn\'t support fill_method parameter\n' + \
                       ' ' * indent + '# ' + line.strip()
            
            content = re.sub(
                r'(\s+)(\w+)\.fillna\(method=([^,)]+)(?:,\s*inplace=True)?\)',
                r'\1# Note: cudf doesn\'t support fill_method parameter\n\1# \2 = \2.fillna(method=\3)',
                content
            )
            changes.append("Commented out fillna method parameter")
        
        # 4. Fix replace inplace issues
        if re.search(r'\.replace\([^)]*inplace=True\)', content):
            content = re.sub(
                r'(\w+)\.replace\(([^,]+),\s*([^,]+),\s*inplace=True\)',
                r'\1 = \1.replace(\2, \3)',
                content
            )
            changes.append("Fixed replace inplace=True")
        
        # 5. Update docstrings (pd.Series -> cudf.Series, pd.DataFrame -> cudf.DataFrame)
        content = re.sub(r'\(pd\.Series\)', '(cudf.Series)', content)
        content = re.sub(r'\(pd\.DataFrame\)', '(cudf.DataFrame)', content)
        if 'pd.Series' in original_content or 'pd.DataFrame' in original_content:
            changes.append("Updated docstrings")
        
        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
        else:
            return False, []
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False, [f"Error: {e}"]

def main():
    """Main function to process all files"""
    base_dir = Path('pandas_ta')
    
    if not base_dir.exists():
        print(f"Error: {base_dir} directory not found")
        return
    
    files_processed = 0
    files_changed = 0
    total_changes = 0
    
    print("=" * 60)
    print("cuDF Conversion Batch Fix Script")
    print("=" * 60)
    print()
    
    # Process all Python files
    for py_file in base_dir.rglob('*.py'):
        file_str = str(py_file)
        
        # Skip excluded files
        if any(exclude in file_str for exclude in EXCLUDE_FILES):
            print(f"⏭  Skipping (excluded): {file_str}")
            continue
        
        # Skip __pycache__
        if '__pycache__' in file_str:
            continue
        
        files_processed += 1
        changed, changes = fix_file(py_file)
        
        if changed:
            files_changed += 1
            total_changes += len(changes)
            print(f"✓ Fixed: {file_str}")
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"○ No changes: {file_str}")
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files processed: {files_processed}")
    print(f"Files changed: {files_changed}")
    print(f"Total changes: {total_changes}")
    print()
    print("Next steps:")
    print("1. Run test script: python test_cudf_conversion.py")
    print("2. Review changes and test with your data")
    print("3. Check for any remaining issues")

if __name__ == '__main__':
    main()
