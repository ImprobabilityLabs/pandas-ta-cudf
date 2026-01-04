#!/usr/bin/env python3
"""
Simple script to fix pandas imports to cudf imports
Run this to fix all remaining pandas imports
"""

import re
from pathlib import Path

# Files to exclude
EXCLUDE = {
    'pandas_ta/utils/data/yahoofinance.py',
    'pandas_ta/utils/data/alphavantage.py',
    'pandas_ta/utils/_time.py',
    'pandas_ta/volume/vp.py',  # Uses pandas cut
}

def fix_file(filepath):
    """Fix pandas imports in a file"""
    file_str = str(filepath)
    
    # Skip excluded files
    if any(ex in file_str for ex in EXCLUDE):
        return False, []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        changes = []
        
        # Fix: from pandas import DataFrame
        if re.search(r'^from pandas import DataFrame$', content, re.MULTILINE):
            content = re.sub(r'^from pandas import DataFrame$', 'from cudf import DataFrame', content, flags=re.MULTILINE)
            changes.append("DataFrame import")
        
        # Fix: from pandas import Series
        if re.search(r'^from pandas import Series$', content, re.MULTILINE):
            content = re.sub(r'^from pandas import Series$', 'from cudf import Series', content, flags=re.MULTILINE)
            changes.append("Series import")
        
        # Fix: from pandas import concat
        if re.search(r'^from pandas import concat$', content, re.MULTILINE):
            content = re.sub(r'^from pandas import concat$', 'from cudf import concat', content, flags=re.MULTILINE)
            changes.append("concat import")
        
        # Fix: from pandas import DataFrame, Series
        if re.search(r'^from pandas import DataFrame, Series$', content, re.MULTILINE):
            content = re.sub(r'^from pandas import DataFrame, Series$', 'from cudf import DataFrame, Series', content, flags=re.MULTILINE)
            changes.append("DataFrame, Series import")
        
        # Fix: from pandas import DataFrame, concat
        if re.search(r'^from pandas import DataFrame, concat$', content, re.MULTILINE):
            content = re.sub(r'^from pandas import DataFrame, concat$', 'from cudf import DataFrame, concat', content, flags=re.MULTILINE)
            changes.append("DataFrame, concat import")
        
        # Fix: from pandas import concat, DataFrame, Series
        if re.search(r'^from pandas import concat, DataFrame, Series$', content, re.MULTILINE):
            content = re.sub(r'^from pandas import concat, DataFrame, Series$', 'from cudf import concat, DataFrame, Series', content, flags=re.MULTILINE)
            changes.append("concat, DataFrame, Series import")
        
        # Fix: from pandas import Series, DataFrame
        if re.search(r'^from pandas import Series, DataFrame$', content, re.MULTILINE):
            content = re.sub(r'^from pandas import Series, DataFrame$', 'from cudf import Series, DataFrame', content, flags=re.MULTILINE)
            changes.append("Series, DataFrame import")
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes
    
    except Exception as e:
        return False, [f"Error: {e}"]
    
    return False, []

def main():
    base_dir = Path('pandas_ta')
    fixed = 0
    total = 0
    
    print("Fixing pandas imports...")
    print("=" * 60)
    
    for py_file in base_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        
        total += 1
        changed, changes = fix_file(py_file)
        
        if changed:
            fixed += 1
            print(f"✓ {py_file}: {', '.join(changes)}")
    
    print("=" * 60)
    print(f"Fixed {fixed} out of {total} files")

if __name__ == '__main__':
    main()
