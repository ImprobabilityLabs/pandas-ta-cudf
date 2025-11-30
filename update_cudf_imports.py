#!/usr/bin/env python3
"""
Script to batch update pandas imports to cudf in indicator files.
Run this script to update all remaining files.
"""
import os
import re
from pathlib import Path

def update_file(filepath):
    """Update a single file to use cudf instead of pandas"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace pandas imports with cudf
        content = re.sub(
            r'from pandas import (DataFrame|Series|concat)',
            r'from cudf import \1',
            content
        )
        content = re.sub(
            r'from pandas import (DataFrame, Series)',
            r'from cudf import DataFrame, Series',
            content
        )
        content = re.sub(
            r'from pandas import (DataFrame, concat)',
            r'from cudf import DataFrame, concat',
            content
        )
        content = re.sub(
            r'import pandas as pd',
            r'import cudf',
            content
        )
        content = re.sub(
            r'import pandas',
            r'import cudf',
            content
        )
        
        # Replace fillna inplace calls
        content = re.sub(
            r'(\w+)\.fillna\(([^,]+),\s*inplace=True\)',
            r'\1 = \1.fillna(\2)',
            content
        )
        content = re.sub(
            r'(\w+)\.fillna\(method=([^,]+),\s*inplace=True\)',
            r'# Note: cudf doesn\'t support fill_method parameter\n        # \1 = \1.fillna(method=\2)',
            content
        )
        
        # Replace pd.DataFrame and pd.Series references
        content = re.sub(r'pd\.DataFrame', 'DataFrame', content)
        content = re.sub(r'pd\.Series', 'Series', content)
        
        # Update docstrings
        content = re.sub(r'\(pd\.Series\)', '(cudf.Series)', content)
        content = re.sub(r'\(pd\.DataFrame\)', '(cudf.DataFrame)', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False

def main():
    """Update all Python files in pandas_ta directory"""
    base_dir = Path('pandas_ta')
    updated_count = 0
    
    for py_file in base_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
        if update_file(py_file):
            print(f"Updated: {py_file}")
            updated_count += 1
    
    print(f"\nUpdated {updated_count} files")

if __name__ == '__main__':
    main()

