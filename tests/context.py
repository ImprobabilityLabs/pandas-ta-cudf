import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cudf
import pandas-Ta  # cudf doesn't have Ta, need to keep pandas-Ta for now