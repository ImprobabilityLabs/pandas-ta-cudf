# -*- coding: utf-8 -*-
import cudf
from .entropy import entropy
from .kurtosis import kurtosis
from .mad import mad
from .median import median
from .quantile import quantile
from .skew import skew
from .stdev import stdev
from .tos_stdevall import tos_stdevall
from .variance import variance
from .zscore import zscore

# Note: You will also need to modify your functions to work with CuDF DataFrames and Series instead of Pandas DataFrames and Series.