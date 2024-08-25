# -*- coding: utf-8 -*-
import cudf as cd
from ._candles import *
from ._core import *
from ._math import *
from ._signals import *
from ._time import *
from ._metrics import *
from .data import *

Note: Since the code is just importing modules, there's not much to refactor to work with CuDF and GPU processing. However, I've replaced the import of pandas (assuming it was pandas) with cudf, which is the GPU-accelerated equivalent.