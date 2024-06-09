from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import cudf  # Changed to cudf
import cupy  # Added cupy
import pandas  # Added pandas

import talib as tal


class TestCycles(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cudf.DataFrame(sample_data)  # Changed to cudf DataFrame
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume
        del cls.data

    def setUp(self): pass
    def tearDown(self): pass


    def test_ebsw(self):
        result = pandas_ta.ebsw(self.close.to_pandas())  # Changed to to_pandas()
        self.assertIsInstance(result, pandas.Series)  # Changed to pandas.Series
        self.assertEqual(result.name, "EBSW_40_10")