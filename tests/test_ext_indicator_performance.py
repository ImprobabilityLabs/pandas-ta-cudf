from .config import sample_data
from .context import pandas_ta

from unittest import TestCase
import cudf
from cuml.metrics import log_return, percent_return
from cuml.metrics.conf import SKIP_OUTPUT

class TestPerformaceExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cudf.DataFrame.from_pandas(sample_data)
        cls.islong = cls.data["close"] > cls.data["close"].rolling(window=50).mean()

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.islong

    def setUp(self): pass
    def tearDown(self): pass


    def test_log_return_ext(self):
        self.data['LOGRET_1'] = log_return(self.data["close"], 1)
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "LOGRET_1")

    def test_cum_log_return_ext(self):
        self.data['CUMLOGRET_1'] = self.data["close"].rolling(window=2).apply(lambda x: log_return(x, 1).sum())
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMLOGRET_1")

    def test_percent_return_ext(self):
        self.data['PCTRET_1'] = percent_return(self.data["close"], 1)
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "PCTRET_1")

    def test_cum_percent_return_ext(self):
        self.data['CUMPCTRET_1'] = self.data["close"].rolling(window=2).apply(lambda x: percent_return(x, 1).sum())
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMPCTRET_1")