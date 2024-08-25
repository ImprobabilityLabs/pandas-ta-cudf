from .config import sample_data
from .context import cudf_ta

from unittest import TestCase
from cudf import Series

class TestPerformace(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data.to_cuda()
        cls.close = cls.data["close"]
        cls.islong = (cls.close > cudf_ta.sma(cls.close, length=8)).astype(cls.close.dtype)
        cls.pctret = cudf_ta.percent_return(cls.close, cumulative=False)
        cls.logret = cudf_ta.percent_return(cls.close, cumulative=False)

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.close
        del cls.islong
        del cls.pctret
        del cls.logret

    def setUp(self): pass
    def tearDown(self): pass


    def test_log_return(self):
        result = cudf_ta.log_return(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LOGRET_1")

    def test_cum_log_return(self):
        result = cudf_ta.log_return(self.close, cumulative=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CUMLOGRET_1")

    def test_percent_return(self):
        result = cudf_ta.percent_return(self.close, cumulative=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PCTRET_1")

    def test_cum_percent_return(self):
        result = cudf_ta.percent_return(self.close, cumulative=True)
        self.assertEqual(result.name, "CUMPCTRET_1")