From cucim import cuimage
import cudf
from cucim.ta import cu_pandas_ta

class TestPerformaceExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cudf.DataFrame(sample_data)
        cls.islong = cls.data["close"] > cu_pandas_ta.sma(cls.data["close"], length=50)

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.islong

    def setUp(self): pass
    def tearDown(self): pass

    def test_log_return_ext(self):
        self.data.ta.log_return(append=True)
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "LOGRET_1")

    def test_cum_log_return_ext(self):
        self.data.ta.log_return(append=True, cumulative=True)
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMLOGRET_1")

    def test_percent_return_ext(self):
        self.data.ta.percent_return(append=True)
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "PCTRET_1")

    def test_cum_percent_return_ext(self):
        self.data.ta.percent_return(append=True, cumulative=True)
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "CUMPCTRET_1")