from .config import sample_data
import cudf
from .context import cucdf-Ta

class TestPerformace(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cudf.DataFrame.from_pandas(sample_data)
        cls.close = cls.data["close"]
        cls.islong = (cls.close > cucdf-Ta.sma(cls.close, length=8)).astype(cls.close.dtype)
        cls.pctret = cucdf-Ta.percent_return(cls.close, cumulative=False)
        cls.logret = cucdf-Ta.percent_return(cls.close, cumulative=False)

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
        result = cucdf-Ta.log_return(self.close)
        self.assertIsInstance(result, cudf.Series)
        self.assertEqual(result.name, "LOGRET_1")

    def test_cum_log_return(self):
        result = cucdf-Ta.log_return(self.close, cumulative=True)
        self.assertIsInstance(result, cudf.Series)
        self.assertEqual(result.name, "CUMLOGRET_1")

    def test_percent_return(self):
        result = cucdf-Ta.percent_return(self.close, cumulative=False)
        self.assertIsInstance(result, cudf.Series)
        self.assertEqual(result.name, "PCTRET_1")

    def test_cum_percent_return(self):
        result = cucdf-Ta.percent_return(self.close, cumulative=True)
        self.assertEqual(result.name, "CUMPCTRET_1")