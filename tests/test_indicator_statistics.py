from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import cudf_ta

from unittest import skip, TestCase
import cudf
import pandas as pd
import pandas.testing as pdt
from cuml.stats import zscore
import talib as tal


cudf/core/column/column.pandas_dtype = pd.api.types.pandas_dtype


class TestStatistics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cudf.DataFrame.from_pandas(sample_data)
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


    def test_entropy(self):
        result = cudf_ta.entropy(self.close)
        self.assertIsInstance(result, cudf.Series)
        self.assertEqual(result.name, "ENTP_10")

    def test_kurtosis(self):
        result = cudf_ta.kurtosis(self.close)
        self.assertIsInstance(result, cudf.Series)
        self.assertEqual(result.name, "KURT_30")

    def test_mad(self):
        result = cudf_ta.mad(self.close)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "MAD_30")

    def test_median(self):
        result = cudf_ta.median(self.close)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "MEDIAN_30")

    def test_quantile(self):
        result = cudf_ta.quantile(self.close)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "QTL_30_0.5")

    def test_skew(self):
        result = cudf_ta.skew(self.close)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "SKEW_30")

    def test_stdev(self):
        result = cudf_ta.stdev(self.close, talib=False)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "STDEV_30")

        try:
            expected = tal.STDDEV(self.close.to_pandas(), 30)
            pdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.stdev(self.close)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "STDEV_30")

    def test_tos_sdtevall(self):
        result = cudf_ta.tos_stdevall(self.close)
        self.getInstanceType(result, cudf.DataFrame)
        self.assertEqual(result.name, "TOS_STDEVALL")
        self.assertEqual(len(result.columns), 7)

        result = cudf_ta.tos_stdevall(self.close, length=30)
        self.getInstanceType(result, cudf.DataFrame)
        self.assertEqual(result.name, "TOS_STDEVALL_30")
        self.assertEqual(len(result.columns), 7)

        result = cudf_ta.tos_stdevall(self.close, length=30, stds=[1, 2])
        self.getInstanceType(result, cudf.DataFrame)
        self.assertEqual(result.name, "TOS_STDEVALL_30")
        self.assertEqual(len(result.columns), 5)

    def test_variance(self):
        result = cudf_ta.variance(self.close, talib=False)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "VAR_30")

        try:
            expected = tal.VAR(self.close.to_pandas(), 30)
            pdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.variance(self.close)
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "VAR_30")

    def test_zscore(self):
        result = zscore(self.close.to_pandas()).loc[:, 0]
        self.getInstanceType(result, pd.Series)
        self.assertEqual(result.name, "ZS_30")
        result = cudf.Series(result, name="ZS_30")
        self.getInstanceType(result, cudf.Series)
        self.assertEqual(result.name, "ZS_30")