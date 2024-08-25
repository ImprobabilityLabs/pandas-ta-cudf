from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import cudf_ta

from unittest import TestCase, skip
import pandas.testing as pdt
from cudf import DataFrame, Series

import talib as tal


class TestVolume(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data.to_pandas()  # Convert sample_data to pandas DataFrame for comparison
        cls.data.columns = cls.data.columns.str.lower()
        cls.open = cls.data["open"]
        cls.high = cls.data["high"]
        cls.low = cls.data["low"]
        cls.close = cls.data["close"]
        if "volume" in cls.data.columns:
            cls.volume_ = cls.data["volume"]

    @classmethod
    def tearDownClass(cls):
        del cls.open
        del cls.high
        del cls.low
        del cls.close
        if hasattr(cls, "volume"):
            del cls.volume_
        del cls.data

    def setUp(self): pass
    def tearDown(self): pass


    def test_ad(self):
        result = cudf_ta.ad(self.high, self.low, self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AD")

        try:
            expected = Series(tal.AD(self.high.to_pandas(), self.low.to_pandas(), self.close.to_pandas(), self.volume_.to_pandas()))  # Convert to pandas.series
            pdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.ad(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "AD")

    def test_ad_open(self):
        result = cudf_ta.ad(self.high, self.low, self.close, self.volume_, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADo")

    def test_adosc(self):
        result = cudf_ta.adosc(self.high, self.low, self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADOSC_3_10")

        try:
            expected = Series(tal.ADOSC(self.high.to_pandas(), self.low.to_pandas(), self.close.to_pandas(), self.volume_.to_pandas()))  # Convert to pandas.series
            pdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.adosc(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ADOSC_3_10")

    def test_aobv(self):
        result = cudf_ta.aobv(self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AOBVe_4_12_2_2_2")

    def test_cmf(self):
        result = cudf_ta.cmf(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CMF_20")

    def test_efi(self):
        result = cudf_ta.efi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EFI_13")

    def test_eom(self):
        result = cudf_ta.eom(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EOM_14_100000000")

    def test_kvo(self):
        result = cudf_ta.kvo(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "KVO_34_55_13")

    def test_mfi(self):
        result = cudf_ta.mfi(self.high, self.low, self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MFI_14")

        try:
            expected = Series(tal.MFI(self.high.to_pandas(), self.low.to_pandas(), self.close.to_pandas(), self.volume_.to_pandas()))  # Convert to pandas.series
            pdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.mfi(self.high, self.low, self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MFI_14")

    def test_nvi(self):
        result = cudf_ta.nvi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "NVI_1")

    def test_obv(self):
        result = cudf_ta.obv(self.close, self.volume_, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OBV")

        try:
            expected = Series(tal.OBV(self.close.to_pandas(), self.volume_.to_pandas()))  # Convert to pandas.series
            pdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.obv(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OBV")

    def test_pvi(self):
        result = cudf_ta.pvi(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVI_1")

    def test_pvol(self):
        result = cudf_ta.pvol(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVOL")

    def test_pvr(self):
        result = cudf_ta.pvr(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVR")
        # sample indicator values from SPY
        self.assertEqual(result.iloc[0], 1)
        self.assertEqual(result.iloc[1], 3)
        self.assertEqual(result.iloc[4], 2)
        self.assertEqual(result.iloc[6], 4)

    def test_pvt(self):
        result = cudf_ta.pvt(self.close, self.volume_)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PVT")

    def test_vp(self):
        result = cudf_ta.vp(self.close, self.volume_)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VP_10")
