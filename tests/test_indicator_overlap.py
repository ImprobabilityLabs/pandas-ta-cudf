from .config import CORRELATION, CORRELATION_THRESHOLD, error_analysis, sample_data, VERBOSE
from .context import cudf_ta

from unittest import TestCase
import cudf.testing as cdt
from cudf import DataFrame, Series

import talib as tal


class TestOverlap(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data
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


    def test_alma(self):
        result = cudf_ta.alma(self.close)# , length=None, sigma=None, distribution_offset=)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ALMA_10_6.0_0.85")

    def test_dema(self):
        result = cudf_ta.dema(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DEMA_10")

        try:
            expected = tal.DEMA(self.close.to_pandas(), 10)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.dema(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DEMA_10")

    def test_ema(self):
        result = cudf_ta.ema(self.close, presma=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

        try:
            expected = tal.EMA(self.close.to_pandas(), 10)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.ema(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

        try:
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.ema(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

    def test_fwma(self):
        result = cudf_ta.fwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "FWMA_10")

    def test_hilo(self):
        result = cudf_ta.hilo(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "HILO_13_21")

    def test_hl2(self):
        result = cudf_ta.hl2(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HL2")

    def test_hlc3(self):
        result = cudf_ta.hlc3(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HLC3")

        try:
            expected = tal.TYPPRICE(self.high.to_pandas(), self.low.to_pandas(), self.close.to_pandas())
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.hlc3(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HLC3")

    def test_hma(self):
        result = cudf_ta.hma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HMA_10")

    def test_hwma(self):
        result = cudf_ta.hwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "HWMA_0.2_0.1_0.1")

    def test_kama(self):
        result = cudf_ta.kama(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "KAMA_10_2_30")

    def test_jma(self):
        result = cudf_ta.jma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "JMA_7_0")

    def test_ichimoku(self):
        ichimoku, span = cudf_ta.ichimoku(self.high, self.low, self.close)
        self.assertIsInstance(ichimoku, DataFrame)
        self.assertIsInstance(span, DataFrame)
        self.assertEqual(ichimoku.name, "ICHIMOKU_9_26_52")
        self.assertEqual(span.name, "ICHISPAN_9_26")

    def test_linreg(self):
        result = cudf_ta.linreg(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LR_14")

        try:
            expected = tal.LINEARREG(self.close.to_pandas())
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.linreg(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LR_14")

    def test_linreg_angle(self):
        result = cudf_ta.linreg(self.close, angle=True, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRa_14")

        try:
            expected = tal.LINEARREG_ANGLE(self.close.to_pandas())
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.linreg(self.close, angle=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRa_14")

    def test_linreg_intercept(self):
        result = cudf_ta.linreg(self.close, intercept=True, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRb_14")

        try:
            expected = tal.LINEARREG_INTERCEPT(self.close.to_pandas())
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.linreg(self.close, intercept=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRb_14")

    def test_linreg_r(self):
        result = cudf_ta.linreg(self.close, r=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRr_14")

    def test_linreg_slope(self):
        result = cudf_ta.linreg(self.close, slope=True, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRm_14")

        try:
            expected = tal.LINEARREG_SLOPE(self.close.to_pandas())
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.linreg(self.close, slope=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LRm_14")

    def test_ma(self):
        result = cudf_ta.ma()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

        result = cudf_ta.ma("ema", self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EMA_10")

        result = cudf_ta.ma("fwma", self.close, length=15)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "FWMA_15")

    def test_mcgd(self):
        result = cudf_ta.mcgd(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MCGD_10")

    def test_midpoint(self):
        result = cudf_ta.midpoint(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPOINT_2")

        try:
            expected = tal.MIDPOINT(self.close.to_pandas(), 2)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.midpoint(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPOINT_2")

    def test_midprice(self):
        result = cudf_ta.midprice(self.high, self.low, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPRICE_2")

        try:
            expected = tal.MIDPRICE(self.high.to_pandas(), self.low.to_pandas(), 2)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.midprice(self.high, self.low)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "MIDPRICE_2")

    def test_ohlc4(self):
        result = cudf_ta.ohlc4(self.open, self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "OHLC4")

    def test_pwma(self):
        result = cudf_ta.pwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "PWMA_10")

    def test_rma(self):
        result = cudf_ta.rma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "RMA_10")

    def test_sinwma(self):
        result = cudf_ta.sinwma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SINWMA_14")

    def test_sma(self):
        result = cudf_ta.sma(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SMA_10")

        try:
            expected = tal.SMA(self.close.to_pandas(), 10)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.sma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SMA_10")

    def test_ssf(self):
        result = cudf_ta.ssf(self.close, poles=2)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_2")

        result = cudf_ta.ssf(self.close, poles=3)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SSF_10_3")

    def test_swma(self):
        result = cudf_ta.swma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SWMA_10")

    def test_supertrend(self):
        result = cudf_ta.supertrend(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "SUPERT_7_3.0")

    def test_t3(self):
        result = cudf_ta.t3(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "T3_10_0.7")

        try:
            expected = tal.T3(self.close.to_pandas(), 10)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.t3(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "T3_10_0.7")

    def test_tema(self):
        result = cudf_ta.tema(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TEMA_10")

        try:
            expected = tal.TEMA(self.close.to_pandas(), 10)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.tema(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TEMA_10")

    def test_trima(self):
        result = cudf_ta.trima(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TRIMA_10")

        try:
            expected = tal.TRIMA(self.close.to_pandas(), 10)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.trima(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "TRIMA_10")

    def test_vidya(self):
        result = cudf_ta.vidya(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VIDYA_14")

    def test_vwap(self):
        result = cudf_ta.vwap(self.high, self.low, self.close, self.volume)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VWAP_D")

    def test_vwma(self):
        result = cudf_ta.vwma(self.close, self.volume)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VWMA_10")

    def test_wcp(self):
        result = cudf_ta.wcp(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WCP")

        try:
            expected = tal.WCLPRICE(self.high.to_pandas(), self.low.to_pandas(), self.close.to_pandas())
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.wcp(self.high, self.low, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WCP")

    def test_wma(self):
        result = cudf_ta.wma(self.close, talib=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WMA_10")

        try:
            expected = tal.WMA(self.close.to_pandas(), 10)
            cdt.assert_series_equal(result.to_pandas(), expected, check_names=False)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.to_pandas(), expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.to_pandas(), CORRELATION, ex)

        result = cudf_ta.wma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "WMA_10")

    def test_zlma(self):
        result = cudf_ta.zlma(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "ZL_EMA_10")
