from .config import error_analysis, sample_data, CORRELATION, CORRELATION_THRESHOLD, VERBOSE
from .context import cudf_ta

from unittest import TestCase, skip
import cudf.testing as cdt
from cudf import DataFrame, Series

import talib as tal


class TestTrend(TestCase):
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


    def test_adx(self):
        result = cudf_ta.adx(self.high, self.low, self.close, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ADX_14")

        try:
            expected = tal.ADX(self.high.to_pandas(), self.low.to_pandas(), self.close.to_pandas())
            cdt.assert_series_equal(result.iloc[:, 0].to_pandas(), expected)
        except AssertionError:
            try:
                corr = cudf_ta.utils.df_error_analysis(result.iloc[:, 0], expected, col=CORRELATION)
                self.assertGreater(corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result, CORRELATION, ex)

        result = cudf_ta.adx(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "ADX_14")

    def test_amat(self):
        result = cudf_ta.amat(self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AMATe_8_21_2")

    def test_aroon(self):
        result = cudf_ta.aroon(self.high, self.low, talib=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")

        try:
            expected = tal.AROON(self.high.to_pandas(), self.low.to_pandas())
            expecteddf = DataFrame({"AROOND_14": expected[0], "AROONU_14": expected[1]})
            cdt.assert_frame_equal(result.to_pandas(), expecteddf)
        except AssertionError:
            try:
                aroond_corr = cudf_ta.utils.df_error_analysis(result.iloc[:, 0], expecteddf.iloc[:, 0], col=CORRELATION)
                self.assertGreater(aroond_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

            try:
                aroonu_corr = cudf_ta.utils.df_error_analysis(result.iloc[:, 1], expecteddf.iloc[:, 1], col=CORRELATION)
                self.assertGreater(aroonu_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 1], CORRELATION, ex, newline=False)

        result = cudf_ta.aroon(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")

    def test_aroon_osc(self):
        result = cudf_ta.aroon(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "AROON_14")

        try:
            expected = tal.AROONOSC(self.high.to_pandas(), self.low.to_pandas())
            cdt.assert_series_equal(result.iloc[:, 2].to_pandas(), expected)
        except AssertionError:
            try:
                aroond_corr = cudf_ta.utils.df_error_analysis(result.iloc[:,2], expected,col=CORRELATION)
                self.assertGreater(aroond_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(result.iloc[:, 0], CORRELATION, ex)

    def test_chop(self):
        result = cudf_ta.chop(self.high, self.low, self.close, ln=False)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOP_14_1_100")

        result = cudf_ta.chop(self.high, self.low, self.close, ln=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "CHOPln_14_1_100")

    def test_cksp(self):
        result = cudf_ta.cksp(self.high, self.low, self.close, tvmode=False)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CKSP_10_3_20")

    def test_cksp_tv(self):
        result = cudf_ta.cksp(self.high, self.low, self.close, tvmode=True)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "CKSP_10_1_9")

    def test_decay(self):
        result = cudf_ta.decay(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LDECAY_5")

        result = cudf_ta.decay(self.close, mode="exp")
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "EXPDECAY_5")

    def test_decreasing(self):
        result = cudf_ta.decreasing(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DEC_1")

        result = cudf_ta.decreasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SDEC_3")

    def test_dpo(self):
        result = cudf_ta.dpo(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "DPO_20")

    def test_increasing(self):
        result = cudf_ta.increasing(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "INC_1")

        result = cudf_ta.increasing(self.close, length=3, strict=True)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SINC_3")

    def test_long_run(self):
        result = cudf_ta.long_run(self.close, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "LR_2")

    def test_psar(self):
        result = cudf_ta.psar(self.high, self.low)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "PSAR_0.02_0.2")

        # Combine Long and Short SAR"s into one SAR value
        psar = result[result.columns[:2]].fillna(0)
        psar = psar[psar.columns[0]] + psar[psar.columns[1]]
        psar.name = result.name

        try:
            expected = tal.SAR(self.high.to_pandas(), self.low.to_pandas())
            cdt.assert_series_equal(psar.to_pandas(), expected)
        except AssertionError:
            try:
                psar_corr = cudf_ta.utils.df_error_analysis(psar, expected, col=CORRELATION)
                self.assertGreater(psar_corr, CORRELATION_THRESHOLD)
            except Exception as ex:
                error_analysis(psar, CORRELATION, ex)

    def test_qstick(self):
        result = cudf_ta.qstick(self.open, self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "QS_10")

    def test_short_run(self):
        result = cudf_ta.short_run(self.close, self.open)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "SR_2")

    def test_ttm_trend(self):
        result = cudf_ta.ttm_trend(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "TTMTREND_6")

    def test_vhf(self):
        result = cudf_ta.vhf(self.close)
        self.assertIsInstance(result, Series)
        self.assertEqual(result.name, "VHF_28")

    def test_vortex(self):
        result = cudf_ta.vortex(self.high, self.low, self.close)
        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.name, "VTX_14")
