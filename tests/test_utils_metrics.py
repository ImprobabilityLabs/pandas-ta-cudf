from unittest import skip, TestCase

import cudf
from cudf import DataFrame
from .config import sample_data
from .context import culda_ta


class TestUtilityMetrics(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cudf.DataFrame(sample_data)
        cls.close = cls.data["close"]
        cls.pctret = culda_ta.percent_return(cls.close, cumulative=False)
        cls.logret = culda_ta.percent_return(cls.close, cumulative=False)

    @classmethod
    def tearDownClass(cls):
        del cls.data
        del cls.pctret
        del cls.logret

    def setUp(self): pass
    def tearDown(self): pass

    def test_cagr(self):
        result = culda_ta.utils.cagr(self.data.close)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    def test_calmar_ratio(self):
        result = culda_ta.calmar_ratio(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = culda_ta.calmar_ratio(self.close, years=0)
        self.assertIsNone(result)

        result = culda_ta.calmar_ratio(self.close, years=-2)
        self.assertIsNone(result)

    def test_downside_deviation(self):
        result = culda_ta.downside_deviation(self.pctret)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = culda_ta.downside_deviation(self.logret)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_drawdown(self):
        result = culda_ta.drawdown(self.pctret)
        self.assertIsInstance(result, cudf.DataFrame)
        self.assertEqual(result.name, "DD")

        result = culda_ta.drawdown(self.logret)
        self.assertIsInstance(result, cudf.DataFrame)
        self.assertEqual(result.name, "DD")

    def test_jensens_alpha(self):
        bench_return = self.pctret.sample(n=self.close.shape[0], random_state=1)
        result = culda_ta.jensens_alpha(self.close, bench_return)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_log_max_drawdown(self):
        result = culda_ta.log_max_drawdown(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_max_drawdown(self):
        result = culda_ta.max_drawdown(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = culda_ta.max_drawdown(self.close, method="percent")
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = culda_ta.max_drawdown(self.close, method="log")
        self isinstance(result, float)
        self.assertGreaterEqual(result, 0)

        result = culda_ta.max_drawdown(self.close, all=True)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["dollar"], float)
        self.assertIsInstance(result["percent"], float)
        self.assertIsInstance(result["log"], float)

    def test_optimal_leverage(self):
        result = culda_ta.optimal_leverage(self.close)
        self.assertIsInstance(result, int)
        result = culda_ta.optimal_leverage(self.close, log=True)
        self.assertIsInstance(result, int)

    def test_pure_profit_score(self):
        result = culda_ta.pure_profit_score(self.close)
        self.assertGreaterEqual(result, 0)

    def test_sharpe_ratio(self):
        result = culda_ta.sharpe_ratio(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_sortino_ratio(self):
        result = culda_ta.sortino_ratio(self.close)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_volatility(self):
        returns_ = culda_ta.percent_return(self.close)
        result = culda_ta.utils.volatility(returns_, returns=True)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

        for tf in ["years", "months", "weeks", "days", "hours", "minutes", "seconds"]:
            result = culda_ta.utils.volatility(self.close, tf)
            with self.subTest(tf=tf):
                self.assertIsInstance(result, float)
                self.assertGreaterEqual(result, 0)