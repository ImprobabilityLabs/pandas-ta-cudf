from multiprocessing import cpu_count


from time import perf_counter


from .config import sample_data


from .context import pandas_ta


from unittest import skip, skipUnless, TestCase


from pandas import DataFrame


cores = cpu_count()


cumulative = False


speed_table = False


strategy_timed = False


timed = True


verbose = False


class TestStrategyMethods(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = sample_data
        cls.data.ta.cores = cores
        cls.speed_test = DataFrame()

    @classmethod
    def tearDownClass(cls):
        cls.speed_test = cls.speed_test.T
        cls.speed_test.index.name = 'Test'
        cls.speed_test.columns = ['Columns', 'Seconds']
        if cumulative:
            cls.speed_test['Cum. Seconds'] = cls.speed_test['Seconds'].cumsum()
        if speed_table:
            cls.speed_test.to_csv('tests/speed_test.csv')
        if timed:
            tca = cls.speed_test['Columns'].sum()
            tcs = cls.speed_test['Seconds'].sum()
            cps = f'[i] Total Columns / Second for All Tests: {tca / tcs:.5f} '
            print('=' * len(cps))
            print(cls.speed_test)
            print(f'[i] Cores: {cls.data.ta.cores}')
            print(f'[i] Total Datapoints per run: {cls.data.shape[0]}')
            print(f'[i] Total Columns added: {tca}')
            print(f'[i] Total Seconds for All Tests: {tcs:.5f}')
            print(cps)
            print('=' * len(cps))
        del cls.data

    def setUp(self):
        self.added_cols = 0
        self.category = ''
        self.init_cols = len(self.data.columns)
        self.time_diff = 0
        self.result = None
        if verbose:
            print()
        if timed:
            self.stime = perf_counter()

    def tearDown(self):
        if timed:
            self.time_diff = perf_counter() - self.stime
        self.added_cols = len(self.data.columns) - self.init_cols
        self.assertGreaterEqual(self.added_cols, 1)
        self.result = self.data[self.data.columns[-self.added_cols:]]
        self.assertIsInstance(self.result, DataFrame)
        self.data.drop(columns=self.result.columns, axis=1, inplace=True)
        self.speed_test[self.category] = [self.added_cols, self.time_diff]

    def test_all(self):
        self.category = 'All'
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)

    def test_all_ordered(self):
        self.category = 'All'
        self.data.ta.strategy(ordered=True, verbose=verbose, timed=
            strategy_timed)
        self.category = 'All Ordered'

    @skipUnless(verbose, 'verbose mode only')
    def test_all_strategy(self):
        self.data.ta.strategy(pandas_ta.AllStrategy, verbose=verbose, timed
            =strategy_timed)

    @skipUnless(verbose, 'verbose mode only')
    def test_all_name_strategy(self):
        self.category = 'All'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_all_multiparams_strategy(self):
        self.category = 'All'
        self.data.ta.strategy(self.category, length=10, verbose=verbose,
            timed=strategy_timed)
        self.data.ta.strategy(self.category, length=50, verbose=verbose,
            timed=strategy_timed)
        self.data.ta.strategy(self.category, fast=5, slow=10, verbose=
            verbose, timed=strategy_timed)
        self.category = 'All Multiruns with diff Args'

    def test_candles_category(self):
        self.category = 'Candles'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_common(self):
        self.category = 'Common'
        self.data.ta.strategy(pandas_ta.CommonStrategy, verbose=verbose,
            timed=strategy_timed)

    def test_cycles_category(self):
        self.category = 'Cycles'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_custom_a(self):
        self.category = 'Custom A'
        print()
        print(self.category)
        momo_bands_sma_ta = [{'kind': 'cdl_pattern', 'name': 'tristar'}, {
            'kind': 'rsi'}, {'kind': 'macd'}, {'kind': 'sma', 'length': 50},
            {'kind': 'sma', 'length': 200}, {'kind': 'bbands', 'length': 20
            }, {'kind': 'log_return', 'cumulative': True}, {'kind': 'ema',
            'close': 'CUMLOGRET_1', 'length': 5, 'suffix': 'CLR'}]
        custom = pandas_ta.Strategy(
            'Commons with Cumulative Log Return EMA Chain',
            momo_bands_sma_ta,
            'Common indicators with specific lengths and a chained indicator')
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        self.assertEqual(len(self.data.columns), 15)

    def test_custom_args_tuple(self):
        self.category = 'Custom B'
        custom_args_ta = [{'kind': 'ema', 'params': (5,)}, {'kind':
            'fisher', 'params': (13, 7)}]
        custom = pandas_ta.Strategy('Custom Args Tuple', custom_args_ta,
            'Allow for easy filling in indicator arguments by argument placement.'
            )
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

    def test_custom_col_names_tuple(self):
        self.category = 'Custom C'
        custom_args_ta = [{'kind': 'bbands', 'col_names': ('LB', 'MB', 'UB',
            'BW', 'BP')}]
        custom = pandas_ta.Strategy('Custom Col Numbers Tuple',
            custom_args_ta, 'Allow for easy renaming of resultant columns')
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

    def test_custom_col_numbers_tuple(self):
        self.category = 'Custom D'
        custom_args_ta = [{'kind': 'macd', 'col_numbers': (1,)}]
        custom = pandas_ta.Strategy('Custom Col Numbers Tuple',
            custom_args_ta, 'Allow for easy selection of resultant columns')
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)

    def test_custom_a(self):
        self.category = 'Custom E'
        amat_logret_ta = [{'kind': 'amat', 'fast': 20, 'slow': 50}, {'kind':
            'log_return', 'cumulative': True}, {'kind': 'ema', 'close':
            'CUMLOGRET_1', 'length': 5}]
        custom = pandas_ta.Strategy('AMAT Log Returns', amat_logret_ta,
            'AMAT Log Returns')
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed,
            ordered=True)
        self.data.ta.tsignals(trend=self.data['AMATe_LR_20_50_2'], append=True)
        self.assertEqual(len(self.data.columns), 13)

    def test_momentum_category(self):
        self.category = 'Momentum'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_overlap_category(self):
        self.category = 'Overlap'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_performance_category(self):
        self.category = 'Performance'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_statistics_category(self):
        self.category = 'Statistics'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_trend_category(self):
        self.category = 'Trend'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_volatility_category(self):
        self.category = 'Volatility'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_volume_category(self):
        self.category = 'Volume'
        self.data.ta.strategy(self.category, verbose=verbose, timed=
            strategy_timed)

    def test_all_no_multiprocessing(self):
        self.category = 'All with No Multiprocessing'
        cores = self.data.ta.cores
        self.data.ta.cores = 0
        self.data.ta.strategy(verbose=verbose, timed=strategy_timed)
        self.data.ta.cores = cores

    def test_custom_no_multiprocessing(self):
        self.category = 'Custom A with No Multiprocessing'
        cores = self.data.ta.cores
        self.data.ta.cores = 0
        momo_bands_sma_ta = [{'kind': 'rsi'}, {'kind': 'macd'}, {'kind':
            'sma', 'length': 50}, {'kind': 'sma', 'length': 100,
            'col_names': 'sma100'}, {'kind': 'sma', 'length': 200}, {'kind':
            'bbands', 'length': 20}, {'kind': 'log_return', 'cumulative': 
            True}, {'kind': 'ema', 'close': 'CUMLOGRET_1', 'length': 5,
            'suffix': 'CLR'}]
        custom = pandas_ta.Strategy(
            'Commons with Cumulative Log Return EMA Chain',
            momo_bands_sma_ta,
            'Common indicators with specific lengths and a chained indicator')
        self.data.ta.strategy(custom, verbose=verbose, timed=strategy_timed)
        self.data.ta.cores = cores
