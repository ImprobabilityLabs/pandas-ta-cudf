from cudf.core.series import Series
from .config import sample_data
from .context import pandas_ta

from unittest import TestCase
from cudf import DataFrame

class TestCylesExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data.to_cudf()

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self): pass
    def tearDown(self): pass


    def test_ebsw_ext(self):
        self.data.ta.ebsw(append=True)
        self.assertIsInstance(self.data, DataFrame)
        self.assertEqual(self.data.columns[-1], "EBSW_40_10")
