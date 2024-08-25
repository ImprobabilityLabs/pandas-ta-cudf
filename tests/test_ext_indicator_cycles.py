from cudf.core.series import Series
from .config import sample_data
from .context import cudf_ta
import cudf

from unittest import TestCase

class TestCylesExtension(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = cudf.DataFrame.from_pandas(sample_data)

    @classmethod
    def tearDownClass(cls):
        del cls.data

    def setUp(self): pass
    def tearDown(self): pass


    def test_ebsw_ext(self):
        self.data.ta.ebsw(append=True)
        self.assertIsInstance(self.data, cudf.DataFrame)
        self.assertEqual(self.data.columns[-1], "EBSW_40_10")