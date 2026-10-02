import importlib
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
study = importlib.import_module('111_language_study')


class LanguageStudyTests(unittest.TestCase):
    def setUp(self):
        self.gate = dict(semantic_accuracy_min=.9, schema_fraction_min=.95,
                         unsupported_false_execution_max=0, not_worse_than_base=True)
        self.base = dict(semantic_correct=36)
        self.pilot = dict(total=42, semantic_accuracy=40/42, schema_legal=42,
                          unsupported_false_execution=0, semantic_correct=40)

    def test_good_pilot_can_proceed(self):
        self.assertTrue(study.pilot_passes(self.pilot,self.base,self.gate))

    def test_format_alone_cannot_authorize_formal(self):
        self.assertFalse(study.pilot_passes(dict(self.pilot,semantic_accuracy=.5),self.base,self.gate))

    def test_unsafe_false_execution_blocks_formal(self):
        self.assertFalse(study.pilot_passes(dict(self.pilot,unsupported_false_execution=1),self.base,self.gate))

    def test_regression_and_empty_report_block_formal(self):
        self.assertFalse(study.pilot_passes(self.pilot,dict(semantic_correct=41),self.gate))
        self.assertFalse(study.pilot_passes(dict(self.pilot,total=0),self.base,self.gate))


if __name__ == '__main__':
    unittest.main()
