import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('contrast', ROOT/'scripts/115_language_contrast_study.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContrastTests(unittest.TestCase):
    def test_unique_balanced_scene_labels(self):
        rows = module.contrast_rows(module.read(module.PHRASES))
        self.assertEqual(len(rows), 296)
        self.assertEqual(len({r['id'] for r in rows}), len(rows))
        self.assertEqual(sum(r['scene']=='first' for r in rows), len(rows)//2)
        self.assertEqual(sum(r['response']['decision']=='reject' for r in rows), 96)
        for row in rows:
            self.assertLessEqual(len(row['instruction']), 120)

    def test_no_exact_evaluation_overlap(self):
        rows = module.contrast_rows(module.read(module.PHRASES))
        phrases = module.read(ROOT/'configs/stage6-language-phrases.json')
        protected = {text for split in ('validation','heldout')
                     for texts in phrases[split].values() for text in texts}
        self.assertFalse(protected.intersection(r['instruction'] for r in rows))


if __name__ == '__main__':
    unittest.main()
