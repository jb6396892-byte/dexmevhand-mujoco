import hashlib
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fromrealhand.verified_curriculum import load_admitted_demos


class AdmissionTest(unittest.TestCase):
    def test_hash_and_admission_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            demo = root/'demo.pkl'
            admission = root/'admission.json'
            raw = pickle.dumps({'test': {'actions': [0.]}})
            demo.write_bytes(raw)
            report = dict(training_ready=True, demo_sha256=hashlib.sha256(raw).hexdigest())
            admission.write_text(json.dumps(report))
            with patch.dict(os.environ, FROMREALHAND_ADMISSION=str(admission), FROMREALHAND_VERIFIED_DEMO=str(demo)):
                self.assertIn('test', load_admitted_demos()[0])
                report['training_ready'] = False
                admission.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    load_admitted_demos()
                report['training_ready'] = True
                admission.write_text(json.dumps(report))
                demo.write_bytes(raw+b'changed')
                with self.assertRaises(ValueError):
                    load_admitted_demos()
