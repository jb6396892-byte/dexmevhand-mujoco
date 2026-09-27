import hashlib
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from fromrealhand.verified_curriculum import load_admitted_demos, restore_contact_model, local_parameter_exports


class AdmissionTest(unittest.TestCase):
    def test_parameter_exports_are_scoped_and_restored(self):
        original = os.path.expanduser
        with tempfile.TemporaryDirectory() as root:
            with local_parameter_exports(root):
                self.assertEqual(os.path.expanduser('~/Desktop/trpo_params'), str(Path(root).resolve()))
                self.assertEqual(os.path.expanduser('~/elsewhere'), original('~/elsewhere'))
            self.assertIs(os.path.expanduser, original)

    def test_contact_parameters_restored_without_changing_dynamics(self):
        model = SimpleNamespace(geom_margin=np.ones(3)*.003, geom_gap=np.zeros(3), body_mass=np.array([.2]))
        restore_contact_model(model, {'physics_model': {'geom_margin': [.0002, .0002, .001], 'geom_gap': [0., 0., 0.]}})
        np.testing.assert_allclose(model.geom_margin, [.0002, .0002, .001])
        np.testing.assert_allclose(model.body_mass, [.2])

    def test_invalid_contact_parameters_are_rejected_atomically(self):
        model = SimpleNamespace(geom_margin=np.ones(3)*.003, geom_gap=np.zeros(3))
        for bad in ({'body_mass': [.1]}, {'geom_gap': [0.]}, {'geom_gap': [0., float('nan'), 0.]},
                    {'geom_margin': [.0002]*3, 'geom_gap': [-1., 0., 0.]}):
            with self.assertRaises(ValueError):
                restore_contact_model(model, {'physics_model': bad})
            np.testing.assert_allclose(model.geom_margin, [.003]*3)

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
