import json
from pathlib import Path
import tempfile
import unittest
from fromrealhand.whole_table.config import create_runtime, inspect, load_config
from fromrealhand.whole_table.contracts import CompositeCommand, ScaffoldOnlyError
from fromrealhand.whole_table.interfaces import SceneBackend, TransitPlanner, LocalPolicyAdapter

ROOT = Path(__file__).resolve().parents[1]


class WholeTableScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT/'configs/whole-table-v1.json')

    def changed_config(self, section, key, value):
        config = json.loads(json.dumps(self.config))
        config[section][key] = value
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            path.write_text(json.dumps(config))
            return load_config(path)

    def test_inspection_is_not_physics_acceptance(self):
        report = inspect(self.config)
        self.assertTrue(report['configuration_valid'])
        self.assertFalse(report['physics_executed'])
        self.assertFalse(report['runtime_ready'])
        self.assertFalse(report['training_started'])

    def test_no_runtime_or_abstract_backend_can_start(self):
        with self.assertRaises(ScaffoldOnlyError): create_runtime(self.config)
        for backend in (SceneBackend, TransitPlanner, LocalPolicyAdapter):
            with self.assertRaises(TypeError): backend()

    def test_control_channels_stay_separate(self):
        command = CompositeCommand((0.4, -0.3, 0.2), (0.0,)*30)
        self.assertEqual(len(command.local_normalized_action), 30)
        with self.assertRaises(ValueError): CompositeCommand((0, 0, 0), (0,)*33)

    def test_bad_commands_rejected(self):
        for xyz in ((0, 0), (0, 0, float('nan'))):
            with self.assertRaises(ValueError): CompositeCommand(xyz, (0,)*30)
        for action in ((2,)*30, (float('inf'),)*30):
            with self.assertRaises(ValueError): CompositeCommand((0, 0, 0), action)

    def test_no_silent_sampling_exclusions(self):
        with self.assertRaises(ValueError): self.changed_config('table', 'central_clear_corridor', True)
        with self.assertRaises(ValueError): self.changed_config('objects', 'resample_after_planning_failure', True)

    def test_names_and_limits(self):
        with self.assertRaises(ValueError): self.changed_config('platform', 'joint_names', ['x', 'x', 'z'])
        with self.assertRaises(ValueError): self.changed_config('platform', 'provisional_max_velocity_m_s', [0, .1, .1])
        with self.assertRaises(ValueError): self.changed_config('platform', 'provisional_joint_max_m', [-1, -1, -1])

    def test_local_policy_dimensions(self):
        with self.assertRaises(ValueError): self.changed_config('local_policy', 'action_dim', 33)


if __name__ == '__main__': unittest.main()
