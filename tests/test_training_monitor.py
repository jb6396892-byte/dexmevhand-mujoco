import copy
import json
from pathlib import Path
import tempfile
import unittest
from fromrealhand.training_monitor import health, atomic_json


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(max_consecutive_zero_updates=3, catastrophic_penetration_m=.005,
                        warning_task_fraction=.5, warning_window_iterations=5)
        self.row = dict(iteration=1, measured_kl=.001, parameter_delta_l2=.01, contact_cost_sum=1.,
            returns=[1.,0.,1.,1.], baseline_errors=[1.,.5], sampled_steps=10,
            reports=[dict(task_pass=True, strict_pass=True, report=dict(finite=True,
                max_hand_scene_penetration_m=.0009, final_distance_m=.01, max_joint_violation_rad=0.))])

    def test_normal_and_empty(self):
        self.assertEqual(health([], self.cfg, .002)['iterations'], 0)
        out = health([self.row], self.cfg, .002)
        self.assertEqual(out['errors'], [])
        self.assertEqual(out['task_pass'], 1)

    def test_exploration_failure_warns_but_does_not_relax_task_gate(self):
        self.row['reports'][0]['task_pass'] = False
        self.row['reports'][0]['report']['max_hand_scene_penetration_m'] = .0011
        out = health([self.row], self.cfg, .002)
        self.assertEqual(out['errors'], [])
        self.assertIn('exploration_task_failures', out['warnings'])
        self.assertEqual(out['task_pass'], 0)

    def test_nonfinite_kl_and_physics_abort(self):
        self.row['measured_kl'] = .003
        self.row['returns'][0] = float('nan')
        self.row['reports'][0]['report']['max_hand_scene_penetration_m'] = .006
        errors = health([self.row], self.cfg, .002)['errors']
        self.assertIn('kl_exceeded', errors)
        self.assertIn('nonfinite_update', errors)
        self.assertIn('catastrophic_penetration', errors)

    def test_three_zero_updates_abort(self):
        rows = [copy.deepcopy(self.row) for _ in range(3)]
        for i, row in enumerate(rows):
            row.update(iteration=i+1, parameter_delta_l2=0.)
        self.assertIn('consecutive_zero_updates', health(rows, self.cfg, .002)['errors'])

    def test_nonfinite_alarm_can_be_persisted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'status.json'
            atomic_json(path, dict(errors=['nonfinite_update'], value=float('nan')))
            self.assertEqual(json.loads(path.read_text())['value'], None)
            self.assertFalse(path.with_name('status.json.tmp').exists())


if __name__ == '__main__': unittest.main()
