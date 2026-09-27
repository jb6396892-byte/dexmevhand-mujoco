import unittest
from test_surface_gate import gate


class SceneGateTest(unittest.TestCase):
    def test_hand_table_collision_rejects_otherwise_valid_grasp(self):
        r = dict(physics_passed=True, max_loaded_gap_m=.0002, max_penetration_m=.0005,
                 max_joint_violation_rad=0., tail_slip_m=0., saturation=.001, finite=True,
                 tail_finger_contact_fraction=dict(thumb=1., index=1.),
                 max_hand_scene_penetration_m=.01, initial_hand_scene_penetration_m=.01)
        self.assertFalse(gate(r))
        r.update(max_hand_scene_penetration_m=.0008, initial_hand_scene_penetration_m=.0001)
        self.assertTrue(gate(r))
        del r['max_hand_scene_penetration_m']
        self.assertFalse(gate(r))
