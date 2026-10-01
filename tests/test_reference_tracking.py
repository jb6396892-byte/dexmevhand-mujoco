import unittest
import numpy as np
from fromrealhand.reference_tracking import contact_features, reference_features, residual_action, task_gate


class ReferenceTrackingTests(unittest.TestCase):
    def test_contact_mapping_and_missing_sentinel(self):
        contacts=[dict(hand='C_thdistal',normal_force_n=2.,distance_m=-.0005),
                  dict(hand='C_thproximal',normal_force_n=1.,distance_m=.0001),
                  dict(hand='C_ffdistal',normal_force_n=4.,distance_m=0.)]
        f=contact_features(contacts)
        np.testing.assert_allclose(f[:5],[.3,.4,0,0,0])
        np.testing.assert_allclose(f[5:],[-.1,0,1,1,1])

    def test_feature_dimensions_and_no_state_writes(self):
        q=np.arange(37)*.01;before=q.copy()
        f=reference_features(np.zeros(84),q,np.zeros(30),np.zeros(3),np.ones(3),[])
        self.assertEqual(f.shape,(130,))
        np.testing.assert_array_equal(q,before)

    def test_residual_bounded_without_second_actuator_scaling(self):
        reference=np.ones(30)*.2
        np.testing.assert_allclose(residual_action(reference,np.ones(30)*2,np.ones(30)*.01),.21)
        np.testing.assert_array_equal(reference,.2)
        with self.assertRaises(ValueError): residual_action(reference,np.ones(30),np.zeros(30))

    def test_task_gate_does_not_relax_penetration(self):
        from pathlib import Path
        import json
        limits=json.loads((Path(__file__).resolve().parents[1]/'configs/v12-study.json').read_text())['task_gate']
        r=dict(finite=True,hold_s=2.,tail_min_bottom_m=.03,tail_min_fingers=3,tail_min_force_n=1.,
               final_distance_m=.025,max_hand_scene_penetration_m=.0009,max_penetration_m=.0009,
               initial_hand_scene_penetration_m=0.,max_loaded_gap_m=.0001,tail_slip_m=.002,
               max_joint_violation_rad=0.,saturation=0.,
               tail_finger_contact_fraction=dict(thumb=1.,index=1.,middle=1.,ring=0.,little=0.))
        self.assertTrue(task_gate(r,limits))
        r['max_hand_scene_penetration_m']=.00101
        self.assertFalse(task_gate(r,limits))


if __name__=='__main__': unittest.main()
