import unittest
import numpy as np
from fromrealhand.desktop.navigation_metrics import joint_limit_metrics


class NavigationMetricsTests(unittest.TestCase):
    def test_navigation_travel_is_not_an_angular_violation(self):
        q=np.zeros(30);q[0]=.8
        ranges=np.tile([-.3,.3],(30,1))
        row=joint_limit_metrics(q,ranges,np.tile([-1.5,1.5],(3,1)))
        self.assertEqual(row,dict(joint_violation_rad=0.,root_translation_violation_m=0.))

    def test_actual_travel_and_rotation_violations_remain_visible(self):
        q=np.zeros(30);q[0]=1.6;q[10]=.5
        row=joint_limit_metrics(q,np.tile([-.3,.3],(30,1)),np.tile([-1.5,1.5],(3,1)))
        self.assertAlmostEqual(row['root_translation_violation_m'],.1)
        self.assertAlmostEqual(row['joint_violation_rad'],.2)

    def test_local_phase_uses_local_translation_range(self):
        q=np.zeros(30);q[2]=-.4
        ranges=np.tile([-.3,.3],(30,1))
        row=joint_limit_metrics(q,ranges,ranges[:3])
        self.assertAlmostEqual(row['root_translation_violation_m'],.1)
        self.assertEqual(row['joint_violation_rad'],0.)
