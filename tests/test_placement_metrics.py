import unittest
import numpy as np
from fromrealhand.whole_table.placement_metrics import world_contact_force, stable_placement


class PlacementMetricsTests(unittest.TestCase):
    def row(self, **updates):
        value = dict(cup_position_m=[.1,.2,.03],tilt_deg=0.,bottom_gap_m=0.,linear_speed_m_s=0.,
                     angular_speed_rad_s=0.,table_contacts=3,table_weight_ratio=1.,hand_cup_contacts=0,
                     penetration_m=.0002,illegal_contacts=[])
        value.update(updates)
        return value

    def test_contact_frame_and_body_order(self):
        frame = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        np.testing.assert_allclose(world_contact_force(frame,[3.,0.,0.],True),[0.,0.,3.])
        np.testing.assert_allclose(world_contact_force(frame,[3.,0.,0.],False),[0.,0.,-3.])

    def test_supported_rest_passes(self):
        self.assertTrue(stable_placement([self.row()]*10,[.1,.2])['passed'])

    def test_height_alone_is_not_support(self):
        self.assertFalse(stable_placement([self.row(table_contacts=0,table_weight_ratio=0.)],[.1,.2])['passed'])

    def test_held_or_tilted_cup_rejected(self):
        for change in (dict(hand_cup_contacts=1),dict(tilt_deg=12.),dict(linear_speed_m_s=.02),dict(penetration_m=.002)):
            self.assertFalse(stable_placement([self.row(**change)],[.1,.2])['passed'])
