import unittest
from fromrealhand.whole_table.task import placement_goal_error


class PlacementGoalTests(unittest.TestCase):
    def setUp(self):
        self.config={'table_size_xy_m':[1.,1.]}
        self.layout={'objects':[
            {'name':'mug','xy':[0.,0.],'bounds_xy':[[-.04,-.04],[.04,.04]]},
            {'name':'box','xy':[.2,.2],'bounds_xy':[[.15,.15],[.25,.25]]}]}

    def test_reject_occupied(self):
        self.assertIn('occupied',placement_goal_error(self.layout,[.2,.2,.2],self.config))

    def test_reject_outside(self):
        self.assertIn('outside',placement_goal_error(self.layout,[.48,0,.2],self.config))

    def test_ignore_cup_initial_footprint(self):
        self.assertIsNone(placement_goal_error(self.layout,[0,0,.2],self.config))

    def test_accept_free_target(self):
        self.assertIsNone(placement_goal_error(self.layout,[-.3,.3,.2],self.config))
