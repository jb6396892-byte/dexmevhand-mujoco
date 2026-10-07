import unittest
import numpy as np
from fromrealhand.whole_table.navigation import plan,segments_clear,inflated_boxes,NavigationRejected

class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(workspace_min_m=[-1,-1,0],workspace_max_m=[1,1,1],grid_resolution_m=.1,clearance_m=.02)
        self.hand=np.array([[-.05,-.15,-.05],[.05,.05,.05]])

    def test_thin_obstacle_swept_edge(self):
        box=np.array([[[-.001,-1,0],[.001,1,1]]])
        self.assertFalse(segments_clear([-.1,0,.5],[.1,0,.5],box)[0])
        self.assertTrue(segments_clear([-.1,0,.5],[-.1,.5,.5],box)[0])

    def test_hand_not_point(self):
        boxes=inflated_boxes(self.hand,[dict(bounds=[[0,0,0],[.1,.1,.3]])],.02)
        self.assertFalse(segments_clear([-.04,0,.2],[-.04,0,.2],boxes)[0])

    def test_part_union_preserves_empty_gap(self):
        parts=np.array([[[-.2,-.05,-.05],[-.1,.05,.05]],[[.1,-.05,-.05],[.2,.05,.05]]])
        boxes=inflated_boxes(parts,[dict(bounds=[[-.02,-.02,.48],[.02,.02,.52]])],.01)
        self.assertTrue(segments_clear([0,0,.5],[0,0,.5],boxes)[0])

    def test_local_features_translation_invariant(self):
        from fromrealhand.tabletop.training_inputs import features
        pose=np.eye(4);pose[:3,3]=[.02,.03,.1]
        palm=np.array([0.,-.1,.2]);goal=np.array([.1,.05,.16]);zero=np.zeros(30)
        before=features(zero,zero,palm,pose,goal,'second','grasp',3,(0,10),zero,zero)
        delta=np.array([.2,-.2,0]);moved=pose.copy();moved[:3,3]+=delta
        after=features(zero,zero,palm+delta,moved,goal+delta,'second','grasp',3,(0,10),zero,zero)
        np.testing.assert_allclose(before,after,atol=1e-7)

    def test_direct_and_detour(self):
        start=[-.7,0,.2];goal=[.7,0,.2]
        self.assertTrue(plan(start,goal,self.hand,[],self.cfg)['direct'])
        obstacles=[dict(name='wall',bounds=[[-.1,-1,0],[.1,1,.4]])]
        report=plan(start,goal,self.hand,obstacles,self.cfg)
        self.assertFalse(report['direct'])
        points=np.array(report['waypoints'])
        self.assertTrue(np.all(segments_clear(points[:-1],points[1:],inflated_boxes(self.hand,obstacles,.02))))
        self.assertGreater(points[:,2].max(),.4)

    def test_blocked_and_invalid(self):
        obstacles=[dict(name='wall',bounds=[[-.1,-2,-1],[.1,2,2]])]
        with self.assertRaises(NavigationRejected):plan([-.7,0,.2],[.7,0,.2],self.hand,obstacles,self.cfg)
        for goal in ([2,0,.3],[0,float('nan'),.3],[0,0,.2]):
            with self.assertRaises(NavigationRejected):plan([-.7,0,.2],goal,self.hand,obstacles,self.cfg)

if __name__=='__main__':unittest.main()
