import unittest
from types import SimpleNamespace
import numpy as np
from fromrealhand.multivideo import conditioned_features, trajectory_arrays, MultiVideoActions


class MultiVideoTests(unittest.TestCase):
    def setUp(self):
        self.q=np.zeros(37);self.q[33]=1.
        self.v=np.zeros(36);self.obs=np.zeros(39)

    def test_identity_and_dimension(self):
        a=conditioned_features(self.obs,self.q,self.v,5,.01,2.,5.,0)
        b=conditioned_features(self.obs,self.q,self.v,5,.01,2.,5.,1)
        self.assertEqual(a.shape,(84,))
        np.testing.assert_array_equal(a[:-2],b[:-2])
        np.testing.assert_array_equal(a[-2:],[1,0])
        np.testing.assert_array_equal(b[-2:],[0,1])
        with self.assertRaises(ValueError): conditioned_features(self.obs,self.q,self.v,0,.01,2.,5.,2)

    def test_independent_horizons_and_references(self):
        for vid,length in [(0,3),(1,5)]:
            demo=dict(observations=np.zeros((length,39)),actions=np.ones((length,30))*.2,
                      sim_data=[dict(qpos=self.q.copy(),qvel=self.v.copy()) for _ in range(length)])
            video=dict(id=vid,horizon=length,control=dict(time_scale=5.))
            geometry=dict(source_frames=np.array([0,60]),fps=30.)
            x,y,r=trajectory_arrays(demo,np.ones((length,30))*.1,video,geometry)
            self.assertEqual(x.shape,(length,84));np.testing.assert_allclose(r,.1)
            with self.assertRaises(ValueError): trajectory_arrays(demo,np.zeros((length+1,30)),video,geometry)

    def test_skill_clock_rejected(self):
        exp=SimpleNamespace(env=SimpleNamespace(control_timestep=.01),duration=2.)
        video=dict(name='first',id=0,horizon=3,control=dict(time_scale=5.))
        cp=dict(references=dict(first=np.zeros((3,30))),clocks=dict(first=[.01,3.,5.]))
        with self.assertRaises(ValueError): MultiVideoActions(cp,exp,video)


if __name__=='__main__': unittest.main()
