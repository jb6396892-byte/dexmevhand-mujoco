import unittest
from types import SimpleNamespace
import numpy as np
import torch
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

    def test_residual_added_once_and_direct_ignores_reference(self):
        env=SimpleNamespace(control_timestep=.01,_get_observations=lambda:self.obs.copy(),
                            sim=SimpleNamespace(data=SimpleNamespace(qpos=self.q,qvel=self.v)))
        exp=SimpleNamespace(env=env,duration=2.)
        policy=SimpleNamespace(model=lambda x:torch.full((1,30),.2))
        references=dict(first=np.full((3,30),.6),second=np.full((5,30),-.3))
        for name,vid,length,expected in [('first',0,3,.8),('second',1,5,-.1)]:
            video=dict(name=name,id=vid,horizon=length,control=dict(time_scale=5.))
            cp=dict(references=references,clocks={name:[.01,2.,5.]},policy=policy,method='residual_bc')
            actions=MultiVideoActions(cp,exp,video)
            self.assertEqual(len(actions),length)
            np.testing.assert_allclose(actions[0],expected,atol=1e-7)
            cp['method']='direct_bc'
            np.testing.assert_allclose(MultiVideoActions(cp,exp,video)[0],.2)

    def test_protocol_development_and_heldout_are_disjoint(self):
        import json
        from pathlib import Path
        protocol=json.loads((Path(__file__).resolve().parents[1]/'configs/v10-study.json').read_text())
        def key(case): return tuple(case['cup_offset_m'])+(case['cup_yaw_deg'],)+tuple(case['goal_offset_m'])
        development={key(c) for c in protocol['development_second']}
        heldout={key(c) for c in protocol['heldout_per_video']}
        self.assertFalse(development.intersection(heldout))
        self.assertEqual(len(heldout),10)
        self.assertFalse(protocol['evaluation']['heldout_reoptimization'])


if __name__=='__main__': unittest.main()
