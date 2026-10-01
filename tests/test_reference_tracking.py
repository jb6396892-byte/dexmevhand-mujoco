import unittest
import numpy as np
from types import SimpleNamespace
from fromrealhand.reference_tracking import contact_features, reference_features, residual_action, task_gate, ReferenceContext
from fromrealhand.residual_sampling import ResidualSampler


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

    def fixture(self):
        q=np.zeros(37);q[33]=1.;q[30]=.004;q.setflags(write=False)
        v=np.zeros(36);v.setflags(write=False)
        model=SimpleNamespace(actuator_biasprm=np.tile([0.,-1.,0.],(30,1)),
                              actuator_gainprm=np.ones((30,3)),body_pos=np.array([[0.,0.,0.],[.104,0.,0.]]),
                              body_quat=np.array([[1.,0.,0.,0.],[1.,0.,0.,0.]]),body_name2id=lambda name:0)
        env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=v)),
                            act_rng=np.ones(30),control_timestep=.01,target_object_bid=1,
                            _get_observations=lambda:np.zeros(39))
        exp=SimpleNamespace(model=model,env=env,duration=2.4)
        video=dict(id=0,name='first',horizon=2,control=dict(time_scale=5.))
        poses=np.tile(np.eye(4),(2,1,1));poses[-1,0,3]=.1
        g=dict(object_poses=poses,source_frames=np.array([1,73]),fps=30.)
        nominal=q.copy();nominal[30]=0.
        demo=dict(actions=np.zeros((2,30)),sim_data=[dict(qpos=nominal.copy()) for _ in range(2)])
        return exp,video,g,demo

    def test_reference_adaptation_reads_but_never_writes_state(self):
        exp,video,g,demo=self.fixture()
        q=exp.env.sim.data.qpos.copy();a=demo['actions'].copy()
        c=ReferenceContext(exp,video,demo,g,lambda env:[])
        self.assertEqual(c.features(0).shape,(130,))
        np.testing.assert_array_equal(exp.env.sim.data.qpos,q)
        np.testing.assert_array_equal(demo['actions'],a)
        np.testing.assert_allclose(c.cupref[0],[.004,0,0])

    def test_sampler_logs_raw_residual_not_clipped_action(self):
        exp,video,g,demo=self.fixture()
        policy=SimpleNamespace(get_action=lambda obs:(np.full(30,2.),{'evaluation':np.full(30,.02)}))
        checkpoint=dict(policy=policy,clocks={'first':[.01,2.4,5.]},references={'first':np.full((2,30),.3)},
                        residual_limits={'first':np.full(30,.01)})
        sampler=ResidualSampler(checkpoint,exp,video,lambda env:[])
        np.testing.assert_allclose(sampler[0],.31)
        np.testing.assert_allclose(sampler.latent_actions[0],2.)
        np.testing.assert_allclose(sampler.executed_actions[0],.31)
        path=sampler.path({'rewards':[1.]})
        self.assertEqual(path['observations'].shape,(1,84))
        np.testing.assert_allclose(path['actions'],2.)

    def test_normal_relief_has_correct_direction_and_cap(self):
        from fromrealhand.contact_guard import normal_joint_delta
        jac=np.zeros((3,30));jac[0,10]=.02
        delta=normal_joint_delta(jac,[1,0,0],.001,.0007,.3,.025)
        self.assertGreater(delta[10],0.)
        np.testing.assert_array_equal(delta[:10],0.)
        np.testing.assert_array_equal(normal_joint_delta(jac,[1,0,0],.0005,.0007,.3,.025),0.)
        self.assertLessEqual(np.max(np.abs(normal_joint_delta(jac,[1,0,0],1.,.0007,10.,.025))),.025)


if __name__=='__main__': unittest.main()
