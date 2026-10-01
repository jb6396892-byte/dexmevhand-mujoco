import unittest
import numpy as np
import json
import importlib
import sys
from types import SimpleNamespace
from pathlib import Path
from fromrealhand.corrective_learning import (CorrectiveActions,phase_weights,tracking_delta,
                                              aligned_phase_indices,aligned_phase_weights)


class CorrectiveTests(unittest.TestCase):
    def test_zero_error_is_zero_correction(self):
        q=np.arange(30)*.01
        np.testing.assert_array_equal(tracking_delta(q,q,q,q,np.ones(30),.5),np.zeros(30))

    def test_feedback_direction_and_single_conversion(self):
        z=np.zeros(30);r=np.ones(30)*.001
        np.testing.assert_allclose(tracking_delta(z,r,z,z,np.full(30,4.),.25),.001)

    def test_errors_are_bounded(self):
        z=np.zeros(30)
        d=tracking_delta(z,np.full(30,1e6),z,np.full(30,1e6),np.ones(30),.5)
        np.testing.assert_allclose(d[:3],.015);np.testing.assert_allclose(d[3:],.06)
        with self.assertRaises(ValueError): tracking_delta(z,z,z,z,np.full(30,np.nan),.5)

    def test_phase_mass_not_frame_count(self):
        masses=[.15,.15,.2,.3,.2];w=phase_weights(1450,masses)
        self.assertAlmostEqual(w.sum(),1.)
        self.assertAlmostEqual(w[:50].sum(),.15)
        self.assertAlmostEqual(w[50:550].sum(),.15)
        self.assertTrue(np.all(w>0))

    def test_new_test_definition_differs_from_old_test(self):
        root=Path(__file__).resolve().parents[1]
        old=json.loads((root/'configs/v10-study.json').read_text())
        new=json.loads((root/'configs/v11-study.json').read_text())
        spec=new['heldout']['axis_cases_per_video']
        self.assertTrue(set(spec['cup_yaw_deg']).isdisjoint({c['cup_yaw_deg'] for c in old['heldout_per_video']}))
        self.assertNotIn(.0035,spec['cup_xy_offsets_m'])
        self.assertNotIn(.017,spec['goal_xy_offsets_m'])
        self.assertFalse(new['heldout']['tuning_after_test'])

    def test_safety_pulse_preserves_approach_and_hold(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts'))
        module=importlib.import_module('71_develop_safe_experts')
        problem=object.__new__(module.SafetyProblem)
        problem.demo={'actions':np.zeros((1450,30))}
        problem.scales=np.ones(30);problem.conversion=np.ones(30)
        parameters=np.zeros(30);parameters[29]=.03;parameters[23]=-.02
        actions=problem.action_sequence(parameters)
        np.testing.assert_array_equal(actions[:631],0.)
        np.testing.assert_array_equal(actions[1120:],0.)
        np.testing.assert_allclose(actions[800,[29,23]],[.03,-.02])
        self.assertLess(np.max(np.abs(np.diff(actions,axis=0))),.001)

    def test_teacher_labels_do_not_write_simulator_state(self):
        q=np.zeros(37);q[33]=1.;v=np.zeros(36)
        model=SimpleNamespace(actuator_biasprm=np.tile([0.,-1.,0.],(30,1)),
                              actuator_gainprm=np.ones((30,3)))
        env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=q,qvel=v)),
                            act_rng=np.ones(30),control_timestep=.01,
                            _get_observations=lambda:np.zeros(39))
        exp=SimpleNamespace(model=model,env=env,duration=1.)
        video={'horizon':1,'control':{'time_scale':5.},'id':0}
        target=q.copy();target[:30]=.01
        expert={'actions':np.zeros((1,30)),'sim_data':[{'qpos':target,'qvel':v.copy()}]}
        controller=CorrectiveActions(exp,video,expert,np.zeros((1,30)),np.full(30,.0002),None,.1,1.)
        before=q.copy()
        np.testing.assert_allclose(controller[0],.0002)
        np.testing.assert_array_equal(q,before)
        np.testing.assert_array_equal(v,0.)
        self.assertEqual(np.shape(controller.features),(1,84))
        self.assertGreater(np.linalg.norm(controller.feedback[0]),0.)

    def test_feedback_phase_report_partitions_frames(self):
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts'))
        module=importlib.import_module('72_train_corrective_residual')
        bins=[0,.5,5.5,7.1666666667,9.6666666667,None]
        report=module.feedback_phases({'feedback':np.zeros((1450,30))},bins)
        self.assertEqual(sum(r['frames'] for r in report.values()),1450)
        self.assertEqual(report['prepare']['frames'],50)
        self.assertEqual(report['approach']['frames'],500)
        self.assertTrue(all(r['mean_abs_action']==0 for r in report.values()))

    def test_residual_gain_scales_network_not_reference(self):
        import torch
        root=Path(__file__).resolve().parents[1]
        sys.path.insert(0,str(root/'scripts'))
        module=importlib.import_module('76_select_residual_gain')
        from mjrl.policies.gaussian_mlp import MLP
        policy=MLP(SimpleNamespace(observation_dim=84,action_dim=30),hidden_sizes=(4,4),seed=4)
        for model in (policy.model,policy.old_model):
            model.set_transformations(np.ones(84),np.full(84,2.),np.full(30,.1),np.full(30,.2))
        x=torch.ones((1,84));before=policy.model(x).detach().numpy().copy()
        reference=np.full((2,30),.3)
        checkpoint={'policy':policy,'method':'residual_bc','references':{'second':reference.copy()}}
        module.scale_residual(checkpoint,.25)
        np.testing.assert_allclose(policy.model(x).detach().numpy(),before*.25,rtol=1e-6)
        np.testing.assert_array_equal(checkpoint['references']['second'],reference)
        with self.assertRaises(ValueError): module.scale_residual(checkpoint,0)

    def test_aligned_weights_follow_nonlinear_clock_for_both_videos(self):
        masses=[.15,.15,.2,.3,.2]
        for start,length,expected in [(3,1417,[562,652,793]),(1,1450,[588,677,819])]:
            geometry={'source_frames':np.arange(start,74),'fps':30.}
            ids=aligned_phase_indices(length,geometry,5.)
            weights=aligned_phase_weights(length,masses,geometry,5.)
            self.assertEqual([int(np.flatnonzero(ids>=i)[0]) for i in [2,3,4]],expected)
            np.testing.assert_allclose([weights[ids==i].sum() for i in range(5)],masses)
            self.assertTrue(np.all(weights>0))
            self.assertEqual(int(np.sum(ids==0)),50)
            self.assertLess(phase_weights(length,masses)[ids==2].sum(),.12)

    def test_aligned_sampler_rejects_invalid_clocks(self):
        geometry={'source_frames':[1,73],'fps':30.}
        with self.assertRaises(ValueError): aligned_phase_indices(1450,geometry,0.)
        with self.assertRaises(ValueError): aligned_phase_indices(1450,geometry,5.,source_boundaries=[40,30,55])
        with self.assertRaises(ValueError): aligned_phase_weights(1450,[1,1,1,1,np.nan],geometry,5.)


if __name__=='__main__': unittest.main()
