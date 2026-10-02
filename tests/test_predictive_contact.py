import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from fromrealhand.predictive_contact import PredictiveContactFilter
from fromrealhand.contact_training import contact_costs,bounded_update


class PredictiveContactTests(unittest.TestCase):
    def test_branch_initialization_handles_absent_optional_arrays(self):
        live=SimpleNamespace(ctrl=np.arange(2.),qfrc_applied=np.ones(2),xfrc_applied=np.zeros((1,6)),
             mocap_pos=None,mocap_quat=None,userdata=None,qacc_warmstart=np.full(2,3.))
        branch=SimpleNamespace(ctrl=np.zeros(2),qfrc_applied=np.zeros(2),xfrc_applied=np.ones((1,6)),
             mocap_pos=None,mocap_quat=None,userdata=None,qacc_warmstart=np.zeros(2))
        states=[]
        obj=PredictiveContactFilter.__new__(PredictiveContactFilter)
        obj.exp=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=live,get_state=lambda:'state')))
        obj.sim=SimpleNamespace(data=branch,set_state=states.append,forward=lambda:None)
        obj.initialize_branch()
        self.assertEqual(states,['state'])
        np.testing.assert_array_equal(branch.ctrl,live.ctrl)
        np.testing.assert_array_equal(branch.qacc_warmstart,live.qacc_warmstart)
        branch.ctrl[:] = -1
        np.testing.assert_array_equal(live.ctrl,[0.,1.])

    def test_prediction_executes_reference_future_and_scales_once(self):
        commands=[]
        d=SimpleNamespace(ctrl=np.zeros(30),qpos=np.zeros(37),qvel=np.zeros(36))
        env=SimpleNamespace(control_timestep=.01,model_timestep=.01,act_mid=np.ones(30),act_rng=np.full(30,2.))
        obj=PredictiveContactFilter.__new__(PredictiveContactFilter)
        obj.exp=SimpleNamespace(env=env)
        obj.sim=SimpleNamespace(data=d,step=lambda:commands.append(d.ctrl.copy()))
        obj.initialize_branch=lambda:None
        obj.contacts=lambda *args:(0.,np.zeros(5),{})
        obj.config=dict(horizon_steps=3,reference_lookahead=True)
        obj.reference=np.tile(np.arange(5)[:,None]*.1,(1,30))
        result=obj.predict(np.full(30,.2),control_step=1)
        np.testing.assert_allclose(np.asarray(commands)[:,0],[1.4,1.6,1.8])
        self.assertEqual(result['first'].shape,(73,))

    def test_contact_cost_uses_post_action_phase_and_rejects_nan(self):
        cfg=dict(penetration_target_m=.0007,penetration_scale_m=.0003,penetration_weight=1.,
                 missing_support_weight=.2,contact_phase_source_frame=40.)
        row=dict(penetration_m=.001,source_frame=39.,th_force_n=1.,ff_force_n=1.,
                 mf_force_n=0.,rf_force_n=0.,lf_force_n=0.)
        np.testing.assert_allclose(contact_costs([row,dict(row,source_frame=40.)],cfg),[1.,1.2])
        with self.assertRaises(ValueError):
            contact_costs([dict(row,penetration_m=float('nan'))],cfg)

    def test_legacy_update_is_shrunk_to_measured_kl(self):
        class Policy:
            new=np.array([0.]);old=np.array([0.])
            def get_param_values(self): return self.new.copy()
            def set_param_values(self,x,set_new=True,set_old=True):
                if set_new: self.new=x.copy()
                if set_old: self.old=x.copy()
        policy=Policy()
        def update(paths):
            policy.set_param_values(np.array([2.]));return [1.,0.,1.,1.]
        agent=SimpleNamespace(policy=policy,train_from_paths=update,
            logger=SimpleNamespace(log={'kl_dist':[4.]}),
            kl_old_new=lambda obs,actions:torch.tensor([(policy.new[0]-policy.old[0])**2]))
        _,audit=bounded_update(agent,[dict(observations=np.zeros((1,2)),actions=np.zeros((1,1)))],.25)
        self.assertEqual(audit['accepted_fraction'],.25)
        self.assertEqual(audit['measured_kl'],.25)
        np.testing.assert_array_equal(policy.new,[.5])
        np.testing.assert_array_equal(policy.old,policy.new)


if __name__=='__main__': unittest.main()
