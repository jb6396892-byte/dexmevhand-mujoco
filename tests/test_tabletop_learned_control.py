import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mjrl.policies.gaussian_mlp import MLP
from fromrealhand.routed_residual import RoutedNetwork
from fromrealhand.tabletop.learned_control import LearnedControl
from fromrealhand.tabletop.training_inputs import FEATURE_DIM, FEATURE_VERSION
from fromrealhand.tabletop.control import perception_interval


class LearnedControlTests(unittest.TestCase):
    def test_lift_sampling_is_faster_without_changing_other_phases(self):
        cfg=dict(perception_period_sim_s=.2,max_pose_sim_age_s=.25,perception_period_by_phase_s={'lift':.1})
        self.assertEqual(perception_interval(cfg,'lift'),.1)
        self.assertEqual(perception_interval(cfg,'transport'),.2)
        cfg['perception_period_by_phase_s']['lift']=0
        with self.assertRaises(ValueError): perception_interval(cfg,'lift')

    def make(self, directory, mode):
        base=MLP(SimpleNamespace(observation_dim=FEATURE_DIM,action_dim=30),hidden_sizes=(8,8),seed=0).model
        model=RoutedNetwork(base,8) if mode=='routed-residual' else base
        for parameter in model.parameters(): parameter.data.zero_()
        path=Path(directory)/'policy.pt'
        torch.save(dict(state_dict=model.state_dict(),mode=mode,feature_version=FEATURE_VERSION,
            feature_dim=FEATURE_DIM,hidden_sizes=[8,8],
            normalization=dict(mean=[0.]*FEATURE_DIM,std=[1.]*FEATURE_DIM)),str(path))
        return LearnedControl(path)

    def test_zero_residual_preserves_reference_without_double_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=self.make(directory,'routed-residual')
            for video in ('first','second'):
                for phase in ('reach','grasp','lift','transport'):
                    reference=np.linspace(-.9,.9,30)
                    np.testing.assert_allclose(policy.action(np.zeros(FEATURE_DIM),reference,video,phase),reference)
            self.assertEqual(policy.calls,8)

    def test_absolute_mode_does_not_add_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=self.make(directory,'shared')
            np.testing.assert_array_equal(policy.action(np.zeros(FEATURE_DIM),np.ones(30)*.4,'first','grasp'),0)

    def test_invalid_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=self.make(directory,'shared')
            with self.assertRaises(ValueError):
                policy.action(np.full(FEATURE_DIM,np.nan),np.zeros(30),'first','reach')

    def test_saturation_is_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=self.make(directory,'routed-residual')
            result=policy.action(np.zeros(FEATURE_DIM),np.ones(30)*1.1,'second','lift')
            np.testing.assert_array_equal(result,1)
            self.assertEqual(policy.clipped_calls,1)

    def test_structured_checkpoint_uses_raw_physical_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'structured.pt'
            torch.save(dict(mode='structured-residual',feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,
                gain=[2.]*6+[0.]*24,normalization=dict(mean=[3.]*FEATURE_DIM,std=[7.]*FEATURE_DIM)),str(path))
            policy=LearnedControl(path); x=np.zeros(FEATURE_DIM); x[109:115]=.1
            action=policy.action(x,np.zeros(30),'second','transport')
            np.testing.assert_allclose(action,np.r_[np.ones(6)*.2,np.zeros(24)])


if __name__=='__main__': unittest.main()
