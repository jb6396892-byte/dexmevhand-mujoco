import unittest
import numpy as np
from fromrealhand.action_recovery import force_to_action


class ActionRecoveryTest(unittest.TestCase):
    def test_root_equilibrium_is_not_joint_position(self):
        q=np.array([.2]); gain=np.array([500.]); bias=np.array([[0.,-200.,0.]])
        action=force_to_action(np.zeros(1),q,np.zeros(1),gain,bias,np.zeros(1),np.array([.25]))
        control=action*.25
        np.testing.assert_allclose(control,[.08])
        np.testing.assert_allclose(gain*control+bias[:,1]*q,[0],atol=1e-12)

    def test_affine_force_and_single_normalization(self):
        force=np.array([2.]); q=np.array([.1]); velocity=np.array([.3])
        bias=np.array([[.2,-3.,-.5]])
        action=force_to_action(force,q,velocity,np.array([10.]),bias,np.array([.1]),np.array([.5]))
        actual=10*(action*.5+.1)+bias[:,0]+bias[:,1]*q+bias[:,2]*velocity
        np.testing.assert_allclose(actual,force)

    def test_clips_outside_control_range(self):
        a=force_to_action(np.array([100.]),np.zeros(1),np.zeros(1),np.ones(1),np.zeros((1,3)),np.zeros(1),np.ones(1))
        np.testing.assert_array_equal(a,[1.])
