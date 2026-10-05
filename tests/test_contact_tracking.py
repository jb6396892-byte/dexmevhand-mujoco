import unittest
from types import SimpleNamespace
import numpy as np
from fromrealhand.tabletop.contact_control import opposition_from_vectors, ContactTracker, FINGERS, object_pose


class ContactTrackingTests(unittest.TestCase):
    def vectors(self):
        return {f:np.zeros(3) for f in FINGERS}

    def test_three_contacts_need_opposing_normals(self):
        v=self.vectors(); v['th']=np.array([1.,0,0]); v['ff']=np.array([1.,0,0]); v['mf']=v['ff']
        self.assertFalse(opposition_from_vectors(v)['opposition'])
        v['ff']=np.array([-1.,0,0])
        self.assertFalse(opposition_from_vectors(v)['opposition'])
        v['mf']=np.array([-.8,.2,0])
        self.assertTrue(opposition_from_vectors(v)['opposition'])

    def test_unloaded_contact_does_not_count(self):
        v=self.vectors(); v['th']=np.array([1.,0,0]); v['ff']=np.array([-.001,0,0]); v['mf']=np.array([-1.,0,0])
        self.assertFalse(opposition_from_vectors(v)['opposition'])

    def test_nonfinite_normal_rejected(self):
        v=self.vectors(); v['th']=np.full(3,np.nan)
        with self.assertRaises(ValueError): opposition_from_vectors(v)

    def test_pose_uses_reference_free_joint_convention(self):
        q=np.zeros(37); q[30:33]=[.1,.2,.3]; q[33]=1
        t=object_pose(q)
        np.testing.assert_allclose(t[:3,3],[.1,.2,.3]); np.testing.assert_allclose(t[:3,:3],np.eye(3))

    def test_bad_gains_and_reference_rejected(self):
        a=SimpleNamespace(actions=np.zeros((4,30)))
        for kp in (-1,float('nan')):
            with self.assertRaises(ValueError): ContactTracker(a,np.zeros((4,5,3)),kp=kp)
        with self.assertRaises(ValueError): ContactTracker(a,np.zeros((3,5,3)))

    def controller_fixture(self, nominal=0.):
        target=np.zeros(30); target[:6]=.02
        adapter=SimpleNamespace(actions=np.zeros((1,30)),dt=.01,
            desired_qpos=lambda i:target.copy(),action=lambda i,q:np.full(30,nominal))
        bias=np.zeros((30,3)); bias[:,1]=-1.
        model=SimpleNamespace(actuator_ctrlrange=np.tile([-1.,1.],(30,1)),
            actuator_gainprm=np.ones((30,1)),actuator_biasprm=bias,nv=30,jnt_range=np.tile([-1.,1.],(30,1)))
        data=SimpleNamespace(qpos=np.zeros(30),qvel=np.zeros(30),get_site_xpos=lambda name:np.zeros(3))
        return ContactTracker(adapter,np.zeros((1,5,3)),kp=0,root_gain=1),SimpleNamespace(model=model,data=data)

    def test_selected_controller_does_not_add_finger_closure(self):
        controller,sim=self.controller_fixture()
        action=controller.action(0,sim,np.eye(4))
        np.testing.assert_allclose(action[:6],.02)
        np.testing.assert_array_equal(action[6:],np.zeros(24))

    def test_saturation_stops_instead_of_clipping(self):
        controller,sim=self.controller_fixture(.99)
        with self.assertRaises(ValueError): controller.action(0,sim,np.eye(4))

    def test_carry_uses_measured_hand_and_fresh_visual_frames_only(self):
        controller,sim=self.controller_fixture()
        controller.local_tips=np.zeros((400,5,3))
        controller.adapter.base=np.eye(4)
        controller.action(0,sim,np.eye(4))
        controller.start_carry(sim,np.eye(4),[.03,0,.01],0,100,400)
        first=controller.action(250,sim,np.eye(4),'001')
        offset=controller.carry['correction'].copy()
        self.assertLessEqual(np.linalg.norm(offset),.003000001)
        np.testing.assert_array_equal(controller.action(250,sim,np.eye(4),'001'),first)
        controller.action(250,sim,np.eye(4),'002')
        self.assertGreater(np.linalg.norm(controller.carry['correction']),np.linalg.norm(offset))
        np.testing.assert_array_equal(controller.carry['q'],sim.data.qpos)

    def test_carry_rejects_unreachable_goal(self):
        controller,sim=self.controller_fixture(); controller.adapter.base=np.eye(4)
        controller.action(0,sim,np.eye(4))
        with self.assertRaises(ValueError): controller.start_carry(sim,np.eye(4),[2.,0,0],0,100,400)


if __name__=='__main__': unittest.main()
