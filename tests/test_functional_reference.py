import unittest
from unittest.mock import patch
import numpy as np
from fromrealhand.tabletop.functional_reference import ClearanceReference, upright_source_correspondence
from fromrealhand.tabletop.control import VisualReference
from fromrealhand.perception.association import select_tracking_box


class FunctionalReferenceTests(unittest.TestCase):
    def test_correspondence_preserves_world_tips_and_handle_axis(self):
        poses=np.tile(np.eye(4),(3,1,1)); poses[:,:3,:3]=np.diag([1.,-1.,-1.])
        tips=np.arange(45,dtype=float).reshape(3,5,3)/100
        old=poses.copy(); local,result,meta=upright_source_correspondence(tips,poses,1)
        for i in range(3):
            np.testing.assert_allclose(local[i] @ result[i,:3,:3].T,tips[i] @ old[i,:3,:3].T)
        np.testing.assert_array_equal(result[:,:3,0],old[:,:3,0])
        np.testing.assert_array_equal(poses,old)
        self.assertTrue(meta['inverted_source']); self.assertFalse(meta['video_exact'])
        self.assertTrue(np.all(result[:,2,2]>0))

    def test_upright_is_unchanged(self):
        poses=np.tile(np.eye(4),(2,1,1)); tips=np.zeros((2,5,3))
        local,result,meta=upright_source_correspondence(tips,poses,0)
        np.testing.assert_array_equal(result,poses); self.assertFalse(meta['inverted_source'])

    def test_clearance_tapers_without_modifying_finger_targets(self):
        adapter=object.__new__(ClearanceReference); adapter.base=np.eye(4)
        adapter.set_clearance(.025,350,500)
        with patch.object(VisualReference,'desired_qpos',return_value=np.zeros(30)):
            self.assertAlmostEqual(adapter.desired_qpos(0)[2],.025)
        with patch.object(VisualReference,'desired_qpos',side_effect=lambda i:np.zeros(30)):
            self.assertAlmostEqual(adapter.desired_qpos(425)[2],.0125)
            np.testing.assert_array_equal(adapter.desired_qpos(500),np.zeros(30))
        for height in (-.01,float('nan')):
            with self.assertRaises(ValueError): adapter.set_clearance(height,350,500)

    def test_carry_preserves_grasp_targets_and_interpolates_translation(self):
        adapter=object.__new__(ClearanceReference); adapter.base=np.eye(4)
        adapter.actions=np.ones((20,30)); adapter.qpos=np.zeros((20,30))
        adapter.reference_base=np.eye(4); adapter.delta=np.eye(4); adapter.factor=np.ones(6)
        adapter.actions[5]=.2; adapter.qpos[5,6:]=.3
        adapter.set_carry(5,10,20,[.02,0,.12])
        np.testing.assert_allclose(adapter.actions[6:,3:],.2)
        np.testing.assert_allclose(adapter.qpos[6:,6:],.3)
        np.testing.assert_allclose(adapter.qpos[5,:3],0)
        np.testing.assert_allclose(adapter.qpos[10,:3],[0,0,.08])
        np.testing.assert_allclose(adapter.qpos[19,:3],[.02,0,.12])

    def test_custom_goal_warp_changes_only_transport_translation(self):
        adapter=object.__new__(ClearanceReference); adapter.base=np.eye(4)
        adapter.goal_warp=(10,20,np.array([.03,-.02,.01]))
        with patch.object(VisualReference,'desired_qpos',side_effect=lambda i:np.zeros(30)):
            np.testing.assert_allclose(adapter.desired_qpos(10),np.zeros(30))
            np.testing.assert_allclose(adapter.desired_qpos(15)[:3],[.015,-.01,.005])
            np.testing.assert_allclose(adapter.desired_qpos(20)[:3],[.03,-.02,.01])
            np.testing.assert_allclose(adapter.desired_qpos(20)[3:],np.zeros(27))


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.vertices=np.array([[-.1,-.1,1],[.1,.1,1]])
        self.k=np.array([[100.,0,100],[0,100,100],[0,0,1]])
        self.box=dict(box_xyxy=[85,85,115,115])

    def select(self,boxes,previous):
        return select_tracking_box(boxes,previous,self.vertices,self.k,np.eye(4))

    def test_initial_ambiguity_remains_rejected(self):
        with self.assertRaises(ValueError): self.select([self.box,self.box],None)

    def test_distant_false_positive_does_not_change_identity(self):
        box,meta=self.select([dict(box_xyxy=[300,300,340,340]),self.box],np.eye(4))
        self.assertEqual(box,self.box['box_xyxy']); self.assertEqual(meta['selected'],1)

    def test_missing_or_overlapping_candidates_are_rejected(self):
        for boxes in ([],[self.box,self.box],[dict(box_xyxy=[300,300,340,340])]):
            with self.assertRaises(ValueError): self.select(boxes,np.eye(4))

    def test_depth_fallback_is_explicit_and_cannot_override_ambiguity(self):
        box,meta=select_tracking_box([],np.eye(4),self.vertices,self.k,np.eye(4),True)
        self.assertTrue(meta['requires_fresh_depth_registration'])
        with self.assertRaises(ValueError):
            select_tracking_box([self.box,self.box],np.eye(4),self.vertices,self.k,np.eye(4),True)
        with self.assertRaises(ValueError):
            select_tracking_box([],None,self.vertices,self.k,np.eye(4),True)


if __name__=='__main__': unittest.main()
