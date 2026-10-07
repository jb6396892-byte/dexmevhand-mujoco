"""Pure visual-control and motor-step safety contracts."""
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from fromrealhand.tabletop.control import rigid, pose_from_estimate, check_pose_jump, VisualReference

ROOT = Path(__file__).resolve().parents[1]


class VisualControlContracts(unittest.TestCase):
    def test_detection_boxes_reject_invalid_or_off_image_regions(self):
        from fromrealhand.perception.tracking import bounded_box
        for box in ([-100,-100,-90,-90], [700,100,800,200], [20,30,10,40],
                    [10,20,30,20], [0,0,float('nan'),30], [1,2,3]):
            with self.assertRaises(ValueError): bounded_box(box,(480,640))

    def test_detection_box_clips_without_negative_numpy_slices(self):
        from fromrealhand.perception.tracking import bounded_box
        np.testing.assert_array_equal(bounded_box([-10,-20,650,500],(480,640)),[0,0,640,480])

    def setUp(self):
        self.cfg = json.loads((ROOT/'configs/tabletop-control-candidate.json').read_text())
        self.pose = np.eye(4); self.pose[2,3] = .04
        self.row = dict(accepted=True, reason='pose_accepted', camera_time_s=1.,
                        T_world_object=self.pose.tolist(), fitness=.98, rmse_m=.001)

    def test_default_locked(self):
        self.assertFalse(self.cfg['automatic_execution_enabled'])
        self.assertEqual(self.cfg['status'], 'development_tested_not_admitted')

    def test_pose_gate_rejects_stale_future_invalid_and_nan(self):
        for row, clock in [(self.row,2.),(self.row,0.),(dict(self.row,accepted=False),1.),
                           (dict(self.row,fitness=float('nan')),1.)]:
            with self.assertRaises(ValueError): pose_from_estimate(row,clock,self.cfg)

    def test_pose_requires_rigid_transform(self):
        for changed in (np.zeros((4,4)),np.diag([-1.,1,1,1]),np.ones((3,3))):
            with self.assertRaises(ValueError): rigid(changed)
        np.testing.assert_allclose(pose_from_estimate(self.row,1.1,self.cfg),self.pose)

    def test_workspace_and_jump_rejection(self):
        pose=self.pose.copy(); pose[0,3]=2
        with self.assertRaises(ValueError):
            pose_from_estimate(dict(self.row,T_world_object=pose.tolist()),1.,self.cfg)
        with self.assertRaises(ValueError): check_pose_jump(self.pose,pose,self.cfg)

    def adapter(self, dx=0):
        estimate=self.pose.copy(); estimate[0,3]+=dx
        model=dict(joint_range=np.tile([-2.,2.],(30,1)), gain=np.ones(30),
                   bias=np.tile([0.,-1.,0.],(30,1)), action_range=np.ones(30))
        return VisualReference(np.zeros((301,30)),np.zeros((301,30)),self.pose,estimate,
                               np.eye(4),model,self.cfg,.01)

    def test_zero_transform_preserves_action(self):
        np.testing.assert_allclose(self.adapter().action(250,np.zeros(30)),np.zeros(30),atol=1e-12)

    def test_translation_blends_without_modifying_fingers(self):
        adapter=self.adapter(.02)
        np.testing.assert_allclose(adapter.action(0,np.zeros(30)),np.zeros(30),atol=1e-12)
        result=adapter.action(250,np.zeros(30))
        self.assertGreater(result[0],0)
        np.testing.assert_array_equal(result[6:],np.zeros(24))
        np.testing.assert_allclose(adapter.goal([0,0,.2]),[.02,0,.2])

    def test_correction_does_not_hide_saturation(self):
        adapter=self.adapter(.02); adapter.actions[:,0]=1
        with self.assertRaises(ValueError): adapter.action(250,np.zeros(30))

    def test_reference_envelope_rejection(self):
        with self.assertRaises(ValueError): self.adapter(.5)

    def test_small_tilt_is_preserved_not_discarded(self):
        from scipy.spatial.transform import Rotation
        adapter=self.adapter()
        adapter.delta[:3,:3]=Rotation.from_euler('x',5,degrees=True).as_matrix()
        adapter.rotation_vector=Rotation.from_matrix(adapter.delta[:3,:3]).as_rotvec()
        out=adapter.action(250,np.zeros(30))
        self.assertGreater(abs(out[3]),.01)
        np.testing.assert_array_equal(out[6:],np.zeros(24))

    def test_bounded_orientation_does_not_change_measured_object_pose(self):
        from scipy.spatial.transform import Rotation
        estimate=self.pose.copy(); estimate[:3,:3]=Rotation.from_euler('z',10,degrees=True).as_matrix()
        estimate[:3,3]+=[.01,.01,0]
        model=dict(joint_range=np.tile([-2.,2.],(30,1)),gain=np.ones(30),
            bias=np.tile([0.,-1.,0.],(30,1)),action_range=np.ones(30))
        adapter=VisualReference(np.zeros((301,30)),np.zeros((301,30)),self.pose,estimate,np.eye(4),model,
            dict(self.cfg,orientation_transfer_fraction=.5),.01)
        np.testing.assert_array_equal(adapter.estimate,estimate)
        np.testing.assert_allclose(adapter.goal(self.pose[:3,3]),estimate[:3,3])
        self.assertAlmostEqual(np.linalg.norm(adapter.rotation_vector),np.deg2rad(5))

    def test_installation_coordinate_change_preserves_world_hand(self):
        adapter=self.adapter(); shift=np.array([-.1,.05,0.])
        adapter.base[:3,3]=-shift
        current=np.zeros(30); current[:3]=shift
        action=adapter.action(0,current)
        np.testing.assert_allclose(action[:3],shift,atol=1e-12)
        np.testing.assert_allclose(adapter.base[:3,:3]@current[:3]+adapter.base[:3,3],np.zeros(3))

    def test_preflight_checks_entire_horizon(self):
        adapter=self.adapter(); adapter.qpos[-1,0]=3
        with self.assertRaises(ValueError): adapter.preflight(len(adapter.actions))

    def motor_env(self):
        from fromrealhand.tabletop.control_scene import ControlEnv
        sim=SimpleNamespace(model=SimpleNamespace(actuator_ctrlrange=np.tile([2.,6.],(30,1)),
            opt=SimpleNamespace(timestep=.002)),data=SimpleNamespace(ctrl=np.zeros(30)),calls=0)
        def step(): sim.calls+=1
        sim.step=step
        return ControlEnv(sim,.01),sim

    def test_normalized_control_scaled_exactly_once(self):
        env,sim=self.motor_env(); audits=[]
        env.step(np.ones(30)*.5,lambda:audits.append(sim.calls))
        np.testing.assert_array_equal(sim.data.ctrl,np.full(30,5.))
        self.assertEqual(audits,[1,2,3,4,5])

    def test_handoff_preserves_starting_force_and_ends_at_policy(self):
        env,sim=self.motor_env()
        sim.model.actuator_gainprm=np.tile([2.,0.,0.],(30,1))
        sim.model.actuator_biasprm=np.zeros((30,3))
        sim.data.qpos=np.zeros(30);sim.data.qvel=np.zeros(30);sim.data.time=0.
        controls=[]
        def step():
            controls.append(sim.data.ctrl.copy());sim.data.time+=.002
        sim.step=step
        env.motor_handoff=dict(start=0.,duration=.004,gain=np.full(30,4.),
            bias=np.zeros((30,3)),ctrl=np.full(30,2.))
        env.step(np.full(30,.5),lambda:None)
        np.testing.assert_allclose(controls[0]*2.,np.full(30,8.))
        np.testing.assert_allclose(controls[-1],np.full(30,5.))
        self.assertIsNone(env.motor_handoff)

    def test_handoff_authority_failure_prevents_integration(self):
        env,sim=self.motor_env()
        sim.model.actuator_gainprm=np.tile([1.,0.,0.],(30,1))
        sim.model.actuator_biasprm=np.zeros((30,3))
        sim.data.qpos=np.zeros(30);sim.data.qvel=np.zeros(30);sim.data.time=0.
        env.motor_handoff=dict(start=0.,duration=1.,gain=np.full(30,100.),
            bias=np.zeros((30,3)),ctrl=np.ones(30))
        with self.assertRaises(ValueError):env.step(np.zeros(30),lambda:None)
        self.assertEqual(sim.calls,0)

    def test_invalid_action_does_not_step(self):
        for action in [np.ones(30)*1.1,np.full(30,np.nan),np.zeros(29)]:
            env,sim=self.motor_env()
            with self.assertRaises(ValueError): env.step(action,lambda:None)
            self.assertEqual(sim.calls,0)

    def test_audit_failure_stops_remaining_substeps(self):
        env,sim=self.motor_env()
        def stop(): raise RuntimeError('injected safety stop')
        with self.assertRaises(RuntimeError): env.step(np.zeros(30),stop)
        self.assertEqual(sim.calls,1)

    def test_vision_wait_stop_and_errors(self):
        import queue,threading
        from fromrealhand.tabletop.vision_client import VisionClient
        client=VisionClient.__new__(VisionClient)
        client.timeout=.01; client.stop=threading.Event(); client.responses=queue.Queue()
        client.stop.set()
        with self.assertRaises(RuntimeError): client.wait()
        client.stop.clear(); client.responses.put(dict(type='error',reason='fault'))
        with self.assertRaises(RuntimeError): client.wait()
        with self.assertRaises(TimeoutError): client.wait()


if __name__ == '__main__': unittest.main()
