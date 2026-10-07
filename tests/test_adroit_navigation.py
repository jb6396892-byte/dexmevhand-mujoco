import unittest
from types import SimpleNamespace
import numpy as np
from fromrealhand.whole_table.navigation import plan,segments_clear,inflated_boxes,NavigationRejected

class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.cfg=dict(workspace_min_m=[-1,-1,0],workspace_max_m=[1,1,1],grid_resolution_m=.1,clearance_m=.02)
        self.hand=np.array([[-.05,-.15,-.05],[.05,.05,.05]])

    def test_thin_obstacle_swept_edge(self):
        box=np.array([[[-.001,-1,0],[.001,1,1]]])
        self.assertFalse(segments_clear([-.1,0,.5],[.1,0,.5],box)[0])
        self.assertTrue(segments_clear([-.1,0,.5],[-.1,.5,.5],box)[0])

    def test_hand_not_point(self):
        boxes=inflated_boxes(self.hand,[dict(bounds=[[0,0,0],[.1,.1,.3]])],.02)
        self.assertFalse(segments_clear([-.04,0,.2],[-.04,0,.2],boxes)[0])

    def test_retreat_planning_margin_is_not_physical_collision(self):
        from fromrealhand.whole_table.loaded_motion import margin_conflicts
        bridge=SimpleNamespace(hand_envelope=self.hand,
            obstacles=lambda:[dict(name='box',bounds=[[0,0,0],[.1,.1,.3]])])
        self.assertEqual(margin_conflicts(bridge,[-.07,0,.2],.025),['box'])
        self.assertEqual(margin_conflicts(bridge,[-.07,0,.2],.01),[])
        self.assertEqual(margin_conflicts(bridge,[-.055,0,.2],.008),['box'])

    def test_part_union_preserves_empty_gap(self):
        parts=np.array([[[-.2,-.05,-.05],[-.1,.05,.05]],[[.1,-.05,-.05],[.2,.05,.05]]])
        boxes=inflated_boxes(parts,[dict(bounds=[[-.02,-.02,.48],[.02,.02,.52]])],.01)
        self.assertTrue(segments_clear([0,0,.5],[0,0,.5],boxes)[0])

    def test_local_features_translation_invariant(self):
        from fromrealhand.tabletop.training_inputs import features
        pose=np.eye(4);pose[:3,3]=[.02,.03,.1]
        palm=np.array([0.,-.1,.2]);goal=np.array([.1,.05,.16]);zero=np.zeros(30)
        before=features(zero,zero,palm,pose,goal,'second','grasp',3,(0,10),zero,zero)
        delta=np.array([.2,-.2,0]);moved=pose.copy();moved[:3,3]+=delta
        after=features(zero,zero,palm+delta,moved,goal+delta,'second','grasp',3,(0,10),zero,zero)
        np.testing.assert_allclose(before,after,atol=1e-7)

    def test_initial_frame_translation_preserves_preplaced_cup(self):
        from unittest.mock import Mock,patch
        from fromrealhand.whole_table.local_adapter import translate_initial_scene
        data=SimpleNamespace(ctrl=np.zeros(30),get_joint_qpos=Mock(return_value=np.array([.2,0,.04,1,0,0,0])),
                             set_joint_qpos=Mock())
        model=SimpleNamespace(body_pos=np.zeros((1,3)),body_name2id=lambda name:0)
        sim=SimpleNamespace(model=model,data=data,get_state=Mock(return_value='state'),
                            set_state=Mock(),forward=Mock())
        env=SimpleNamespace(sim=sim,reference_base=np.eye(4))
        poses=np.eye(4)[None];scene={}
        module=SimpleNamespace(functions=SimpleNamespace(mj_setConst=Mock()))
        with patch.dict('sys.modules',{'mujoco_py':module}):
            result=translate_initial_scene(env,[.2,0,0],poses,scene,translate_object=False)
        data.set_joint_qpos.assert_not_called()
        np.testing.assert_allclose(model.body_pos[0],[.2,0,0])
        np.testing.assert_allclose(result[0,:3,3],[.2,0,0])
        np.testing.assert_allclose(poses[0,:3,3],0)
        self.assertFalse(scene['cup_translated_at_initialization'])

    def test_initial_frame_translation_keeps_legacy_cup_shift(self):
        from unittest.mock import Mock,patch
        from fromrealhand.whole_table.local_adapter import translate_initial_scene
        data=SimpleNamespace(ctrl=np.zeros(30),get_joint_qpos=Mock(return_value=np.array([0.,0,.04,1,0,0,0])),
                             set_joint_qpos=Mock())
        model=SimpleNamespace(body_pos=np.zeros((1,3)),body_name2id=lambda name:0)
        sim=SimpleNamespace(model=model,data=data,get_state=Mock(),set_state=Mock(),forward=Mock())
        env=SimpleNamespace(sim=sim,reference_base=np.eye(4));scene={}
        module=SimpleNamespace(functions=SimpleNamespace(mj_setConst=Mock()))
        with patch.dict('sys.modules',{'mujoco_py':module}):
            translate_initial_scene(env,[-.2,.04,0],np.eye(4)[None],scene)
        np.testing.assert_allclose(data.set_joint_qpos.call_args[0][1][:3],[-.2,.04,.04])
        self.assertTrue(scene['cup_translated_at_initialization'])

    def test_engineering_gate_keeps_strict_result(self):
        from fromrealhand.whole_table.navigation_runner import navigate
        cfg=dict(self.cfg,timestep_s=.01,control_period_s=.02,max_velocity_m_s=[.1]*3,
            max_acceleration_m_s2=[.05]*3,max_jerk_m_s3=[.2]*3,position_tolerance_m=.003,
            tracking_tolerance_m=.015,posture_tolerance_rad=.025,execution_clearance_m=.001,
            acceptance_max_acceleration_m_s2=[.5]*3)
        scene=SimpleNamespace(config=cfg,hand_envelope=self.hand,target=np.array([-.7,0,.5]),
            sim=SimpleNamespace(data=SimpleNamespace(time=0.)))
        scene.position=lambda:scene.target.copy()
        scene.obstacles=lambda:[]
        scene.velocity=lambda:np.zeros(3)
        scene.acceleration=lambda:np.array([0,0,.3])
        scene.contacts=lambda:dict(hand_environment_contacts=0,posture_error_rad=0.)
        scene.hand_shapes=lambda:self.hand+scene.position()
        def step(target):scene.target=np.array(target);scene.sim.data.time+=.01
        scene.step=step
        result=navigate(scene,[-.6,0,.5])
        self.assertTrue(result['passed']);self.assertFalse(result['strict_passed'])
        self.assertFalse(result['strict_checks']['acceleration'])
        scene.acceleration=lambda:np.array([0,0,.6])
        self.assertFalse(navigate(scene,[-.5,0,.5])['passed'])

    def test_navigation_geometry_excludes_visual_only_meshes(self):
        from fromrealhand.whole_table.hand_scene import HandScene
        scene=HandScene.__new__(HandScene)
        scene.config=dict(collision_geometry_only=True);scene.hand_geoms=[0,1]
        scene.sim=SimpleNamespace(model=SimpleNamespace(geom_contype=[0,1],geom_conaffinity=[0,0],
            geom_type=[6,6],geom_size=np.array([[10,10,10],[.01,.02,.03]])),
            data=SimpleNamespace(geom_xpos=np.zeros((2,3)),geom_xmat=np.tile(np.eye(3).ravel(),(2,1))))
        boxes=scene.hand_shapes()
        self.assertEqual(boxes.shape,(1,2,3))
        np.testing.assert_allclose(boxes[0],[[ -.01,-.02,-.03],[.01,.02,.03]])

    def test_direct_and_detour(self):
        start=[-.7,0,.2];goal=[.7,0,.2]
        self.assertTrue(plan(start,goal,self.hand,[],self.cfg)['direct'])
        obstacles=[dict(name='wall',bounds=[[-.1,-1,0],[.1,1,.4]])]
        report=plan(start,goal,self.hand,obstacles,self.cfg)
        self.assertFalse(report['direct'])
        points=np.array(report['waypoints'])
        self.assertTrue(np.all(segments_clear(points[:-1],points[1:],inflated_boxes(self.hand,obstacles,.02))))
        self.assertGreater(points[:,2].max(),.4)

    def test_blocked_and_invalid(self):
        obstacles=[dict(name='wall',bounds=[[-.1,-2,-1],[.1,2,2]])]
        with self.assertRaises(NavigationRejected):plan([-.7,0,.2],[.7,0,.2],self.hand,obstacles,self.cfg)
        for goal in ([2,0,.3],[0,float('nan'),.3],[0,0,.2]):
            with self.assertRaises(NavigationRejected):plan([-.7,0,.2],goal,self.hand,obstacles,self.cfg)

if __name__=='__main__':unittest.main()
