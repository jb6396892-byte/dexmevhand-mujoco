import json
import tempfile
import unittest
from pathlib import Path
from fromrealhand.whole_table.task import NavigationTask,load_config,validate_settings

ROOT=Path(__file__).resolve().parents[1]


class NavigationTaskTests(unittest.TestCase):
    def test_new_entry_keeps_physical_gates(self):
        old=load_config(ROOT/'configs/tabletop-navigation-v4.json')
        new=load_config(ROOT/'configs/tabletop-navigation-v5.json')
        self.assertEqual(new['entry_frames'],[0,420,380])
        changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
        self.assertEqual(changed,{'version','entry_frames','selection_note'})

    def test_configuration_and_parameter_bounds(self):
        config=load_config(ROOT/'configs/tabletop-navigation-v4.json')
        self.assertEqual(config['execution_clearance_m'],.008)
        self.assertEqual(config['carry_egress_clearance_m'],.01)
        self.assertEqual(config['acceptance_max_acceleration_m_s2'],[.5]*3)
        validate_settings(config,[.2,.2,.22],.5,.04)
        for args in [(None,1.1,.025),(None,.4,.025),(None,1.,.024),([0,0,.5],1.,.025)]:
            with self.assertRaises(ValueError):validate_settings(config,*args)

    def planner(self):
        task=NavigationTask(ROOT,ROOT/'unused.pt',ROOT/'configs/tabletop-navigation-v4.json')
        task.config['entry_frames']=[0]
        task.config['grasp_yaws_deg']=[0]
        task.calls=[]
        def candidate(video,layout,output,stop,cfg,observer=None,cancelled=None,entry_frame=0,yaw_deg=0):
            task.calls.append((video,Path(output).name))
            passed=video=='second'
            return dict(passed=passed,status='success' if passed else 'stopped',reason='test',wall_s=0.,steps=1)
        task._candidate=candidate
        return task

    def test_candidate_selection_then_one_execution(self):
        task=self.planner();layout=dict(goal_world_m=[0,0,.2])
        with tempfile.TemporaryDirectory() as path:
            result=task.run(layout,Path(path)/'run')
        self.assertEqual(task.calls,[('first','preview-first'),('second','preview-second'),('second','execution')])
        self.assertEqual(result['actual_executions'],1)
        self.assertEqual(result['execution_resets'],0)
        self.assertEqual(result['selected_video'],'second')

    def test_failed_previews_are_not_success_or_execution(self):
        task=self.planner()
        with tempfile.TemporaryDirectory() as path:
            result=task.run(dict(goal_world_m=[0,0,.2]),Path(path)/'run',mode='fixed')
        self.assertFalse(result['passed']);self.assertEqual(result['actual_executions'],0)
        self.assertEqual(task.calls,[('first','preview-first')])

    def test_cancellation_does_not_launch_candidate(self):
        task=self.planner()
        with tempfile.TemporaryDirectory() as path:
            result=task.run(dict(goal_world_m=[0,0,.2]),Path(path)/'run',cancelled=lambda:True)
        self.assertEqual(result['reason'],'user_stop');self.assertEqual(task.calls,[])

    def test_actual_failure_does_not_reset_and_try_again(self):
        task=self.planner();old=task._candidate
        def candidate(*args,**kwargs):
            row=old(*args,**kwargs)
            if Path(args[2]).name=='execution':row.update(passed=False,status='stopped')
            return row
        task._candidate=candidate
        with tempfile.TemporaryDirectory() as path:
            result=task.run(dict(goal_world_m=[0,0,.2]),Path(path)/'run')
        self.assertFalse(result['passed']);self.assertEqual(result['actual_executions'],1)
        self.assertEqual(len(task.calls),3)

    def test_yaw_keeps_object_delta_equivariant(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from fromrealhand.whole_table.local_adapter import yaw_correspondence
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.1,.2,.3]).as_matrix()
        pose[:3,3]=[.2,.1,.05];estimate=Rotation.from_euler('z',.4).as_matrix()
        original=pose.copy();pivot=np.array([.3,.2,.04])
        moved,r=yaw_correspondence(pose[None],pivot,estimate,180)
        np.testing.assert_allclose(estimate@moved[0,:3,:3].T,r@(estimate@pose[:3,:3].T)@r.T,atol=1e-12)
        np.testing.assert_allclose(moved[0,:3,3],r@(pose[:3,3]-pivot)+pivot)
        np.testing.assert_array_equal(pose,original)
