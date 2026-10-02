import json
from pathlib import Path
import unittest
import numpy as np
from fromrealhand.skill_pipeline import confirmed_segments, handoff_audit, GuardedSkill, geometry_group


class PipelineTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.cfg = dict(json.loads((root/'configs/stage4-skills.json').read_text()),
                        **json.loads((root/'configs/stage4-pipeline-v2.json').read_text()))
        self.cfg.update(confirmation_steps=dict(reach=2,grasp=3,lift=3),
                        handoff_window_steps=2,support_loss_stop_steps=2,transport_hold_steps=2)
        self.empty = dict(bottom_m=0.,target_distance_m=.2,scene_penetration_m=0.,joint_violation_rad=0.,
            finite=True,source_frame=0.,th_force_n=0.,ff_force_n=0.,mf_force_n=0.,rf_force_n=0.,lf_force_n=0.)
        self.contact = dict(self.empty,ff_force_n=1.)
        self.grasp = dict(self.contact,th_force_n=1.,mf_force_n=1.)
        self.lift = dict(self.grasp,bottom_m=.06)
        self.goal = dict(self.lift,target_distance_m=.02)

    def test_causal_confirmation_ignores_future_rows(self):
        rows = [self.contact]*2+[self.grasp]*3+[self.lift]*3+[self.goal]*4
        segments = confirmed_segments(rows,self.cfg,.01)
        self.assertEqual([s['stop'] for s in segments],[2,5,8,12])
        altered = rows[:8]+[self.empty]*4
        self.assertEqual([s['stop'] for s in confirmed_segments(altered,self.cfg,.01)],[2,5,8,12])
        self.assertFalse(handoff_audit(altered,segments,self.cfg)[-1]['passed'])

    def test_support_loss_requires_persistence_but_stops(self):
        guard = GuardedSkill('transport',self.cfg,self.lift)
        self.assertEqual(guard.update(self.empty),'running')
        guard.update(self.goal)
        self.assertEqual(guard.lost,0)
        guard.update(self.empty)
        self.assertEqual(guard.update(self.empty),'failed')
        self.assertEqual(guard.reason,'persistent_support_loss')

    def test_safety_overrides_contact_grace(self):
        guard = GuardedSkill('lift',self.cfg,self.grasp)
        self.assertEqual(guard.update(dict(self.lift,scene_penetration_m=.002)),'failed')
        self.assertEqual(guard.reason,'scene_penetration')

    def test_geometry_group_is_path_independent_and_video_specific(self):
        a=dict(x=np.array([1.,2.]),y=np.eye(3))
        b=dict(y=a['y'].copy(),x=a['x'].copy())
        self.assertEqual(geometry_group(a,'first'),geometry_group(b,'first'))
        self.assertNotEqual(geometry_group(a,'first'),geometry_group(b,'second'))
        b['x'][0]+=1e-3
        self.assertNotEqual(geometry_group(a,'first'),geometry_group(b,'first'))


if __name__=='__main__': unittest.main()
