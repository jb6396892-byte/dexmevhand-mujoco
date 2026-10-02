import json
from pathlib import Path
import unittest

from fromrealhand.skill_review import audit_boundaries, first_run_end, longest_run, source_frame_record
from fromrealhand.skills import propose_segments


class SkillReviewTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.config = json.loads((root/'configs/stage4-skills.json').read_text())
        self.review = json.loads((root/'configs/stage4-boundary-review.json').read_text())
        self.config.update(event_persistence_steps=2,transport_hold_steps=2)
        empty = dict(bottom_m=0.,target_distance_m=.2,scene_penetration_m=0.,
                     joint_violation_rad=0.,finite=True,source_frame=0.,
                     th_force_n=0.,ff_force_n=0.,mf_force_n=0.,rf_force_n=0.,lf_force_n=0.)
        contact = dict(empty,ff_force_n=1.)
        grasp = dict(contact,th_force_n=1.,mf_force_n=1.)
        lift = dict(grasp,bottom_m=.06)
        goal = dict(lift,target_distance_m=.02)
        self.rows = [contact]*2+[grasp]*2+[lift]*2+[goal]*4
        self.segments = propose_segments(self.rows,self.config,.01)

    def test_persistence_counts_post_action_indices(self):
        self.assertEqual(first_run_end([False,True,False,True,True],2),5)
        self.assertEqual(longest_run([False,True,False,True,True]),2)
        self.assertIsNone(first_run_end([False]*3,2))

    def test_review_cannot_approve_itself_as_human(self):
        report = audit_boundaries(self.rows,self.segments,self.config,self.review,.01)
        self.assertTrue(report['mechanical_pass'])
        self.assertEqual(report['human_review_status'],'pending')
        self.assertEqual(report['boundaries'][0]['confirmation_action_range'],[0,2])
        self.assertEqual(report['boundaries'][0]['first_confirming_post_time_s'],.01)
        self.assertEqual(report['finger_timing']['rf']['first_confirmed_state_index'],None)

    def test_changed_boundary_and_clock_fail(self):
        segments = [dict(s) for s in self.segments]
        segments[0]['stop']=3; segments[1]['start']=3
        report = audit_boundaries(self.rows,segments,self.config,self.review,.01)
        self.assertFalse(report['checks']['boundaries_reproduced'])
        rows = [dict(r) for r in self.rows]
        rows[0]['source_frame']=10
        report = audit_boundaries(rows,self.segments,self.config,self.review,.01)
        self.assertFalse(report['checks']['monotonic_source_clock'])

    def test_handoff_loss_is_explicit(self):
        rows = [dict(r) for r in self.rows]
        rows[6]['th_force_n']=0
        report = audit_boundaries(rows,self.segments,self.config,self.review,.01)
        self.assertGreater(report['handoff_warning_count'],0)

    def test_source_mapping_not_assumed_identity(self):
        mapping = [dict(source_frame=20,output_frame=0,valid=False),
                   dict(source_frame=24,output_frame=1,valid=True),
                   dict(source_frame=28,output_frame=2,valid=True)]
        self.assertEqual(source_frame_record(mapping,23.6)['output_frame'],1)
        with self.assertRaises(ValueError):
            source_frame_record([],20)


if __name__ == '__main__': unittest.main()
