import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np
from fromrealhand.skills import (SKILLS, SegmentedReference, SkillContract, execute_reference,
                                propose_segments, sustained_event, validate_segments)


class SkillTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'configs/stage4-skills.json').read_text())
        self.config.update(event_persistence_steps=2, transport_hold_steps=2)
        self.empty = dict(bottom_m=0., target_distance_m=.2, scene_penetration_m=0.,
                          joint_violation_rad=0., finite=True, source_frame=0.,
                          th_force_n=0., ff_force_n=0., mf_force_n=0., rf_force_n=0., lf_force_n=0.)
        self.contact = dict(self.empty, ff_force_n=1.)
        self.grasp = dict(self.contact, th_force_n=1., mf_force_n=1.)
        self.lift = dict(self.grasp, bottom_m=.06)
        self.goal = dict(self.lift, target_distance_m=.02)
        self.rows = [self.contact]*2+[self.grasp]*2+[self.lift]*2+[self.goal]*2
        self.segments = propose_segments(self.rows, self.config, .01)

    def test_post_action_boundaries_cover_every_action_once(self):
        self.assertEqual([(s['start'],s['stop']) for s in self.segments], [(0,2),(2,4),(4,6),(6,8)])
        self.assertTrue(all(s['review_status']=='pending' for s in self.segments))

    def test_contact_flicker_is_not_a_boundary(self):
        self.assertEqual(sustained_event([self.contact,self.empty,self.contact,self.contact],
                                         'reach',self.config),4)

    def test_missing_event_is_not_fabricated(self):
        with self.assertRaises(ValueError):
            propose_segments([self.empty]*20,self.config,.01)

    def test_invalid_partition_rejected(self):
        for key, value in [('start',3),('stop',0)]:
            bad = [dict(s) for s in self.segments]
            bad[1][key] = value
            with self.assertRaises(ValueError):
                validate_segments(bad,8)

    def test_safety_precedes_success_and_stays_failed(self):
        contract = SkillContract('transport',self.config,self.lift)
        self.assertEqual(contract.update(self.goal),'running')
        self.assertEqual(contract.update(self.goal),'success')
        self.assertEqual(contract.update(dict(self.goal,scene_penetration_m=.002)),'failed')
        self.assertEqual(contract.update(self.goal),'failed')

    def test_entry_requires_thumb_opposition(self):
        contract = SkillContract('lift',self.config,dict(self.grasp,th_force_n=0.,rf_force_n=1.))
        self.assertEqual(contract.reason,'entry_condition')

    def test_timeout_and_nonfinite(self):
        self.config['timeout_steps']['reach'] = 2
        contract = SkillContract('reach',self.config,self.empty)
        contract.update(self.empty)
        self.assertEqual(contract.update(self.empty),'timeout')
        contract = SkillContract('reach',self.config,self.empty)
        self.assertEqual(contract.update(dict(self.empty,bottom_m=float('nan'))),'failed')

    def test_final_transport_must_still_be_held(self):
        contract = SkillContract('transport',self.config,self.lift)
        contract.update(self.goal); contract.update(self.goal)
        self.assertEqual(contract.update(self.empty),'running')

    def test_provider_rejects_invalid_actions(self):
        for actions in (np.zeros((8,29)),np.full((8,30),np.nan),np.full((8,30),2.)):
            with self.assertRaises(ValueError):
                SegmentedReference(actions,self.segments)

    def test_stitching_has_no_reset_or_state_write_api(self):
        calls = []
        env = SimpleNamespace(step=lambda action:calls.append(action.copy()))
        provider = SegmentedReference(np.zeros((8,30)),self.segments)
        results = execute_reference(env,provider,self.config,
            lambda:self.empty if not calls else self.rows[len(calls)-1])
        self.assertEqual(len(calls),8)
        self.assertEqual([r['skill'] for r in results],list(SKILLS))
        self.assertTrue(all(r['status']=='success' for r in results))


if __name__ == '__main__':
    unittest.main()
