import json
from pathlib import Path
import unittest
import numpy as np
from fromrealhand.tabletop.random_scene import sample, validate_goal, validate_cup, DISTRACTORS

ROOT=Path(__file__).resolve().parents[1]


class RandomSceneTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/tabletop-random-v5.json').read_text())

    def test_reproducible(self):
        self.assertEqual(sample(10,self.cfg),sample(10,self.cfg))
        self.assertNotEqual(sample(10,self.cfg),sample(11,self.cfg))

    def test_counts_bounds_and_separation(self):
        counts=set(); variants=set()
        for seed in range(200):
            row=sample(seed,self.cfg); counts.add(row['distractor_count']); variants.add(row['table_variant'])
            self.assertEqual(row['objects'][0]['name'],'mug')
            self.assertEqual(len(row['objects']),row['distractor_count']+1)
            self.assertEqual(len(set(o['name'] for o in row['objects'])),len(row['objects']))
            cup=np.array(row['objects'][0]['xy'])
            self.assertTrue(np.all(cup>=self.cfg['cup_xy_min_m']) and np.all(cup<=self.cfg['cup_xy_max_m']))
            validate_goal(row['goal_world_m'],self.cfg)
            for i,obj in enumerate(row['objects'][1:],1):
                self.assertIn(obj['name'],DISTRACTORS)
                for other in row['objects'][:i]:
                    self.assertGreaterEqual(np.linalg.norm(np.array(obj['xy'])-other['xy']),self.cfg['distractor_min_separation_m'])
        self.assertEqual(counts,set(range(5))); self.assertEqual(variants,{0,1,2})

    def test_target_override_preserves_seed_layout(self):
        original=sample(15,self.cfg)
        changed=sample(15,self.cfg,goal=[.02,-.04,.17])
        self.assertEqual(original['objects'],changed['objects'])
        self.assertEqual(changed['goal_world_m'],[.02,-.04,.17])

    def test_count_override(self):
        for count in range(5):
            self.assertEqual(sample(10,self.cfg,count=count)['distractor_count'],count)

    def test_initial_cup_override_and_goal_are_independent(self):
        row=sample(10,self.cfg,cup_xy=[.02,-.02])
        self.assertEqual(row['objects'][0]['xy'],[.02,-.02])
        self.assertEqual(row['goal_world_m'],sample(10,self.cfg)['goal_world_m'])
        both=sample(10,self.cfg,cup_xy=[.02,-.02],goal=[-.03,-.05,.18])
        self.assertEqual(both['objects'][0],row['objects'][0])

    def test_initial_cup_invalid_input_is_rejected(self):
        for cup in ([0], [0,0,0], [float('nan'),0], [1,0]):
            with self.assertRaises(ValueError): validate_cup(cup,self.cfg)

    def test_wide_protocol_is_larger_and_uses_fresh_seeds(self):
        wide=json.loads((ROOT/'configs/tabletop-random-v6.json').read_text())
        for low,high in [('cup_xy_min_m','cup_xy_max_m'),('goal_min_m','goal_max_m')]:
            self.assertTrue(np.all(np.array(wide[low])<=self.cfg[low]))
            self.assertTrue(np.all(np.array(wide[high])>=self.cfg[high]))
            self.assertGreater(np.prod(np.array(wide[high])-wide[low]),np.prod(np.array(self.cfg[high])-self.cfg[low]))
        self.assertFalse(set(wide['heldout_seeds']) & set(self.cfg['heldout_seeds']+wide['development_seeds']))

    def test_reject_invalid_input(self):
        for goal in ([0,0], [float('nan'),0,.17], [1,0,.17], [0,0,-1]):
            with self.assertRaises(ValueError): validate_goal(goal,self.cfg)
        for count in (-1,5,1.5):
            with self.assertRaises(ValueError): sample(10,self.cfg,count=count)
        for seed in (-1,2**32,1.5):
            with self.assertRaises(ValueError): sample(seed,self.cfg)

    def test_splits_disjoint(self):
        self.assertFalse(set(self.cfg['development_seeds']) & set(self.cfg['heldout_seeds']))
        self.assertEqual(len(set(self.cfg['heldout_seeds'])),40)


if __name__=='__main__': unittest.main()
