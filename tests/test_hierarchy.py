import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from fromrealhand.hierarchy import (SkillRegistry, RulePlanner, SkillExecutor,
    TransientActionError, ReplanRequested, BackendStopped)

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self, fault=None):
        self.contract_config = json.loads((ROOT/'configs/stage4-skills.json').read_text())
        self.contract_config.update(confirmation_steps=dict(reach=2, grasp=2, lift=2),
                                    support_loss_stop_steps=2, transport_hold_steps=2)
        self.bounds = dict(reach=(0, 2), grasp=(2, 4), lift=(4, 6), transport=(6, 8))
        self.empty = dict(bottom_m=0., target_distance_m=.2, scene_penetration_m=0.,
            joint_violation_rad=0., finite=True, th_force_n=0., ff_force_n=0.,
            mf_force_n=0., rf_force_n=0., lf_force_n=0.)
        contact = dict(self.empty, ff_force_n=1.)
        grasp = dict(contact, th_force_n=1., mf_force_n=1.)
        lift = dict(grasp, bottom_m=.06)
        goal = dict(lift, target_distance_m=.01)
        self.rows = [contact]*2+[grasp]*2+[lift]*2+[goal]*2
        self.cursor, self.stopped, self.fired = 0, False, False
        self.calls, self.fault = [], fault

    def metrics(self):
        return (self.empty if self.cursor == 0 else self.rows[self.cursor-1]).copy()

    def action(self, skill):
        if self.cursor == 2 and (not self.fired or self.fault in ('always', 'replan_always')):
            self.fired = True
            if self.fault == 'advanced_failure':
                self.cursor += 1
                raise TransientActionError()
            if self.fault in ('once', 'always'):
                raise TransientActionError()
            if self.fault in ('replan', 'replan_always'):
                raise ReplanRequested()
            if self.fault == 'cancel':
                raise BackendStopped()
            if self.fault == 'nan':
                return np.full(30, np.nan)
        return np.full(30, self.cursor*.01)

    def step(self, action):
        if self.stopped:
            raise AssertionError('A halted backend was stepped')
        self.calls.append(self.cursor)
        self.cursor += 1
        return self.metrics()

    def halt(self, reason):
        self.stopped = True


class HierarchyTests(unittest.TestCase):
    def setUp(self):
        self.registry = SkillRegistry.load(ROOT/'configs/skill_registry.yaml')
        self.planner = RulePlanner(self.registry)
        self.plan = self.planner.plan('\u6293\u8d77\u676f\u5b50')

    def execute(self, backend, plan=None, registry=None, **kwargs):
        return SkillExecutor(registry or self.registry, **kwargs).run(plan or self.plan, backend)

    def test_instruction_aliases_and_stop(self):
        for goal, aliases in self.registry.config['instructions'].items():
            for alias in aliases:
                self.assertEqual(self.planner.plan(' '+alias+'! ')['goal'], goal)
        self.assertEqual(self.plan['skills'], ['reach', 'grasp', 'lift'])
        self.assertEqual(self.planner.plan('\u505c\u6b62')['skills'], [])

    def test_negation_new_object_pouring_and_arbitrary_goal_rejected(self):
        for text in ('\u4e0d\u8981\u6293\u8d77\u676f\u5b50', '\u5012\u6c34',
                     '\u6293\u8d77\u74f6\u5b50', 'lift mug to x=100', '', 'x'*121):
            with self.assertRaises(ValueError):
                self.planner.plan(text)

    def test_invalid_order_object_and_extra_fields_rejected(self):
        for patch in (dict(skills=['lift']), dict(skills=['reach', 'lift', 'grasp']),
                      dict(object='bottle'), dict(target=[1, 2, 3]), dict(scene='unknown')):
            with self.assertRaises(ValueError):
                self.registry.validate_plan(dict(self.plan, **patch))

    def test_registry_rejects_unsafe_yaml_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'registry.yaml'
            for text in ('backend: a\nbackend: b\n', '!!python/object/apply:os.system ["true"]'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    SkillRegistry.load(path)

    def test_registry_cannot_silently_change_contract_or_budget(self):
        for patch in ('semantics', 'nan', 'backend', 'path'):
            config = copy.deepcopy(self.registry.config)
            if patch == 'semantics': config['skills']['lift']['precondition'] = 'anything'
            if patch == 'nan': config['limits']['wall_timeout_s'] = float('nan')
            if patch == 'backend': config['backend'] = 'unvalidated_checkpoint'
            if patch == 'path': config['dataset'] = '../outside'
            with self.assertRaises(ValueError): SkillRegistry(config)

    def test_continuous_prefix_and_complete_transport(self):
        for goal, steps in (('lift', 6), ('transport', 8)):
            backend = FakeBackend()
            plan = dict(self.plan, goal=goal, skills=list(backend.bounds)[:3 if goal=='lift' else 4])
            report = self.execute(backend, plan)
            self.assertEqual(report['status'], 'success')
            self.assertEqual(backend.calls, list(range(steps)))
            self.assertTrue(backend.stopped)

    def test_transient_retry_preserves_cursor_and_no_repeated_actions(self):
        backend = FakeBackend('once')
        report = self.execute(backend)
        self.assertEqual(report['status'], 'success')
        self.assertEqual(report['provider_retries'], 1)
        self.assertEqual(backend.calls, list(range(6)))

    def test_repeated_failure_stops_with_bounded_retries(self):
        backend = FakeBackend('always')
        report = self.execute(backend)
        self.assertEqual(report['reason'], 'retry_budget_exhausted')
        self.assertEqual(report['provider_retries'], 2)
        self.assertEqual(backend.calls, [0, 1])
        self.assertEqual(report['completed'], ['reach'])

    def test_replan_checks_live_entry_without_restarting_reach(self):
        backend = FakeBackend('replan')
        report = self.execute(backend)
        self.assertEqual(report['status'], 'success')
        self.assertEqual(report['replans'], 1)
        self.assertEqual(backend.calls, list(range(6)))
        replanned = next(e for e in report['events'] if e['kind'] == 'replanned')
        self.assertEqual(replanned['remaining'], ['grasp', 'lift'])

    def test_replan_is_bounded_and_rejects_clock_jump(self):
        report = self.execute(FakeBackend('replan_always'))
        self.assertEqual(report['reason'], 'replan_budget_exhausted')
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            self.planner.replan(self.plan, ['reach'], backend.rows[1], backend.contract_config, 3, backend.bounds)

    def test_nonfinite_action_and_cancellation_stop_before_action(self):
        for fault, reason in (('nan', 'invalid_action'), ('cancel', 'cancelled')):
            backend = FakeBackend(fault)
            report = self.execute(backend)
            self.assertEqual(report['reason'], reason)
            self.assertEqual(backend.calls, [0, 1])

    def test_state_advance_on_provider_failure_is_not_retried(self):
        report = self.execute(FakeBackend('advanced_failure'))
        self.assertEqual(report['reason'], 'provider_advanced_on_failure')
        self.assertEqual(report['provider_retries'], 0)

    def test_safety_failure_stops_next_skill_even_after_success(self):
        backend = FakeBackend()
        backend.rows[2] = dict(backend.rows[2], scene_penetration_m=.002)
        report = self.execute(backend)
        self.assertEqual(report['reason'], 'scene_penetration')
        self.assertEqual(backend.calls, [0, 1, 2])
        self.assertEqual(report['provider_retries'], 0)

    def test_wrong_entry_and_nonfinite_state_stop(self):
        backend = FakeBackend()
        backend.rows[2:4] = [backend.rows[1]]*2
        report = self.execute(backend)
        self.assertEqual(report['reason'], 'reference_exhausted_before_event')
        self.assertEqual(report['completed'], ['reach'])
        backend = FakeBackend()
        backend.empty['bottom_m'] = float('nan')
        self.assertEqual(self.execute(backend)['reason'], 'nonfinite_state')
        self.assertEqual(backend.calls, [])

    def test_step_and_wall_time_budgets_do_not_restart_on_retry(self):
        config = copy.deepcopy(self.registry.config)
        config['skills']['grasp']['max_steps'] = 1
        backend = FakeBackend('once')
        report = self.execute(backend, registry=SkillRegistry(config))
        self.assertEqual(report['reason'], 'skill_timeout')
        self.assertEqual(backend.calls, [0, 1, 2])
        backend = FakeBackend()
        times = iter(range(1000))
        config['limits']['wall_timeout_s'] = .5
        report = self.execute(backend, registry=SkillRegistry(config), clock=lambda: next(times))
        self.assertEqual(report['reason'], 'wall_timeout')
        self.assertEqual(backend.calls, [])

    def test_stop_instruction_has_no_action(self):
        backend = FakeBackend()
        report = self.execute(backend, self.planner.plan('\u505c\u6b62'))
        self.assertEqual(report['reason'], 'user_stop')
        self.assertEqual(backend.calls, [])


if __name__ == '__main__':
    unittest.main()
