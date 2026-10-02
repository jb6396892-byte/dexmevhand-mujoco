"""Bounded sequential execution; no simulator reset/state-write API is used here."""
import time
import numpy as np
from ..skills import safety_failure
from ..skill_pipeline import GuardedSkill
from .planner import RulePlanner


class TransientActionError(RuntimeError):
    """A provider failure before stepping; retry must not replay physical actions."""


class ReplanRequested(RuntimeError):
    """A provider asks to revalidate the remaining plan at a skill boundary."""


class BackendStopped(RuntimeError):
    pass


class SkillExecutor:
    def __init__(self, registry, clock=time.monotonic):
        self.registry, self.clock = registry, clock

    def run(self, plan, backend):
        plan = self.registry.validate_plan(plan)
        limits = self.registry.config['limits']
        completed, events = [], []
        started = self.clock()
        retries = replans = 0
        active, row = None, None

        def record(kind, **details):
            events.append(dict(kind=kind, skill=active, global_step=backend.cursor,
                               wall_s=self.clock()-started, **details))

        def finish(status, reason):
            backend.halt(reason)
            record('plan_finished', status=status, reason=reason)
            return dict(schema_version=1, status=status, reason=reason, plan=plan,
                backend=self.registry.config['backend'], completed=completed, events=events,
                steps=backend.cursor, provider_retries=retries, replans=replans, final_metrics=row,
                wall_s=self.clock()-started, stop_semantics='No further simulation steps; not a hardware emergency stop')

        try:
            if plan['goal'] == 'stop':
                return finish('stopped', 'user_stop')
            row = backend.metrics()
            for active in plan['skills']:
                if self.clock()-started >= limits['wall_timeout_s']:
                    return finish('stopped', 'wall_timeout')
                if backend.cursor != backend.bounds[active][0]:
                    return finish('stopped', 'reference_clock_mismatch')
                guard = GuardedSkill(active, backend.contract_config, row)
                record('skill_started', prerequisite_passed=guard.status != 'failed')
                if guard.status == 'failed':
                    return finish('stopped', guard.reason)
                skill_steps = 0
                while backend.cursor < backend.bounds[active][1]:
                    if self.clock()-started >= limits['wall_timeout_s']:
                        return finish('stopped', 'wall_timeout')
                    if backend.cursor >= limits['max_total_steps']:
                        return finish('stopped', 'total_step_budget')
                    if skill_steps >= self.registry.config['skills'][active]['max_steps']:
                        return finish('stopped', 'skill_timeout')
                    failure = safety_failure(row, backend.contract_config)
                    if failure:
                        return finish('stopped', failure)
                    before = backend.cursor
                    try:
                        action = backend.action(active)
                    except (TransientActionError, ReplanRequested) as error:
                        if backend.cursor != before:
                            return finish('stopped', 'provider_advanced_on_failure')
                        row = backend.metrics()
                        record('provider_failure', reason=type(error).__name__)
                        failure = safety_failure(row, backend.contract_config)
                        if failure:
                            return finish('stopped', failure)
                        if skill_steps == 0 and GuardedSkill(active, backend.contract_config, row).status == 'failed':
                            return finish('stopped', 'entry_condition')
                        if isinstance(error, TransientActionError):
                            if retries >= limits['max_provider_retries']:
                                return finish('stopped', 'retry_budget_exhausted')
                            retries += 1
                            record('provider_retry', attempt=retries, cursor_preserved=True)
                        else:
                            if replans >= limits['max_replans']:
                                return finish('stopped', 'replan_budget_exhausted')
                            try:
                                remaining = RulePlanner(self.registry).replan(plan, completed, row,
                                    backend.contract_config, backend.cursor, backend.bounds)
                            except ValueError as problem:
                                record('replan_rejected', detail=str(problem))
                                return finish('stopped', 'no_verified_recovery')
                            replans += 1
                            record('replanned', remaining=remaining, cursor_preserved=True)
                        continue
                    action = np.asarray(action, dtype=float)
                    if action.shape != (30,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.+1e-8:
                        return finish('stopped', 'invalid_action')
                    row = backend.step(action)
                    skill_steps += 1
                    if backend.cursor != before+1:
                        return finish('stopped', 'backend_clock_mismatch')
                    guard.update(row)
                    if guard.status in ('failed', 'timeout'):
                        return finish('stopped', guard.reason)
                if guard.status != 'success':
                    return finish('stopped', 'reference_exhausted_before_event')
                completed.append(active)
                record('skill_succeeded', steps=skill_steps, event=guard.reason,
                       bottom_m=row['bottom_m'], target_distance_m=row['target_distance_m'])
            return finish('success', 'plan_completed')
        except (KeyboardInterrupt, BackendStopped):
            return finish('stopped', 'cancelled')
        except Exception as error:
            record('backend_exception', error_type=type(error).__name__, detail=str(error))
            return finish('stopped', 'backend_exception')
