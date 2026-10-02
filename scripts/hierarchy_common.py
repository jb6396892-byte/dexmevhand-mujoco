"""Legacy MuJoCo adapter for the hierarchy; physics remains outside the planner."""
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.hierarchy import (SkillRegistry, SkillExecutor, TransientActionError,
                                   ReplanRequested, BackendStopped)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(safe_json(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_delivery(registry):
    run = ROOT/registry.config['dataset']
    freeze_path = run/'delivery/freeze.json'
    receipt = read(run/'delivery/receipt.json')
    expected = registry.config['delivery_freeze_sha256']
    if not receipt['passed'] or digest(freeze_path) != expected or receipt['freeze_sha256'] != expected:
        raise ValueError('Stage4 qualified delivery is missing or changed')
    for path, sha in read(freeze_path)['sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen source changed: '+path)
    return run


class ReferenceBackend:
    """Qualified nominal reference only. A skill switch never resets the scene."""
    def __init__(self, registry, scene, render=False, capture=False, fault=None):
        from stage4_common import experiment, restore_once, measure
        from stage4_pipeline_common import load_pieces
        run = verify_delivery(registry)
        manifest = read(run/'build/manifest.json')
        trajectory = registry.config['scenes'][scene]
        self.entry = next(e for e in manifest['trajectories'] if e['trajectory'] == trajectory)
        self.pieces = load_pieces(run/'build', self.entry)
        self.contract_config = copy.deepcopy(manifest['config'])
        self.bounds = {s['skill']: (s['start'], s['stop']) for s in self.entry['segments']}
        self.actions = np.concatenate([p['actions'] for p in self.pieces])
        self.expected = np.concatenate([p['expected_post_states'] for p in self.pieces])
        self.cursor, self.audit_index, self.max_error = 0, 0, 0.
        self.stopped, self.fault, self.fired = False, fault, False
        self.rows, self.executed_actions = [], []
        self.render_enabled, self.context, self.last_row = render, None, None
        self.exp = experiment(self.entry['geometry'], self.pieces[0])
        self.measure = measure
        try:
            # Context construction performs forward(), so it precedes the sole restore.
            if render:
                from mujoco_py import MjViewer
                self.context = MjViewer(self.exp.env.sim)
            elif capture:
                from mujoco_py import MjRenderContextOffscreen
                self.context = MjRenderContextOffscreen(self.exp.env.sim)
            if self.context is not None:
                self.context.cam.type = 0
                with np.load(self.entry['geometry']) as geometry:
                    self.context.cam.lookat[:] = geometry['object_poses'][[0, -1], :3, 3].mean(axis=0)
                self.context.cam.distance, self.context.cam.azimuth, self.context.cam.elevation = .55, 135., -30.
            restore_once(self.exp.env, self.pieces[0]['initial_snapshot'])
            self.initialization_count = 1
            self.started = time.monotonic()
            self.metrics()
        except BaseException:
            self.exp.env.close()
            raise

    def metrics(self):
        self.last_row = self.measure(self.exp, self.audit_index)
        self.audit_index = len(self.exp.audit)
        return self.last_row.copy()

    def action(self, skill):
        if self.stopped:
            raise BackendStopped('Backend was halted')
        lo, hi = self.bounds[skill]
        if not lo <= self.cursor < hi:
            raise ValueError('Action clock outside active skill')
        if self.cursor == self.bounds['grasp'][0] and self.fault:
            if self.fault in ('provider_always', 'replan_always') or not self.fired:
                self.fired = True
                if self.fault in ('provider_once', 'provider_always'):
                    raise TransientActionError('Validation-only action provider interruption')
                if self.fault in ('replan_once', 'replan_always'):
                    raise ReplanRequested('Validation-only plan revalidation request')
                if self.fault == 'invalid_action':
                    return np.full(30, np.nan)
        return self.actions[self.cursor].copy()

    def step(self, action):
        if self.stopped:
            raise BackendStopped('No steps are allowed after halt')
        if self.render_enabled:
            import glfw
            if glfw.window_should_close(self.context.window):
                raise BackendStopped('Window closed')
        self.exp.env.step(action)
        self.executed_actions.append(action.copy())
        self.cursor += 1
        row = self.metrics()
        actual = np.r_[self.exp.env.sim.data.qpos, self.exp.env.sim.data.qvel]
        error = float(np.max(np.abs(actual-self.expected[self.cursor-1])))
        self.max_error = max(self.max_error, error)
        self.rows.append(dict(row, global_step=self.cursor, replay_error=error))
        if not np.isfinite(error) or error > self.contract_config['replay_state_tolerance']:
            raise RuntimeError('Reference state mismatch; further actions halted')
        if self.render_enabled:
            self.context.render()
            remaining = self.cursor*self.entry['dt']-(time.monotonic()-self.started)
            if remaining > 0:
                time.sleep(remaining)
        return row

    def halt(self, reason):
        self.stopped = True

    def summary(self):
        return dict(trajectory=self.entry['trajectory'], max_state_replay_error=self.max_error,
            max_penetration_m=max([r['scene_penetration_m'] for r in self.rows]+[self.last_row['scene_penetration_m']]),
            initialization_count=self.initialization_count, state_writes_during_execution=0,
            actual_env_steps=len(self.executed_actions), halted=self.stopped, final_metrics=self.last_row,
            scope='Qualified fixed expert reference; not the DAPG policy or a learned skill controller')

    def screenshot(self, path, label):
        import cv2
        if self.context is None or self.render_enabled:
            raise ValueError('Use an offscreen capture context for evidence')
        before = np.r_[self.exp.env.sim.data.qpos, self.exp.env.sim.data.qvel].copy()
        self.context.render(960, 720, camera_id=-1)
        frame = cv2.cvtColor(self.context.read_pixels(960, 720, depth=False)[::-1], cv2.COLOR_RGB2BGR)
        if not np.array_equal(before, np.r_[self.exp.env.sim.data.qpos, self.exp.env.sim.data.qvel]):
            raise RuntimeError('Rendering changed physical state')
        if frame.std() < 5.:
            raise RuntimeError('Blank evidence frame')
        cv2.putText(frame, label, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
        cv2.putText(frame, 'EXPERT | step %d | green = goal marker' % self.cursor,
                    (16, 58), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
        if not cv2.imwrite(str(path), frame):
            raise IOError('Could not save screenshot')
        return dict(file=Path(path).name, sha256=digest(path), state_unchanged=True,
                    pixel_std=float(frame.std()), width=960, height=720)

    def hold_window(self):
        if self.render_enabled:
            import glfw
            while not glfw.window_should_close(self.context.window):
                self.context.render()
                time.sleep(.03)

    def close(self):
        self.exp.env.close()


def run_plan(registry, plan, output, render=False, capture=False, fault=None, hold_window=False):
    plan = registry.validate_plan(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/'plan.json', plan)
    if plan['goal'] == 'stop':
        report = dict(status='stopped', reason='user_stop', steps=0, simulation_created=False, plan=plan)
        write(output/'report.json', report)
        return report
    try:
        backend = ReferenceBackend(registry, plan['scene'], render=render, capture=capture, fault=fault)
    except Exception as error:
        report = dict(status='stopped', reason='initialization_failed', steps=0, plan=plan,
                      error_type=type(error).__name__, detail=str(error))
        write(output/'report.json', report)
        return report
    try:
        report = SkillExecutor(registry).run(plan, backend)
        report.update(physics=backend.summary(), fault_injection=fault,
                      created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if capture:
            report['screenshot'] = backend.screenshot(output/'final.jpg',
                '%s | %s | %s' % (plan['scene'], plan['goal'], report['status']))
        np.savez_compressed(output/'trace.npz', actions=np.asarray(backend.executed_actions),
                            bottom_m=np.asarray([r['bottom_m'] for r in backend.rows]),
                            penetration_m=np.asarray([r['scene_penetration_m'] for r in backend.rows]))
        write(output/'report.json', report)
        if hold_window:
            backend.hold_window()
        return report
    finally:
        backend.close()
