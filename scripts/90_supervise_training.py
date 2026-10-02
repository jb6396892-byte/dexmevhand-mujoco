#!/usr/bin/env python3
"""Supervise the immutable 200-iteration run; never replace an existing policy."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.training_monitor import atomic_json, health


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/v14-long-supervision.json')
    args = parser.parse_args()
    config = args.config.resolve()
    cfg = json.loads(config.read_text())
    plan = ROOT/cfg['study']
    version = json.loads(plan.read_text())['version']
    run = ROOT/'data/processed'/('dual_video_'+version)
    output = run/('supervision_%d' % cfg['iterations'])
    training = run/'long_training'
    if args.command == 'status':
        print((output/'status.json').read_text())
        return
    freeze = json.loads((run/'freeze.json').read_text())
    ready = json.loads((run/'readiness.json').read_text())
    learning = json.loads((ROOT/'configs/v14-dapg.json').read_text())
    protected = dict(freeze['code_sha256'])
    protected.update({str(plan): freeze['protocol_sha256'], freeze['policy']: freeze['policy_sha256'],
                      str(config): digest(config)})
    if cfg['iterations'] != 200 or not ready['ready_for_long_training'] or not all(ready['checks'].values()):
        raise RuntimeError('Registered 200-iteration readiness is required')
    if ready['policy_sha256'] != freeze['policy_sha256'] or ready['training_config_sha256'] != digest(ROOT/'configs/v14-dapg.json'):
        raise RuntimeError('Readiness hashes changed')
    if training.exists():
        raise FileExistsError('Refusing to overwrite or duplicate existing training: '+str(training))
    for name, expected in protected.items():
        if digest(ROOT/name) != expected:
            raise RuntimeError('Frozen input changed: '+name)
    output.mkdir(exist_ok=False)
    command = [sys.executable, '-u', str(ROOT/'scripts/87_train_v14_dapg.py'), '--config', str(plan),
               '--long', '--iterations', str(cfg['iterations'])]
    atomic_json(output/'launch.json', dict(started_at=utc(), command=command, protected_sha256=protected,
        policy_sha256=freeze['policy_sha256'], commit=subprocess.check_output(
            ['git','rev-parse','HEAD'], cwd=str(ROOT), universal_newlines=True).strip(),
        runtime=dict(actor='CPU', simulation='CPU', value_baseline='CUDA'),
        automatic_policy_replacement=False))
    started = time.monotonic()
    rows, previous_count, last_progress = [], 0, started
    status = dict(state='starting', iterations=0, requested_iterations=cfg['iterations'], started_at=utc())
    atomic_json(output/'status.json', status)
    stopped = []
    def on_signal(number, frame):
        stopped.append('signal_'+str(number))
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    with (output/'training.log').open('x') as stream:
        child = subprocess.Popen(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        status['training_pid'] = child.pid
        try:
            while True:
                error = None
                try:
                    new = json.loads((training/'iterations.json').read_text())
                    if len(new) < len(rows):
                        error = 'iteration_log_regressed'
                    else:
                        rows = new
                except (FileNotFoundError, json.JSONDecodeError):
                    pass  # The frozen writer is not atomic; retry rather than read a partial log.
                audit = health(rows, cfg, learning['max_measured_kl'])
                now = time.monotonic()
                if len(rows) > previous_count:
                    last_progress = now
                    previous_count = len(rows)
                    print(json.dumps(audit), flush=True)
                if any(digest(ROOT/name) != sha for name, sha in protected.items()):
                    error = 'frozen_input_changed'
                if now-last_progress > cfg['stall_seconds']:
                    error = 'no_iteration_progress_timeout'
                if audit['errors']:
                    error = ','.join(audit['errors'])
                if stopped:
                    error = stopped[0]
                status.update(audit, updated_at=utc(), elapsed_seconds=now-started,
                    seconds_without_iteration=now-last_progress, state='running',
                    estimated_remaining_seconds=(now-started)/len(rows)*(cfg['iterations']-len(rows)) if rows else None,
                    last_checkpoint=max([p.name for p in training.glob('iteration_*.pickle')] or ['none']),
                    process_exit_code=child.poll(), stop_reason=error)
                if error:
                    status['state'] = 'stopped_by_monitor'
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                if child.poll() is not None:
                    status['process_exit_code'] = child.returncode
                    if not error:
                        status['state'] = 'failed' if child.returncode else 'completed'
                        if child.returncode == 0:
                            rows = json.loads((training/'iterations.json').read_text())
                            final_audit = health(rows, cfg, learning['max_measured_kl'])
                            status.update(final_audit)
                            summary = json.loads((training/'summary.json').read_text())
                            status['completed_iterations'] = summary['completed_iterations']
                            status['nominal_post_task_pass'] = all(r['task_pass'] for r in summary['post_nominal'])
                            status['nominal_post_strict_pass'] = all(r['strict_pass'] for r in summary['post_nominal'])
                            status['independent_evaluation_done'] = False
                            if summary['completed_iterations'] != cfg['iterations'] or final_audit['errors']:
                                status['state'] = 'failed'
                                status['stop_reason'] = 'incomplete_or_unhealthy_final_result'
                    atomic_json(output/'status.json', status)
                    print(json.dumps(status), flush=True)
                    break
                atomic_json(output/'status.json', status)
                time.sleep(cfg['poll_seconds'])
        except BaseException as exc:
            status.update(state='supervisor_error', updated_at=utc(), stop_reason=repr(exc))
            atomic_json(output/'status.json', status)
            raise
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    if status['state'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
