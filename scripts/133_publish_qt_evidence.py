#!/usr/bin/env python3
"""Verify the Qt delivery and publish small evidence; never copy models or raw data."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.desktop.runtime import LEGACY_PYTHON, physics_environment


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, env=None, cwd=ROOT, success=True, timeout=180):
    print('CHECK', ' '.join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), cwd=str(cwd), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=timeout)
    record = dict(command=list(map(str, command)), cwd=str(cwd), returncode=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
    if success and result.returncode:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/media/smgbro/shared/lora'))
    parser.add_argument('--gui-run', type=Path, required=True)
    parser.add_argument('--native-run', type=Path, required=True)
    args = parser.parse_args()
    gui = args.gui_run.resolve()
    summary = read(gui/'summary.json')
    if len(summary['cases']) != 9 or not summary['passed']:
        raise ValueError('Nine actual GUI acceptance cases must pass')
    for case in summary['cases']:
        if not case['passed'] or case['sidebar_overlaps'] or not case['status_current_run']:
            raise ValueError('GUI layout/state acceptance missing')
        raw = read(Path(case['run'])/'desktop-result.json')
        if raw['report'] != case['report'] or raw['frames_received'] != case['frames_received']:
            raise ValueError('GUI summary does not match raw result')
    native = read(args.native_run/'physics/report.json')
    if native['status'] != 'success' or native['physics']['max_state_replay_error'] != 0:
        raise ValueError('Native window physical acceptance failed')

    destination = ROOT/'docs/presentation/qt_desktop/evidence'
    destination.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ('PYTHONHOME', 'MUJOCO_PY_FORCE_CPU', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH'):
        env.pop(name, None)
    env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),
               LD_LIBRARY_PATH='/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu',
               LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONNOUSERSITE='1')
    tests = run([LEGACY_PYTHON, '-m', 'unittest', 'discover', '-s', 'tests'], env)
    count = re.search(r'Ran (\d+) tests', tests['stderr'])
    if not count or int(count[1]) < 187:
        raise ValueError('Incomplete project regression suite')
    tests['tests_run'] = int(count[1])
    frozen = run([LEGACY_PYTHON, 'scripts/122_verify_guard_revision.py', 'verify', '--root', args.root], env)
    scripts = sorted(p for p in (ROOT/'scripts').iterdir()
                     if p.name[:3].isdigit() and 124 <= int(p.name[:3]) <= 133)
    sources = sorted((ROOT/'src/fromrealhand/desktop').glob('*.py'))
    sources += sorted((ROOT/'tests').glob('test_desktop_*.py')) + scripts
    sources += [ROOT/'scripts/116_run_stage6_model.sh', ROOT/'configs/requirements-stage6-qt.txt']
    for path in sources:
        if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    shell = [run(['bash', '-n', path]) for path in sources if path.suffix == '.sh']
    whitespace = run(['git', 'diff', '--check'])

    with tempfile.TemporaryDirectory(prefix='qt-gl-reproduction-') as scratch:
        failure = run([LEGACY_PYTHON, ROOT/'scripts/124_probe_mujoco_render.py', '--mode', 'window'],
                      env, cwd=Path(scratch), success=False, timeout=30)
    failure['expected_failure_reproduced'] = (failure['returncode'] != 0
        and 'Missing GL version' in failure['stdout']+failure['stderr'])
    if not failure['expected_failure_reproduced']:
        raise ValueError('Old CPU/OSMesa window failure was not reproduced; investigate before publishing')
    failure['environment_scope'] = 'Legacy conda CPU/OSMesa import; GPU package not on PYTHONPATH'
    write(destination/'gl-failure.json', failure)
    probes = []
    for mode in ('window', 'stream'):
        output = destination/('probe-'+mode+'.json')
        probes.append(run([LEGACY_PYTHON, ROOT/'scripts/124_probe_mujoco_render.py',
                           '--mode', mode, '--output', output], physics_environment(), timeout=45))
        if read(output)['pixel_std'] < 5 or 'NVIDIA' not in read(output)['renderer']:
            raise ValueError('GPU probe is blank or not using the expected renderer')

    for name in ('second-transport.png', 'reject-handoff.png'):
        shutil.copyfile(str(gui/name), str(destination/name))
    pixels = run([LEGACY_PYTHON, '-c',
        'import cv2,json; from pathlib import Path; '
        'p=Path("docs/presentation/qt_desktop/evidence"); '
        'r={x.name:dict(shape=list(cv2.imread(str(x)).shape),std=float(cv2.imread(str(x)).std())) for x in p.glob("*.png")}; '
        'assert len(r)==2 and all(x["std"]>5 for x in r.values()); print(json.dumps(r))'], env)
    shutil.copyfile(str(gui/'summary.json'), str(destination/'gui-summary.json'))
    shutil.copyfile(str(args.native_run/'physics/report.json'), str(destination/'native-window-report.json'))
    quality = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tests=tests, frozen_guard=frozen, shell_syntax=shell, whitespace=whitespace,
        gpu_probes=probes, screenshot_pixels=pixels, passed=True,
        scope='Desktop integration; no new training or generalization evaluation')
    write(destination/'quality.json', quality)

    docs = [ROOT/'README.md', ROOT/'docs/index.html', ROOT/'docs/QT_DESKTOP.md',
            ROOT/'docs/run_logs/2026-10-02-qt-desktop.md',
            ROOT/'docs/presentation/qt_desktop/README.md']
    paths = sources + docs + sorted(p for p in destination.iterdir() if p.name != 'manifest.json')
    manifest = dict(schema_version=1, created_at=quality['created_at'],
        base_commit=run(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
        source_gui_run=str(gui), source_native_run=str(args.native_run.resolve()),
        language_acceptance_sha256=digest(args.root/'language/study_v4_guard2/acceptance/summary.json'),
        sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
        model='Qwen2.5-0.5B-Instruct + frozen LoRA v4 / guard2',
        low_level='verified_reference; not DAPG', unchanged_physics=True)
    write(destination/'manifest.json', manifest)
    archive = args.root/'language/desktop-delivery-v1'
    archive.mkdir(exist_ok=True)
    for source in paths+[destination/'manifest.json']:
        target = archive/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(target))
        if digest(target) != digest(source): raise ValueError('Archive copy mismatch')
    print(json.dumps(dict(passed=True, gui_cases=len(summary['cases']), tests=tests['tests_run'],
                          evidence=str(destination), archive=str(archive)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
