#!/usr/bin/env python3
"""Archive honest framework-only evidence, regression results and completed DAPG status."""
import argparse
import contextlib
import datetime
import io
from pathlib import Path
import shutil
import subprocess
import unittest

from hierarchy_common import ROOT, SkillRegistry, verify_delivery
from fromrealhand.language_planner.contracts import digest, read, write
from fromrealhand.language_planner.sft import verify_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', type=Path, default=ROOT/'docs/presentation/stage6/evidence/v1')
    args = parser.parse_args()
    local = ROOT/'data/processed/stage6_language_v1'
    verify_protocol(ROOT, local/'dataset')
    verify_delivery(SkillRegistry.load(ROOT/'configs/skill_registry.yaml'))
    framework = read(local/'framework/summary.json')
    if not framework['framework_physics_passed'] or framework['model_trained']:
        raise ValueError('Expected a framework-only passing report')
    launch = read(ROOT/'data/processed/dual_video_v14c/supervision_200/launch.json')
    for path, expected in launch['protected_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Protected DAPG input changed: '+path)
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        suite = unittest.TestLoader().discover(str(ROOT/'tests'))
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    if not result.wasSuccessful():
        print(output.getvalue())
        raise SystemExit('Tests failed; evidence not published')
    args.publish.mkdir(parents=True, exist_ok=False)
    (args.publish/'unit-tests.txt').write_text(output.getvalue(), encoding='utf-8')
    write(args.publish/'unit-tests.json', dict(run=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), skipped=len(result.skipped), passed=result.wasSuccessful()))
    for name in ('protocol.json', 'feasibility.json', 'execution_audit.json'):
        shutil.copy2(str(local/'dataset'/name), str(args.publish/('dataset-'+name)))
    for name in ('summary.json', 'physical.json', 'negative_guards.json'):
        shutil.copy2(str(local/'framework'/name), str(args.publish/('framework-'+name)))
    shutil.copy2(str(local/'framework/second-transport/final.jpg'), str(args.publish/'second-transport-fixture.jpg'))
    status = read(ROOT/'data/processed/dual_video_v14c/supervision_200/status.json')
    summary = read(ROOT/'data/processed/dual_video_v14c/long_training/summary.json')
    iterations = read(ROOT/'data/processed/dual_video_v14c/long_training/iterations.json')
    failures = [dict(iteration=row['iteration'], video=r['video'], case=r['case'],
        max_scene_penetration_m=r['report']['max_hand_scene_penetration_m'],
        goal_distance_m=r['report']['final_distance_m'])
        for row in iterations for r in row['reports'] if not r['task_pass']]
    write(args.publish/'training-status.json', status)
    write(args.publish/'training-summary.json', summary)
    write(args.publish/'training-failures.json', failures)
    storage = subprocess.check_output(['findmnt', '-T', '/media/smgbro/shared', '-o', 'TARGET,FSTYPE,OPTIONS'],
                                     universal_newlines=True)
    write(args.publish/'model-evaluation-status.json', dict(status='not_run', lora_adapter=None,
        language_accuracy=None, raw_json_validity=None, learned_planner_physics=None,
        reason='User explicitly selected framework only; no environment install, model download, or SFT this turn',
        storage_observation=storage.strip(),
        storage_note='Host rw; initial sandbox ro observation was not a disk fault. No repair/remount performed.'))
    write(args.publish/'publication.json', dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose='Framework validation and DAPG completion, NOT trained-language acceptance',
        protected_dapg_files_verified=len(launch['protected_sha256']),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(),
            local/'dataset/protocol.json', local/'framework/summary.json',
            ROOT/'data/processed/dual_video_v14c/long_training/summary.json']},
        sha256={p.name: digest(p) for p in args.publish.iterdir() if p.is_file()}))
    print(dict(published=str(args.publish), tests=result.testsRun, dapg_iterations=status['iterations'],
               lora_trained=False, model_evaluation_done=False))


if __name__ == '__main__':
    main()
