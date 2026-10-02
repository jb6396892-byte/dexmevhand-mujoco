#!/usr/bin/env python3
"""Recheck frozen inputs, nonblank evidence and unit tests without retraining."""
import argparse
import datetime
import importlib
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.language_planner.contracts import read, write, digest
from fromrealhand.language_planner.sft import verify_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Keep previous verification receipt')
    importlib.import_module('109_evaluate_post200').verify(
        ROOT/'data/processed/dual_video_v14c/post200_independent_v1')
    verify_protocol(ROOT,ROOT/'data/processed/stage6_language_v1/dataset')
    evidence = ROOT/'docs/presentation/post200/evidence/v1'
    import cv2
    screenshots = read(evidence/'screenshots.json')
    for screenshot in screenshots:
        path = evidence/(screenshot['video']+'.jpg')
        pixels = cv2.imread(str(path))
        if digest(path) != screenshot['sha256'] or pixels is None or pixels.std() < 5:
            raise ValueError('Screenshot changed or blank')
        if screenshot['replay_error'] > 1e-8:
            raise ValueError('Physical screenshot replay mismatch')
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tests_run=result.testsRun, failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),passed=result.wasSuccessful(),
        frozen_stage3_inputs_unchanged=True,frozen_stage6_framework_unchanged=True,
        screenshot_count=len(screenshots), screenshot_replay_max_error=max(r['replay_error'] for r in screenshots),
        stage3_evaluation_sha256=digest(evidence/'summary.json'),
        test_scope='Software regressions and evidence integrity; not a claim of policy acceptance')
    write(args.output,receipt)
    args.output.with_suffix('.log').write_text(stream.getvalue(),encoding='utf-8')
    print(receipt)
    if not result.wasSuccessful():
        print(stream.getvalue())
        raise SystemExit(1)


if __name__ == '__main__':
    main()
