#!/usr/bin/env python3
"""Validate and execute a fixed skill plan; no reset at skill boundaries."""
import argparse
import datetime
import json
from pathlib import Path
from hierarchy_common import ROOT, SkillRegistry, run_plan


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate plan field: '+key)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'configs/skill_plan_pickup.json')
    parser.add_argument('--registry', type=Path, default=ROOT/'configs/skill_registry.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--render', action='store_true', help='Native MuJoCo window; close it to exit')
    modes.add_argument('--capture', action='store_true', help='Save one final offscreen screenshot')
    args = parser.parse_args()
    try:
        registry = SkillRegistry.load(args.registry)
        plan = registry.validate_plan(json.loads(args.plan.read_text(), object_pairs_hook=unique_pairs))
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = args.output or ROOT/'data/processed/hierarchy_runs'/stamp
        report = run_plan(registry, plan, output, args.render, args.capture, hold_window=args.render)
        print(json.dumps(dict(status=report['status'], reason=report['reason'], output=str(output)), ensure_ascii=False))
        if report['status'] != 'success' and report['reason'] != 'user_stop':
            raise SystemExit(1)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, 'Rejected before further execution: %s\n' % error)


if __name__ == '__main__':
    main()
