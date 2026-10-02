#!/usr/bin/env python3
"""Convert an allowlisted Chinese instruction into a validated skill plan."""
import argparse
import datetime
import json
from pathlib import Path
from hierarchy_common import ROOT, SkillRegistry, run_plan
from fromrealhand.hierarchy import RulePlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('instruction')
    parser.add_argument('--scene', choices=['first', 'second'], default='first')
    parser.add_argument('--registry', type=Path, default=ROOT/'configs/skill_registry.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--render', action='store_true', help='Native MuJoCo window; close it to exit')
    modes.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    try:
        registry = SkillRegistry.load(args.registry)
        plan = RulePlanner(registry).plan(args.instruction, args.scene)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
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
