#!/usr/bin/env python3
"""Native window acceptance without the user-facing final hold loop."""
import argparse
import json
from pathlib import Path
from hierarchy_common import ROOT,SkillRegistry,run_plan
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args = parser.parse_args()
result = run_plan(SkillRegistry.load(ROOT/'configs/skill_registry.yaml'),
    json.loads(args.plan.read_text()),args.output,render=True,hold_window=False)
print(json.dumps(result,ensure_ascii=False))
raise SystemExit(0 if result['status']=='success' or result['reason']=='user_stop' else 1)
