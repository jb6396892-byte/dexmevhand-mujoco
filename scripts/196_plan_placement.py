#!/usr/bin/env python3
"""Create an explicit placement rule plan without modifying LoRA admission."""
import argparse
import json
from pathlib import Path
from hierarchy_common import ROOT
from fromrealhand.desktop.runtime import emit
from fromrealhand.desktop.placement_planning import placement_plan

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('instruction');p.add_argument('--scene',required=True)
p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.resolve().parent!=a.root.resolve()/'language/desktop_runs':raise ValueError('Invalid output')
a.output.mkdir(parents=True,exist_ok=False)
plan=placement_plan(a.instruction,a.scene)
record=dict(instruction=a.instruction,scene=a.scene,plan=plan,producer='placement_rules_v1',model_called=False)
(a.output/'placement-request.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
guard=dict(accepted=plan is not None,reason='placement_rules_v1' if plan else 'instruction_not_admitted')
if plan:guard['response']=dict(plan=plan)
emit('plan',accepted=plan is not None,guard=guard,model_called=False,producer='placement_rules_v1')
raise SystemExit(0 if plan else 2)
