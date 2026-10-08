"""Explicit rule-only placement requests, separate from the frozen LoRA contract."""
import json
from pathlib import Path

INSTRUCTIONS = ('把杯子放到目标位置的桌面上', '把杯子放到目标位置并返回起点',
                '抓起杯子搬到目标位置放下并返回起点', '搬运杯子放下并返航')
SKILLS = ['reach','grasp','lift','transport','place','return_home']


def placement_plan(instruction, scene):
    text=instruction.strip().rstrip('。！!')
    if scene not in ('first','second'):raise ValueError('Unknown grasp scene')
    if text not in INSTRUCTIONS:return None
    return dict(scene=scene,goal='place',skills=list(SKILLS),planner='placement_rules_v1')


def validated_placement_plan(storage, output):
    output=Path(output).resolve()
    if output.parent!=Path(storage).resolve()/'language/desktop_runs':
        raise ValueError('Invalid placement run directory')
    record=json.loads((output/'placement-request.json').read_text())
    expected=placement_plan(record['instruction'],record['scene'])
    if expected is None or record.get('plan')!=expected or record.get('producer')!='placement_rules_v1':
        raise ValueError('Invalid placement rule provenance')
    return expected
