#!/usr/bin/env python3
"""Resume an externally interrupted frozen placement batch without changing cases."""
import argparse
import hashlib
import json
from pathlib import Path
from hierarchy_common import ROOT, write
from fromrealhand.whole_table.task import NavigationTask

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--config', type=Path, default=ROOT/'configs/tabletop-placement-v1.json')
p.add_argument('--checkpoint', type=Path, default=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt'))
a = p.parse_args()
manifest = json.loads((a.output/'manifest.json').read_text())
records = json.loads((a.output/'partial.json').read_text())
assert not (a.output/'summary.json').exists(), 'Batch already finished'
assert [r['seed'] for r in records] == [c['seed'] for c in manifest['cases'][:len(records)]]
assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest() == v
           for k, v in manifest['source_sha256'].items()), 'Frozen source changed'
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == manifest['checkpoint_sha256']
task = NavigationTask(ROOT, a.checkpoint, a.config)
assert task.config == manifest['config'], 'Frozen configuration changed'
resume = dict(completed_before_resume=len(records), reason='external conversation interruption',
              source_verified=True, checkpoint_verified=True, archived=[])
try:
    for case in manifest['cases'][len(records):]:
        out = a.output/('seed-'+str(case['seed']))
        result = None
        if (out/'summary.json').exists():
            result = json.loads((out/'summary.json').read_text())
        elif out.exists():
            # Never retry an interrupted actual execution as if it had not happened.
            assert not (out/'execution').exists(), 'Actual execution interrupted; explicit audit required'
            archive = out.with_name(out.name+'-interrupted-preview')
            assert not archive.exists()
            out.rename(archive)
            resume['archived'].append(str(archive))
        write(a.output/'resume.json', resume)
        print('RESUME '+str(case['seed']), flush=True)
        try:
            if result is None:
                result = task.run(case['layout'], out, preferred=case['preferred'], stop_skill='place')
            c = result.get('carry', {}); adapter = result.get('adapter', {})
            placement = result.get('placement', {})
            row = dict(seed=case['seed'], preferred=case['preferred'], selected=result['selected_video'],
                selected_entry_frame=result['selected_entry_frame'], selected_yaw_deg=result['selected_yaw_deg'],
                passed=bool(result['passed'] and result['actual_executions'] == 1 and c.get('passed')
                    and c.get('hold_supported_fraction') == 1. and placement.get('passed')
                    and placement.get('placement_passed') and placement.get('return_home', {}).get('passed')),
                reason=result['reason'], attempts=result['planning_attempts'], actual_executions=result['actual_executions'],
                cup_error_m=c.get('cup_error_m'), carry_penetration_m=c.get('max_grasp_penetration_m'),
                local_penetration_m=result.get('max_penetration_m'), hold_supported_fraction=c.get('hold_supported_fraction'),
                execution_resets=result['execution_resets'], planning_seconds=result['planning_seconds'],
                total_wall_s=result['total_wall_s'], carry_detour=c.get('direct') is False,
                egress=bool(c.get('egress')), strict_passed=all(r.get('strict_passed', False) for r in
                    [c, adapter.get('navigation', {}), adapter.get('approach', {})]),
                cup_xy=case['layout']['objects'][0]['xy'], goal=case['layout']['goal_world_m'], placement=placement)
        except Exception as error:
            row = dict(seed=case['seed'], passed=False, reason=type(error).__name__+': '+str(error))
        records.append(row)
        write(a.output/'partial.json', records)
        print(json.dumps(dict(seed=row['seed'], passed=row['passed'], reason=row['reason'])), flush=True)
    unchanged = all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest() == v
                    for k, v in manifest['source_sha256'].items())
    summary = dict(cases=records, passed=sum(r['passed'] for r in records), total=len(manifest['cases']),
        label=manifest['label'], success_rate=sum(r['passed'] for r in records)/len(manifest['cases']),
        source_unchanged=unchanged, reaches_80_percent=sum(r['passed'] for r in records) >= .8*len(manifest['cases']),
        claim='system with model-based candidate previews; not independent raw-policy trials', resume=resume)
    write(a.output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}, indent=2), flush=True)
finally:
    task.close()
