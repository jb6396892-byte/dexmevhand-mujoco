#!/usr/bin/env python3
"""Syntax/hash inventory only. Does not import runtime modules or run tests."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FILES=[
    'configs/tabletop-control-candidate.json',
    'src/fromrealhand/tabletop/control.py', 'src/fromrealhand/tabletop/control_scene.py',
    'src/fromrealhand/tabletop/vision_client.py', 'src/fromrealhand/perception/tracking.py',
    'src/fromrealhand/desktop/tabletop_window.py', 'scripts/142_tabletop_vision_worker.py',
    'scripts/143_stream_visual_grasp.py', 'scripts/144_qt_tabletop.py',
    'scripts/145_launch_tabletop_qt.sh', 'scripts/146_record_tabletop_pretest.py',
    'tests/test_tabletop_control_candidate.py', 'docs/TABLETOP_CONTROL_PRETEST.md',
    'docs/presentation/tabletop_control_pretest/README.md',
    'docs/presentation/tabletop_control_pretest/qt-layout-offline.png',
    'docs/run_logs/2026-10-03-tabletop-control-pretest.md']


def main():
    checked=[]
    for name in FILES:
        p=ROOT/name
        if p.suffix=='.py': ast.parse(p.read_text(),filename=name); checked.append(name)
        elif p.suffix=='.json': json.loads(p.read_text()); checked.append(name)
        elif p.suffix=='.sh': subprocess.run(['bash','-n',str(p)],check=True); checked.append(name)
    # Preserve previously frozen perception source; this is a hash check, not rerunning its benchmark.
    protocol=json.loads((ROOT/'docs/presentation/tabletop_rgbd/evidence/protocol.json').read_text())
    for name, sha in protocol['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Previous perception source changed: '+name)
    subprocess.run(['git','diff','--check'],cwd=str(ROOT),check=True)
    report=dict(status='ready_for_authorized_testing_not_validated',client_date='2026-10-03',
        syntax_checked=checked,previous_perception_source_unchanged=True,
        unit_tests_executed=False,physics_tests_executed=False,end_to_end_tests_executed=False,
        new_training_started=False,automatic_tabletop_execution_locked=True,
        screenshot='Offline Qt layout with previous perception evidence; not a grasp test',
        sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES})
    destination=ROOT/'docs/presentation/tabletop_control_pretest/pretest-inventory.json'
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(status=report['status'],files=len(FILES),tests_executed=False)))


if __name__=='__main__': main()
