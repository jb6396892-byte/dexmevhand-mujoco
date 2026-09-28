#!/usr/bin/env python3
"""Export replay-verified precise transport demonstrations for the first video."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
from importlib import import_module
import numpy as np

transport = import_module('55_optimize_transport')
from check_codex_budget import latest_usage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--quota-stop', type=float, default=85.)
    args = p.parse_args()
    raw = args.candidate.read_bytes()
    source = json.loads(raw)
    if not source.get('training_ready') or not source.get('precision_20mm_passed'):
        raise ValueError('Precise physical and fidelity admission is required')
    root = transport.finger.surface.ROOT
    expected = root/'data/processed/seq_dexycb_001/scene_fidelity_v2/retarget/geometry.npz'
    if Path(source['geometry']).resolve() != expected.resolve():
        raise ValueError('This augmentation set is defined only for the first-video geometry')
    b = source['best']
    cases = [('video_seed_%d'%seed, Path(source['geometry']), seed) for seed in range(6)]
    for dataset in ('expanded', 'orientation'):
        folder = root/'data/processed/seq_dexycb_001/learning_v4'/dataset
        admission = json.loads((folder/'admission.json').read_text())
        for report in admission['reports']:
            if report.get('admitted') and report['name'] != 'nominal':
                cases.append(('synthetic_'+report['name'], folder/report['name']/'geometry.npz', 0))
    args.output.mkdir(parents=True, exist_ok=False)
    demos, reports = {}, []
    for name, geometry, seed in cases:
        quota = latest_usage()
        if quota and quota['used_percent'] >= args.quota_stop:
            break
        e = transport.TransportExperiment(geometry)
        transport.configure(e, b)
        e.transport_kp, e.transport_ki = b['transport_kp'], b['transport_ki']
        e.max_shift, e.max_speed = b['transport_max_shift_m'], b['transport_max_speed_m_s']
        kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'], seed=seed)
        try:
            report, actions = e.run_surface(**kwargs)
            demo = e.last_demo
            replay, _ = e.run_surface(**kwargs, saved_actions=actions)
            error = float(np.max(np.abs(demo['observations']-e.last_demo['observations'])))
            admitted = bool(error < 1e-8 and all(r['surface_physics_passed'] and r['fidelity_passed'] and r['final_distance_m'] <= .02 for r in (report,replay)))
            if admitted:
                demos[name] = demo
            reports.append(dict(name=name, geometry=str(geometry.resolve()), geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
                                report=report, replay=replay, replay_error=error, admitted=admitted))
            (args.output/'progress.json').write_text(json.dumps(reports, indent=2)+'\n')
            print(json.dumps(dict(name=name, admitted=admitted, final_distance_m=report['final_distance_m'])),flush=True)
        finally:
            e.env.close()
    ready = all('video_seed_%d'%seed in demos for seed in range(6)) and len(reports) == len(cases)
    result = dict(training_ready=ready, reports=reports, completed_cases=len(reports), planned_cases=len(cases),
                  trajectory_count=len(demos), independent_real_sequences=1,
                  source_candidate=str(args.candidate.resolve()), source_candidate_sha256=hashlib.sha256(raw).hexdigest(), quota=latest_usage())
    if ready:
        path = args.output/'demonstrations.pkl'
        with path.open('xb') as stream:
            pickle.dump(demos, stream)
        result.update(demo=str(path.resolve()), demo_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
    print('COMPLETE',json.dumps(dict(training_ready=ready, trajectories=len(demos))),flush=True)


if __name__ == '__main__':
    main()
