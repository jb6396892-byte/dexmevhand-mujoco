#!/usr/bin/env python3
"""Develop only preregistered second-video cases; keep every failure visible."""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from v10_common import ROOT, protocol, reference_demo, setup_case, run_case, full_gate, seeded_actions, shooting_demo, dynamic, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/processed/dual_video_v10/development')
    p.add_argument('--repair-evaluations',type=int,default=300)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    study,sha=protocol();video=study['videos'][1]
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    config=dict(protocol_sha256=sha,repair_evaluations=args.repair_evaluations)
    cp=args.output/'configuration.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Resume configuration changed')
    cp.write_text(json.dumps(config,indent=2)+'\n')
    demos,reports={},[]
    for case in study['development_second']:
        folder=args.output/case['name']
        status=folder/'admission.json'
        if args.resume and status.exists():
            result=json.loads(status.read_text())
            if result['admitted']:
                demos[case['name']]=pickle.loads(Path(result['rollout']).read_bytes())['video_faithful']
            reports.append(result);continue
        source,g=setup_case(video,case,folder)
        (folder/'source.json').write_text(json.dumps(source,indent=2)+'\n')
        candidates=[]
        for name,actions in [('fixed',reference_demo(video)['actions']),('seed',seeded_actions(video,case,source))]:
            report,demo=run_case(video,source,actions,folder/name)
            candidates.append((name,report,actions,demo))
            print(json.dumps(dict(case=case['name'],variant=name,passed=full_gate(report),goal_mm=report['final_distance_m']*1000,ring=report['tail_finger_contact_fraction']['ring'],depth_mm=report['max_hand_scene_penetration_m']*1000)),flush=True)
        passed=[c for c in candidates if full_gate(c[1])]
        if not passed and args.repair_evaluations:
            seed=min(candidates,key=lambda c:(not c[1]['surface_physics_passed'],c[1]['final_distance_m']))
            repair=folder/'repair';repair.mkdir()
            opts=SimpleNamespace(output=repair,start=0,transition=300,ring_weight=6.,goal_weight=1.,force_weight=.02,fd_step=.01,bound=2.)
            problem=dynamic.ShootingProblem(source,shooting_demo(video,source,seed[2]),opts)
            try:
                def cost(x):
                    r=problem.evaluate(x);return float(r@r)
                result=minimize(cost,np.zeros(30),method='Powell',bounds=[(-2.,2.)]*30,
                    options=dict(maxfev=args.repair_evaluations,maxiter=10,xtol=.002,ftol=1e-5))
                (repair/'optimization.json').write_text(json.dumps(dict(success=bool(result.success),message=str(result.message),evaluations=problem.evaluations),indent=2)+'\n')
                (repair/'progress.json').write_text(json.dumps(problem.history,indent=2)+'\n')
            finally:
                problem.env.close()
            actions=np.load(repair/'best_actions.npy')
            report,demo=run_case(video,source,actions,folder/'optimized')
            candidates.append(('optimized',report,actions,demo))
            passed=[c for c in candidates if full_gate(c[1])]
        result=dict(case=case,admitted=False,geometry=source['geometry'],protocol_sha256=sha,
                    candidates=[dict(name=n,report=r) for n,r,_,_ in candidates])
        if passed:
            name,report,actions,demo=min(passed,key=lambda c:c[1]['final_distance_m'])
            replay,replay_demo=run_case(video,source,actions,folder/'replay')
            error=float(np.max(np.abs(demo['observations']-replay_demo['observations'])))
            half,_=run_case(video,source,actions,folder/'half_timestep',half=True)
            admitted=bool(error<1e-8 and full_gate(replay) and full_gate(half))
            result.update(selected=name,replay=replay,half_timestep=half,replay_error=error,admitted=admitted,
                          rollout=str((folder/name/'diagnostic_rollout.pkl').resolve()))
            if admitted:demos[case['name']]=demo
        status.write_text(json.dumps(result,indent=2)+'\n')
        reports.append(result)
        (args.output/'progress.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('CASE',json.dumps(dict(name=case['name'],admitted=result['admitted'])),flush=True)
    path=args.output/'demonstrations.pkl'
    if path.exists():raise FileExistsError(path)
    with path.open('xb') as stream:pickle.dump(demos,stream)
    result=dict(protocol_sha256=sha,completed=len(reports)==len(study['development_second']),
                admitted_count=len(demos),planned_count=len(study['development_second']),
                demo=str(path.resolve()),demo_sha256=digest(path),reports=reports,
                independent_real_sequences=1,heldout_evaluated=False,training_started=False)
    (args.output/'admission.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',json.dumps({k:result[k] for k in ['completed','admitted_count','planned_count']}),flush=True)


if __name__=='__main__':main()
