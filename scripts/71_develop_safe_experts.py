#!/usr/bin/env python3
"""Increase action safety headroom on development only, with native dynamics."""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize
from v10_common import run_case,full_gate,reference_demo,seeded_actions,dynamic,digest
from v11_common import RUN,protocol,development_cases


class SearchStopped(Exception):
    pass


class SafetyProblem(dynamic.ShootingProblem):
    def action_sequence(self,parameters):
        root=parameters.copy();root[3:]=0
        raw=dynamic.corrected_actions(self.demo['actions'],root,self.scales,self.conversion,0,300)
        t=np.arange(len(raw))*.01
        smooth=lambda x:np.clip(x,0,1)**3*(10-15*np.clip(x,0,1)+6*np.clip(x,0,1)**2)
        pulse=smooth((t-6.3)/.9)*(1-smooth((t-9.8)/1.4))
        fingers=parameters.copy();fingers[:3]=0
        return raw+pulse[:,None]*(fingers*self.scales*self.conversion)[None,:]


def search(video,source,demo,folder,study):
    folder.mkdir()
    args=SimpleNamespace(output=folder,start=study['safety']['shooting_start'],transition=300,
                         ring_weight=6.,goal_weight=1.,force_weight=.02,fd_step=.01,bound=.6)
    problem=SafetyProblem(source,demo,args)
    history=[];best=[float('inf'),None]
    limit=study['safety']['max_shooting_evaluations_per_case']
    preferred=study['safety']['preferred_penetration_m']
    try:
        def objective(x):
            if len(history)>=limit: raise SearchStopped('hard evaluation cap')
            parameters=np.zeros(30)
            parameters[[29,23,14]]=x[:3]*[1,-1,-1]
            parameters[:3]=x[3:]
            residual=problem.evaluate(parameters).copy()
            residual[-32]=80*max(0.,problem.max_depth-(preferred-.0001))
            cost=float(residual@residual)
            row=dict(problem.history[-1]);row.update(safety_cost=cost,parameters=x.tolist())
            history.append(row)
            if cost<best[0]:
                best[:]=[cost,parameters.copy()]
                raw=problem.action_sequence(parameters)
                np.save(folder/'safety_best_actions.npy',np.clip(raw,-1,1))
                (folder/'safety_best.json').write_text(json.dumps(row,indent=2)+'\n')
            final=problem.measure(len(demo['actions'])-1)
            if (row['max_penetration_m']<=preferred-.00003 and row['final_distance_m']<.012
                    and row['min_support_contact_fraction']>=.95 and row['ring_contact_fraction']>=.95
                    and np.linalg.norm(final['tip_error'],axis=1).mean()<.0145):
                raise SearchStopped('preferred margin candidate found; full audit still required')
            return cost
        try:
            result=minimize(objective,np.zeros(6),method='Powell',bounds=[(0,.5)]*3+[(-.6,.6)]*3,
                            options=dict(maxfev=limit,maxiter=10,xtol=.002,ftol=1e-5,direc=np.eye(6)*.05))
            reason=str(result.message)
        except SearchStopped as exc:
            reason=str(exc)
        unchanged=all(np.array_equal(getattr(problem.m,k),v) for k,v in problem.physics.items())
        if not unchanged: raise RuntimeError('Physics parameters changed')
        (folder/'safety_search.json').write_text(json.dumps(dict(reason=reason,evaluations=len(history),physics_unchanged=unchanged,history=history),indent=2)+'\n')
        return np.load(folder/'safety_best_actions.npy')
    finally:
        problem.env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUN/'experts')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();study,parent,sha=protocol()
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    reports=[];demos={};references={}
    for case in development_cases(parent):
        video=parent['videos'][case['video_id']]
        name=video['name']+'/'+case['name'];folder=args.output/name;status=folder/'admission.json'
        if status.exists() and args.resume:
            result=json.loads(status.read_text())
            if result['protocol_sha256']!=sha: raise ValueError('Resume protocol mismatch')
            if result['admitted']: demos[name]=pickle.loads(Path(result['rollout']).read_bytes())['video_faithful']
            reports.append(result);continue
        folder.mkdir(parents=True,exist_ok=True)
        source=dict(geometry=case['geometry'],best=video['control'])
        candidates=[]
        initial=case.get('demo',reference_demo(video))['actions']
        variants=[('original',initial)]
        if 'case' in case: variants.append(('scene_seed',seeded_actions(video,case['case'],source)))
        for tag,actions in variants:
            report,demo=run_case(video,source,actions,folder/tag,seed=case['seed'])
            candidates.append((tag,report,demo))
        preferred=study['safety']['preferred_penetration_m']
        if not any(full_gate(r) and r['max_hand_scene_penetration_m']<=preferred for _,r,_ in candidates):
            start=min(candidates,key=lambda c:(not full_gate(c[1]),c[1]['max_hand_scene_penetration_m'],c[1]['final_distance_m']))
            actions=search(video,source,start[2],folder/'search',study)
            report,demo=run_case(video,source,actions,folder/'optimized',seed=case['seed'])
            candidates.append(('optimized',report,demo))
        passed=sorted([c for c in candidates if full_gate(c[1])],
                      key=lambda c:(c[1]['max_hand_scene_penetration_m']>preferred,c[1]['max_hand_scene_penetration_m'],c[1]['final_distance_m']))
        result=dict(name=case['name'],video=video['name'],video_id=video['id'],geometry=case['geometry'],seed=case['seed'],
                    protocol_sha256=sha,origin=case['origin'],admitted=False,
                    candidates=[dict(name=n,report=r) for n,r,d in candidates])
        for tag,r,demo in passed:
            replay,rd=run_case(video,source,demo['actions'],folder/('replay_'+tag),seed=case['seed'])
            half,_=run_case(video,source,demo['actions'],folder/('half_'+tag),seed=case['seed'],half=True)
            error=float(np.max(np.abs(rd['observations']-demo['observations'])))
            if error<1e-8 and full_gate(replay) and full_gate(half):
                result.update(admitted=True,selected=tag,report=r,half=half,replay_error=error,
                              preferred_margin=bool(max(r['max_hand_scene_penetration_m'],half['max_hand_scene_penetration_m'])<=preferred),
                              rollout=str((folder/tag/'diagnostic_rollout.pkl').resolve()))
                demos[name]=demo;break
        status.write_text(json.dumps(result,indent=2)+'\n');reports.append(result)
        (args.output/'progress.json').write_text(json.dumps(reports,indent=2)+'\n')
        print('CASE',json.dumps(dict(name=name,admitted=result['admitted'],preferred=result.get('preferred_margin'),
                    depth_mm=result.get('report',candidates[0][1])['max_hand_scene_penetration_m']*1000)),flush=True)
    for video in parent['videos']:
        key=video['name']+('/video_seed_0' if video['id']==0 else '/nominal')
        if key not in demos: raise RuntimeError('No admitted nominal reference for '+video['name'])
        references[video['name']]=demos[key]
    for name,data in [('demonstrations.pkl',demos),('references.pkl',references)]:
        with (args.output/name).open('xb') as stream: pickle.dump(data,stream)
    result=dict(protocol_sha256=sha,completed=True,planned_count=len(reports),admitted_count=len(demos),
                preferred_count=sum(r.get('preferred_margin',False) for r in reports),reports=reports,
                demo_sha256=digest(args.output/'demonstrations.pkl'),reference_sha256=digest(args.output/'references.pkl'),heldout_used=False)
    (args.output/'admission.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',json.dumps({k:result[k] for k in ['admitted_count','planned_count','preferred_count']}),flush=True)


if __name__=='__main__': main()
