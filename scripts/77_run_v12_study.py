#!/usr/bin/env python3
"""Controlled phase ablation, reference/contact BC, then frozen fresh evaluation."""
import argparse
import copy
import importlib
import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from v10_common import ROOT, digest, run_case, setup_case, full_gate, surface, failure_labels
from v11_common import protocol as parent_protocol
from fromrealhand.corrective_learning import aligned_phase_weights, aligned_phase_indices
from fromrealhand.multivideo import trajectory_arrays, phase_diagnostics
from fromrealhand.reference_tracking import ReferenceContext, ReferenceActions, task_gate
from fromrealhand.policy_learning import lift_success
from mjrl.policies.gaussian_mlp import MLP

training = importlib.import_module('72_train_corrective_residual')
RUN = ROOT/'data/processed/dual_video_v12'
PLAN = ROOT/'configs/v12-study.json'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def load_inputs():
    _, parent, _ = parent_protocol()
    plan = json.loads(PLAN.read_text())
    directory = ROOT/'data/processed/dual_video_v11'
    admission = json.loads((directory/'experts/admission.json').read_text())
    if not admission['completed'] or not all(r['admitted'] for r in admission['reports']):
        raise ValueError('All development demonstrations must be admitted')
    for name, key in [('demonstrations.pkl', 'demo_sha256'), ('references.pkl', 'reference_sha256')]:
        if digest(directory/'experts'/name) != admission[key]:
            raise ValueError('Expert input changed')
    demos = pickle.loads((directory/'experts/demonstrations.pkl').read_bytes())
    refs = pickle.loads((directory/'experts/references.pkl').read_bytes())
    base = pickle.loads((directory/'learning/phase_bc_150.pickle').read_bytes())
    return plan, parent, admission, demos, refs, base


def gates(report, plan):
    return dict(task_pass=task_gate(report, plan['task_gate']), strict_pass=full_gate(report),
                lift_pass=lift_success(report), failures_strict=failure_labels(report))


def summarize(rows):
    names = sorted({r['video'] for r in rows})
    avg = lambda k: float(np.mean([np.mean([r[k] for r in rows if r['video']==v]) for v in names]))
    return dict(count=len(rows), task_count=sum(r['task_pass'] for r in rows),
                strict_count=sum(r['strict_pass'] for r in rows), lift_count=sum(r['lift_pass'] for r in rows),
                task_fraction=avg('task_pass'), strict_fraction=avg('strict_pass'),
                mean_goal_m=float(np.mean([np.mean([r['report']['final_distance_m'] for r in rows if r['video']==v]) for v in names])),
                mean_peak_penetration_m=float(np.mean([r['report']['max_hand_scene_penetration_m'] for r in rows])),
                per_video={v:dict(task=sum(r['task_pass'] for r in rows if r['video']==v),
                                  strict=sum(r['strict_pass'] for r in rows if r['video']==v),
                                  count=sum(r['video']==v for r in rows)) for v in names})


def execute(checkpoint, video, entry, output=None, half=False):
    source = dict(geometry=entry['geometry'])
    if checkpoint.get('contact_guard'):
        from fromrealhand.multivideo import MultiVideoActions
        from fromrealhand.contact_guard import GuardedActions
        factory=lambda exp: GuardedActions(MultiVideoActions(checkpoint,exp,video),exp,checkpoint['contact_guard'])
        return run_case(video,source,None,output,half=half,seed=entry.get('seed',0),action_factory=factory)
    if checkpoint.get('feature_mode') == 'reference_contact':
        factory = lambda exp: ReferenceActions(checkpoint, exp, video, surface.video.contact_details)
        return run_case(video, source, None, output, half=half, seed=entry.get('seed', 0), action_factory=factory)
    return run_case(video, source, None, output, half=half, seed=entry.get('seed', 0), checkpoint=checkpoint)


def collect_reference_data(parent, admission, demos, refs, checkpoint):
    entries=[]
    for entry in admission['reports']:
        video=parent['videos'][entry['video_id']]; key=video['name']+'/'+entry['name']
        cached=RUN/'reference_data'/video['name']/(entry['name']+'.npz')
        if cached.exists():
            metadata=json.loads(cached.with_suffix('.json').read_text())
            if metadata['protocol_sha256']!=digest(PLAN): raise ValueError('Stale reference features')
            a=np.load(cached); entries.append((video,entry,a['features'],a['labels'])); continue
        geometry={k:v.copy() for k,v in np.load(ROOT/video['paths']['geometry']).items()}
        boxes=[]
        class Replay:
            def __init__(self, exp):
                self.context=ReferenceContext(exp,video,refs[video['name']],geometry,surface.video.contact_details)
                self.features=[]; self.labels=[]; boxes.append(self)
            def __len__(self): return video['horizon']
            def __getitem__(self, step):
                self.features.append(self.context.features(step))
                action=demos[key]['actions'][step]
                self.labels.append(action-self.context.actions[step])
                return action.copy()
        report,replay=run_case(video,dict(geometry=entry['geometry']),None,seed=entry['seed'],action_factory=Replay)
        error=float(np.max(np.abs(replay['observations']-demos[key]['observations'])))
        if not full_gate(report) or error>1e-8: raise RuntimeError('Expert replay mismatch: '+key)
        x=np.asarray(boxes[0].features); y=np.asarray(boxes[0].labels)
        cached.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cached,features=x,labels=y)
        write_json(cached.with_suffix('.json'),dict(replay_error=error,report=report,protocol_sha256=digest(PLAN)))
        entries.append((video,entry,x,y))
        print('FEATURES',key,len(x),'replay_error',error,flush=True)
    return entries


def train(mode):
    plan,parent,admission,demos,refs,base=load_inputs()
    folder=RUN/mode;folder.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for this BC comparison')
    torch.set_num_threads(1)
    entries=[]
    if mode=='aligned_bc':
        for entry in admission['reports']:
            video=parent['videos'][entry['video_id']];key=video['name']+'/'+entry['name']
            x,_,y=trajectory_arrays(demos[key],refs[video['name']]['actions'],video,np.load(entry['geometry']))
            entries.append((video,entry,x,y))
    else:
        entries=collect_reference_data(parent,admission,demos,refs,base)
    xx=[]; yy=[]; ww=[]; audits=[]
    for video,entry,x,y in entries:
        geometry=np.load(entry['geometry'])
        w=aligned_phase_weights(len(x),plan['phase_mass'],geometry,video['control']['time_scale'],
                                source_boundaries=plan['source_phase_boundaries'])
        phase=aligned_phase_indices(len(x),geometry,video['control']['time_scale'])
        count=sum(v['id']==video['id'] for v,_,_,_ in entries)
        xx.append(x);yy.append(np.clip(y,-base['residual_limits'][video['name']],base['residual_limits'][video['name']]))
        ww.append(.5/count*w)
        audits.append(dict(video=video['name'],case=entry['name'],phase_mass=[float(w[phase==i].sum()) for i in range(5)],
                           label_clip_fraction=float(np.mean(np.abs(y)>base['residual_limits'][video['name']]))))
    x,y,w=np.concatenate(xx),np.concatenate(yy),np.concatenate(ww)
    np.savez_compressed(folder/'input.npz',features=x,labels=y,weights=w)
    policy=MLP(SimpleNamespace(observation_dim=x.shape[1],action_dim=30),hidden_sizes=tuple(plan['hidden_sizes']),seed=plan['seed'],init_log_std=-2.)
    shift=np.average(x,axis=0,weights=w); scale=np.maximum(np.sqrt(np.average((x-shift)**2,axis=0,weights=w)),.001)
    out=np.average(y,axis=0,weights=w); outscale=np.maximum(np.sqrt(np.average((y-out)**2,axis=0,weights=w)),.001)
    for model in (policy.model,policy.old_model): model.set_transformations(shift,scale,out,outscale)
    write_json(folder/'input.json',dict(protocol_sha256=digest(PLAN),frames=len(x),feature_dim=x.shape[1],
        gpu=torch.cuda.get_device_name(0),demo_sha256=admission['demo_sha256'],reference_sha256=admission['reference_sha256'],
        phase_audit=audits,heldout_used=False))
    training.fit(policy,x,y,w,plan['epochs'],plan['seed'],.001,folder/'training')
    checkpoint={k:copy.deepcopy(base[k]) for k in ('method','references','clocks','residual_limits')}
    checkpoint.update(policy=policy,protocol_sha256=digest(PLAN),training_label=mode,
                      feature_mode='reference_contact' if mode=='contact_reference_bc' else 'phase')
    if mode=='contact_reference_bc':
        checkpoint['reference_demos']=refs
        checkpoint['nominal_geometry']={v['name']:{k:a.copy() for k,a in np.load(ROOT/v['paths']['geometry']).items()} for v in parent['videos']}
    with (folder/'policy.pickle').open('xb') as stream: pickle.dump(checkpoint,stream)
    print('TRAINED',mode,digest(folder/'policy.pickle'),flush=True)


def develop(mode):
    plan,parent,admission,demos,refs,_=load_inputs()
    folder=RUN/mode;path=folder/'policy.pickle'; checkpoint=pickle.loads(path.read_bytes())
    result_path=folder/'development.json'
    previous=json.loads(result_path.read_text()) if result_path.exists() else None
    if previous and (previous['protocol_sha256']!=digest(PLAN) or previous['policy_sha256']!=digest(path)):
        raise ValueError('Cannot resume changed development experiment')
    rows=previous['reports'] if previous else []
    for entry in admission['reports']:
        video=parent['videos'][entry['video_id']];key=video['name']+'/'+entry['name']
        if any(r['video']==video['name'] and r['case']==entry['name'] for r in rows): continue
        r,demo=execute(checkpoint,video,entry)
        row=dict(video=video['name'],case=entry['name'],report=r,**gates(r,plan))
        row['phases']=phase_diagnostics(demo,refs[video['name']]['actions'],video,np.load(entry['geometry']),demos[key])
        if entry['name'] in ('video_seed_0','nominal'):
            with (folder/(video['name']+'_rollout.pkl')).open('wb') as stream: pickle.dump({'video_faithful':demo},stream)
        rows.append(row)
        write_json(result_path,dict(protocol_sha256=digest(PLAN),policy_sha256=digest(path),reports=rows,summary=summarize(rows)))
        print('DEV',mode,video['name'],entry['name'],json.dumps({**gates(r,plan),'goal_mm':r['final_distance_m']*1000,'depth_mm':r['max_hand_scene_penetration_m']*1000}),flush=True)
    print('DEVELOPED',mode,json.dumps(summarize(rows)),flush=True)


def freeze():
    plan,_,admission,_,_,_=load_inputs(); rows=[]
    for mode in plan['candidates']:
        path=RUN/mode/'policy.pickle'; result=json.loads((RUN/mode/'development.json').read_text())
        if result['summary']['count']!=35 or result['policy_sha256']!=digest(path): raise ValueError('Incomplete/changed development')
        rows.append(dict(method=mode,path=str(path),sha256=digest(path),**result['summary']))
    amendment=ROOT/'configs/v12-contact-guard-amendment.json'
    if amendment.exists():
        for gain in json.loads(amendment.read_text())['gains']:
            mode='guard_%03d'%round(gain*100)
            path=RUN/mode/'policy.pickle';result=json.loads((RUN/mode/'development.json').read_text())
            if result['summary']['count']!=35 or result['policy_sha256']!=digest(path): raise ValueError('Incomplete guard comparison')
            rows.append(dict(method=mode,path=str(path),sha256=digest(path),**result['summary']))
    selected=min(rows,key=lambda r:(-r['task_fraction'],-r['strict_fraction'],r['mean_goal_m']))
    receipt=dict(protocol_sha256=digest(PLAN),selected=selected,candidates=rows,
                 demo_sha256=admission['demo_sha256'],heldout_used=False,long_training=False)
    if amendment.exists(): receipt['guard_amendment_sha256']=digest(amendment)
    path=ROOT/'docs/presentation/v12/evidence/frozen-policy.json'
    if path.exists(): raise FileExistsError(path)
    write_json(path,receipt);write_json(RUN/'frozen-policy.json',receipt)
    print('FROZEN',json.dumps(receipt),flush=True)


def heldout_cases(plan):
    spec=plan['heldout'];rng=np.random.RandomState(spec['seed']);cases=[]
    for i in range(spec['count_per_video']):
        cup=np.r_[rng.uniform(-spec['cup_xy_bound_m'],spec['cup_xy_bound_m'],2),0.]
        goal=np.r_[rng.uniform(-spec['goal_xy_bound_m'],spec['goal_xy_bound_m'],2),0.]
        cases.append(dict(name='fresh_%02d'%i,cup_offset_m=cup.tolist(),goal_offset_m=goal.tolist(),
                          cup_yaw_deg=float(rng.uniform(-spec['yaw_bound_deg'],spec['yaw_bound_deg']))))
    return cases


def heldout():
    plan,parent,_,_,_,_=load_inputs()
    receipt_path=ROOT/'docs/presentation/v12/evidence/frozen-policy.json'
    frozen=json.loads(receipt_path.read_text())
    if frozen['protocol_sha256']!=digest(PLAN): raise ValueError('Changed protocol')
    if frozen.get('guard_amendment_sha256')!=digest(ROOT/'configs/v12-contact-guard-amendment.json'):
        raise ValueError('Changed guard amendment')
    committed=subprocess.check_output(['git','show','HEAD:docs/presentation/v12/evidence/frozen-policy.json'],cwd=str(ROOT))
    if committed!=receipt_path.read_bytes(): raise ValueError('Commit freeze receipt before testing')
    paths={r['method']:Path(r['path']) for r in frozen['candidates']}
    for r in frozen['candidates']:
        if digest(r['path'])!=r['sha256']: raise ValueError('Changed frozen policy')
    paths['old_residual']=ROOT/'data/processed/dual_video_v10/learning/residual_bc/epoch_050.pickle'
    guard=min([r for r in frozen['candidates'] if r['method'].startswith('guard_')],
              key=lambda r:(-r['task_fraction'],-r['strict_fraction'],r['mean_goal_m']))
    paths['selected_guard']=Path(guard['path'])
    paths['fixed_window_v11']=ROOT/'data/processed/dual_video_v11/learning/phase_bc_150.pickle'
    methods=plan['heldout']['methods']+['selected_guard','fixed_window_v11']
    checkpoints={k:pickle.loads(p.read_bytes()) for k,p in paths.items()}
    output=RUN/'heldout';output.mkdir(exist_ok=False)
    rows=[]
    write_json(output/'receipt.json',dict(frozen_sha256=digest(receipt_path),protocol_sha256=digest(PLAN),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip(),cases=heldout_cases(plan)))
    for video in parent['videos']:
        for case in heldout_cases(plan):
            folder=output/video['name']/case['name'];source,g=setup_case(video,case,folder)
            for method in methods:
                r,demo=execute(checkpoints[method],video,source)
                row=dict(video=video['name'],case=case['name'],method=method,report=r,**gates(r,plan))
                rows.append(row)
                if case['name']=='fresh_00':
                    with (folder/(method+'.pkl')).open('xb') as stream: pickle.dump({'video_faithful':demo},stream)
                write_json(output/'progress.json',rows)
                print('HELDOUT',method,video['name'],case['name'],json.dumps({**gates(r,plan),'goal_mm':r['final_distance_m']*1000,'depth_mm':r['max_hand_scene_penetration_m']*1000}),flush=True)
    result=dict(protocol_sha256=digest(PLAN),selected_before_test=frozen['selected']['method'],reports=rows,
                summary={m:summarize([r for r in rows if r['method']==m]) for m in methods},
                tuning_after_test=False,long_training=False)
    write_json(output/'summary.json',result)
    print('HELDOUT_SUMMARY',json.dumps(result['summary']),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','develop','freeze','heldout'])
    parser.add_argument('--mode',default='aligned_bc')
    args=parser.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    if args.stage=='train': train(args.mode)
    elif args.stage=='develop': develop(args.mode)
    elif args.stage=='freeze': freeze()
    else: heldout()


if __name__=='__main__': main()
