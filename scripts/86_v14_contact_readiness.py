#!/usr/bin/env python3
"""Develop a predictive contact filter and test predeclared training readiness."""
import argparse
import copy
import importlib
import json
import pickle
import subprocess
from pathlib import Path
import numpy as np
from v10_common import ROOT,digest,run_case,setup_case
from fromrealhand.routed_residual import student_actions
from fromrealhand.predictive_contact import PredictiveActions

old = importlib.import_module('77_run_v12_study')
PLAN = ROOT/'configs/v14-study.json'
RUN = ROOT/'data/processed/dual_video_v14'
write_json = old.write_json


def inputs():
    p=json.loads(PLAN.read_text())
    gates,parent,admission,demos,refs,base=old.load_inputs()
    cp=pickle.loads((ROOT/p['baseline']).read_bytes())
    return p,gates,parent,admission,cp


def execute(cp,video,entry,config,half=False,actions=None):
    controllers=[]
    def factory(exp):
        base=student_actions(cp,exp,video)
        controller=PredictiveActions(base,exp,config)
        controllers.append(controller)
        return controller
    report,demo=run_case(video,dict(geometry=entry['geometry']),actions,seed=entry.get('seed',0),
                         half=half,action_factory=factory if actions is None else None)
    logs=controllers[0].filter.logs if controllers else []
    errors=[r['prediction_error'] for r in logs if r['prediction_error'] is not None]
    audit=dict(prediction_error_max=max(errors or [0.]),
        modified_steps=sum(max(r['offset_max_rad'],r.get('root_offset_max_m',0.))>1e-9 for r in logs),
        max_offset_rad=max([r['offset_max_rad'] for r in logs] or [0.]),
        max_root_offset_m=max([r.get('root_offset_max_m',0.) for r in logs] or [0.]),
        total_filter_seconds=sum(r['elapsed_s'] for r in logs),
        candidate_evaluations=sum(r['candidates'] for r in logs))
    if audit['prediction_error_max'] > 1e-7:
        raise RuntimeError('Branch simulation does not match live physics: '+str(audit))
    return report,demo,audit,logs


def evaluate(pilot):
    p,gates,parent,admission,cp=inputs()
    folder=RUN/('pilot' if pilot else 'development')
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(folder)
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for entry in admission['reports']:
        if pilot and not (entry['video']=='second' and entry['name'] in p['pilot_cases']): continue
        video=parent['videos'][entry['video_id']]
        report,demo,audit,logs=execute(cp,video,entry,p['predictive_filter'])
        case=folder/video['name']/entry['name'];case.mkdir(parents=True)
        with (case/'rollout.pkl').open('xb') as stream: pickle.dump({'video_faithful':demo},stream)
        write_json(case/'filter.json',logs)
        row=dict(video=video['name'],case=entry['name'],report=report,audit=audit,**old.gates(report,gates))
        rows.append(row)
        write_json(folder/'summary.json',dict(protocol_sha256=digest(PLAN),baseline_sha256=digest(ROOT/p['baseline']),
                    reports=rows,summary=old.summarize(rows)))
        print('FILTER',video['name'],entry['name'],row['task_pass'],row['strict_pass'],
              report['max_hand_scene_penetration_m']*1000,report['final_distance_m']*1000,audit,flush=True)


def freeze():
    import torch
    p,gates,parent,admission,cp=inputs()
    dev=json.loads((RUN/'development/summary.json').read_text())
    if dev['protocol_sha256']!=digest(PLAN) or dev['summary']['count']!=35:
        raise ValueError('Incomplete development')
    enough=all(v['task']/v['count']>=p['readiness']['development_task_fraction_each_video']
               for v in dev['summary']['per_video'].values())
    if not enough: raise RuntimeError('Development readiness failed; no freeze for long training')
    standard=[r for r in dev['reports'] if (r['video'],r['case']) in
              [('first','video_seed_0'),('second','nominal')]]
    if len(standard)!=2 or not all(r['task_pass'] for r in standard):
        raise RuntimeError('Nominal standard-step gate failed')
    nominal=[]
    for video in parent['videos']:
        entry=next(e for e in admission['reports'] if e['video']==video['name'] and
                   e['name']==('video_seed_0' if video['id']==0 else 'nominal'))
        r,d,a,_=execute(cp,video,entry,p['predictive_filter'],half=True)
        nominal.append(dict(video=video['name'],report=r,audit=a,**old.gates(r,gates)))
    if not all(r['task_pass'] for r in nominal):
        write_json(RUN/'failed-half-check.json',nominal)
        raise RuntimeError('Nominal half-step gate failed')
    native=pickle.loads((old.RUN/'aligned_bc/policy.pickle').read_bytes())
    head=cp['network'].heads[0]
    for model in (native['policy'].model,native['policy'].old_model):
        model.load_state_dict(head.state_dict())
        model.set_transformations(*[getattr(head,k).numpy() for k in ('in_shift','in_scale','out_shift','out_scale')])
    native['policy'].set_param_values(native['policy'].get_param_values())
    x=torch.as_tensor(np.load(old.RUN/'aligned_bc/input.npz')['features'],dtype=torch.float32)
    with torch.no_grad(): error=float(torch.max(torch.abs(native['policy'].model(x)-head(x))))
    if error>1e-8: raise RuntimeError('Gaussian policy conversion changed mean action')
    native.update(predictive_filter=copy.deepcopy(p['predictive_filter']),protocol_sha256=digest(PLAN),
                  training_label=p['version']+'_predictive_contact')
    target=RUN/'frozen-policy.pickle'
    with target.open('xb') as stream: pickle.dump(native,stream)
    receipt=dict(protocol_sha256=digest(PLAN),policy=str(target),policy_sha256=digest(target),
        baseline_sha256=digest(ROOT/p['baseline']),mean_action_conversion_error=error,
        development=dev['summary'],standard_nominal=standard,half_step=nominal,heldout_used=False,long_training=False,
        code_sha256={name:digest(ROOT/name) for name in ['src/fromrealhand/predictive_contact.py',
                    'src/fromrealhand/residual_sampling.py','scripts/86_v14_contact_readiness.py',
                    'src/fromrealhand/contact_training.py','scripts/87_train_v14_dapg.py','configs/v14-dapg.json']})
    write_json(RUN/'freeze.json',receipt)
    write_json(ROOT/'docs/presentation/v14/evidence/frozen-policy.json',receipt)
    print('FROZEN',target,flush=True)


def heldout():
    p,gates,parent,admission,baseline=inputs()
    path=ROOT/'docs/presentation/v14/evidence/frozen-policy.json'
    record=json.loads(path.read_text())
    if record['protocol_sha256']!=digest(PLAN) or record['policy_sha256']!=digest(record['policy']):
        raise ValueError('Frozen policy or protocol changed')
    for name,sha in record['code_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Frozen execution code changed: '+name)
    if subprocess.check_output(['git','show','HEAD:docs/presentation/v14/evidence/frozen-policy.json'],cwd=str(ROOT))!=path.read_bytes():
        raise ValueError('Commit freeze receipt before heldout evaluation')
    cp=pickle.loads(Path(record['policy']).read_bytes())
    folder=RUN/'heldout';folder.mkdir(exist_ok=False)
    cases=old.heldout_cases(p);rows=[]
    write_json(folder/'receipt.json',dict(freeze_sha256=digest(path),protocol_sha256=digest(PLAN),cases=cases,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),text=True).strip()))
    for video in parent['videos']:
        for case in cases:
            source,g=setup_case(video,case,folder/video['name']/case['name'])
            for mode in ('baseline','predictive'):
                if mode=='baseline':
                    r,d=run_case(video,source,None,action_factory=lambda exp:student_actions(baseline,exp,video))
                    audit={}
                else:
                    r,d,audit,_=execute(cp,video,source,p['predictive_filter'])
                rows.append(dict(video=video['name'],case=case['name'],method=mode,report=r,audit=audit,**old.gates(r,gates)))
                summary={m:old.summarize([r for r in rows if r['method']==m]) for m in {r['method'] for r in rows}}
                write_json(folder/'summary.json',dict(protocol_sha256=digest(PLAN),reports=rows,summary=summary))
                print('HELDOUT',video['name'],case['name'],mode,rows[-1]['task_pass'],rows[-1]['strict_pass'],
                      r['max_hand_scene_penetration_m']*1000,r['final_distance_m']*1000,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['pilot','develop','freeze','heldout'])
    parser.add_argument('--config',type=Path,default=PLAN)
    args=parser.parse_args()
    PLAN=args.config.resolve()
    RUN=ROOT/'data/processed'/('dual_video_'+json.loads(PLAN.read_text())['version'])
    if args.command in ('pilot','develop'): evaluate(args.command=='pilot')
    elif args.command=='freeze': freeze()
    else: heldout()
