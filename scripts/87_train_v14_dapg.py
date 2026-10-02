#!/usr/bin/env python3
"""Contact-aware residual DAPG: smoke by default, long mode gated by readiness."""
import argparse
import copy
import importlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

study=importlib.import_module('86_v14_contact_readiness')
from v10_common import ROOT,digest,run_case,surface
from fromrealhand.residual_sampling import ResidualSampler
from fromrealhand.contact_training import contact_costs,bounded_update
from mjrl.algos.dapg import DAPG
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.process_samples import compute_returns,compute_advantages


def sample(cp,video,entry,reward,deterministic=False):
    box=[]
    def factory(exp):
        c=ResidualSampler(cp,exp,video,surface.video.contact_details,deterministic)
        box.append(c);return c
    report,demo=run_case(video,dict(geometry=entry['geometry']),None,seed=entry['seed'],action_factory=factory)
    c=box[0];path=c.path(demo)
    costs=contact_costs(c.exp.last_step_metrics,reward)
    if len(costs)!=len(path['rewards']): raise RuntimeError('Reward and state timing mismatch')
    path['native_rewards']=path['rewards'].copy()
    path['contact_costs']=costs
    path['rewards']=path['rewards']-costs
    return report,demo,path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v14c-study.json')
    parser.add_argument('--long',action='store_true')
    parser.add_argument('--iterations',type=int,default=20)
    args=parser.parse_args()
    study.PLAN=args.config.resolve()
    study.RUN=ROOT/'data/processed'/('dual_video_'+json.loads(study.PLAN.read_text())['version'])
    p,gates,parent,admission,_=study.inputs()
    cfg_path=ROOT/'configs/v14-dapg.json';cfg=json.loads(cfg_path.read_text())
    if args.iterations<1 or (not args.long and args.iterations!=cfg['smoke_iterations']):
        parser.error('Default mode requires the registered 20-iteration smoke; long training requires --long and readiness')
    freeze=json.loads((study.RUN/'freeze.json').read_text())
    if digest(freeze['policy'])!=freeze['policy_sha256'] or freeze['protocol_sha256']!=digest(study.PLAN):
        raise ValueError('Changed frozen checkpoint')
    for name,sha in freeze['code_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Frozen execution code changed: '+name)
    if args.long:
        ready=json.loads((study.RUN/'readiness.json').read_text())
        if not ready['ready_for_long_training'] or ready['policy_sha256']!=freeze['policy_sha256'] or ready['training_config_sha256']!=digest(cfg_path):
            raise RuntimeError('Long-training readiness missing or changed')
    output=study.RUN/('long_training' if args.long else 'dapg_smoke')
    output.mkdir(exist_ok=False)
    cp=pickle.loads(Path(freeze['policy']).read_bytes());policy=cp['policy']
    torch.set_num_threads(1);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable for value baseline')
    nominal=[next(e for e in admission['reports'] if e['video']==v['name'] and e['name']==
                 ('video_seed_0' if v['id']==0 else 'nominal')) for v in parent['videos']]
    checks=[]
    for video,entry in zip(parent['videos'],nominal):
        r,d,trajectory=sample(cp,video,entry,cfg['reward'],True)
        baseline=pickle.loads((study.RUN/'development'/video['name']/entry['name']/'rollout.pkl').read_bytes())['video_faithful']
        obs_error=float(np.max(np.abs(d['observations']-baseline['observations'])))
        act_error=float(np.max(np.abs(d['actions']-baseline['actions'])))
        if max(obs_error,act_error)>1e-8: raise RuntimeError('Residual sampler changed the frozen controller')
        checks.append(dict(video=video['name'],observation_error=obs_error,action_error=act_error,
                          **study.old.gates(r,gates)))
    if not all(c['task_pass'] for c in checks): raise RuntimeError('Nominal preflight failed')
    study.write_json(output/'preflight.json',dict(checks=checks,policy_sha256=freeze['policy_sha256'],
        training_config_sha256=digest(cfg_path),long_training=args.long))
    std=np.clip(policy.model.out_scale.detach().numpy()*.2,1e-4,.005)
    policy.min_log_std=-10.;policy.log_std.data=torch.as_tensor(np.log(std),dtype=torch.float32)
    policy.set_param_values(policy.get_param_values())
    data=np.load(study.old.RUN/'aligned_bc/input.npz');stride=cfg['demo_stride']
    # Demonstrations label the expert's raw proposal; the filter maps proposals to executed controls.
    demo_paths=[dict(observations=data['features'][::stride],actions=data['labels'][::stride])]
    baseline=MLPBaseline(SimpleNamespace(observation_dim=84),epochs=2,use_gpu=True)
    agent=DAPG(None,policy,baseline,demo_paths=demo_paths,lam_0=cfg['lambda_0'],lam_1=cfg['lambda_1'],
        FIM_invert_args={'iters':cfg['cg_iterations'],'damping':1e-3},seed=cfg['seed'],save_logs=True)
    logs=[]
    for iteration in range(1,args.iterations+1):
        paths=[];reports=[]
        for video in parent['videos']:
            entries=[e for e in admission['reports'] if e['video']==video['name']]
            entry=entries[(iteration-1)%len(entries)]
            r,_,path=sample(cp,video,entry,cfg['reward'])
            paths.append(path)
            reports.append(dict(video=video['name'],case=entry['name'],report=r,**study.old.gates(r,gates)))
        compute_returns(paths,.995);compute_advantages(paths,baseline,.995,.97)
        stats,update=bounded_update(agent,paths,cfg['max_measured_kl'])
        errors=baseline.fit(paths,return_errors=True)
        if not np.isfinite(policy.get_param_values()).all() or not np.isfinite(errors).all():
            raise RuntimeError('Nonfinite DAPG update')
        row=dict(iteration=iteration,sampled_steps=sum(len(t['actions']) for t in paths),reports=reports,
                 returns=stats,baseline_errors=list(errors),contact_cost_sum=sum(float(t['contact_costs'].sum()) for t in paths),**update)
        logs.append(row);study.write_json(output/'iterations.json',logs)
        if iteration in (1,5,20) or iteration%20==0 or iteration==args.iterations:
            with (output/('iteration_%04d.pickle'%iteration)).open('xb') as f: pickle.dump(cp,f)
        print('DAPG',json.dumps({k:v for k,v in row.items() if k!='reports'}),flush=True)
    post=[]
    for video,entry in zip(parent['videos'],nominal):
        r,d,_=sample(cp,video,entry,cfg['reward'],True)
        post.append(dict(video=video['name'],report=r,**study.old.gates(r,gates)))
        with (output/(video['name']+'_rollout.pkl')).open('xb') as f: pickle.dump({'video_faithful':d},f)
    study.write_json(output/'summary.json',dict(completed_iterations=len(logs),nonzero_updates=sum(r['parameter_delta_l2']>0 for r in logs),
        finite=True,max_measured_kl=max(r['measured_kl'] for r in logs),preflight=checks,post_nominal=post,
        training_config_sha256=digest(cfg_path),frozen_policy_sha256=freeze['policy_sha256'],
        heldout_evaluated=False,long_training=args.long,
        scope='Policy plus predictive action filter, CPU MuJoCo and actor; CUDA value baseline. No standalone-network claim.'))
    print('COMPLETE',len(logs),'updates',flush=True)


if __name__=='__main__': main()
