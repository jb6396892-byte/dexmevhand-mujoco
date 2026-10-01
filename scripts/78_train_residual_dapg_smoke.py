#!/usr/bin/env python3
"""20-iteration DAPG interface check on development scenes, not long training."""
import argparse
import importlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

study = importlib.import_module('77_run_v12_study')
from v10_common import ROOT, run_case, surface, digest
from fromrealhand.residual_sampling import ResidualSampler
from mjrl.algos.dapg import DAPG
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.process_samples import compute_returns, compute_advantages


def sample(checkpoint, video, entry, deterministic=False):
    box=[]
    def factory(exp):
        controller=ResidualSampler(checkpoint,exp,video,surface.video.contact_details,deterministic)
        box.append(controller)
        return controller
    report,demo=run_case(video,dict(geometry=entry['geometry']),None,seed=entry['seed'],action_factory=factory)
    return report,demo,box[0].path(demo)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args()
    plan,parent,admission,demos,_,_=study.load_inputs()
    freeze=json.loads((study.RUN/'frozen-policy.json').read_text())
    chosen=freeze['selected']; path=Path(chosen['path'])
    if digest(path)!=chosen['sha256']: raise ValueError('Frozen candidate changed')
    checkpoint=pickle.loads(path.read_bytes());policy=checkpoint['policy']
    output=study.RUN/'dapg_smoke'; output.mkdir(exist_ok=False)
    torch.set_num_threads(1);np.random.seed(plan['seed']+78);torch.manual_seed(plan['seed']+78)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for the value baseline')
    nominal=[next(r for r in admission['reports'] if r['video_id']==v['id'] and r['name']==
                 ('video_seed_0' if v['id']==0 else 'nominal')) for v in parent['videos']]
    checks=[]
    for video,entry in zip(parent['videos'],nominal):
        report,demo,trajectory=sample(checkpoint,video,entry,True)
        original,expected=study.execute(checkpoint,video,entry)
        error=float(np.max(np.abs(demo['observations']-expected['observations'])))
        action_error=float(np.max(np.abs(demo['actions']-expected['actions'])))
        if max(error,action_error)>1e-8: raise RuntimeError('Residual sampler changed physical execution')
        half,_=study.execute(checkpoint,video,entry,half=True)
        checks.append(dict(video=video['name'],observation_error=error,action_error=action_error,
                           feature_dim=trajectory['observations'].shape[1],horizon=len(trajectory['actions']),
                           report=report,half_report=half,half_task_pass=study.gates(half,plan)['task_pass'],
                           **study.gates(report,plan)))
    preflight=dict(policy_sha256=digest(path),protocol_sha256=digest(study.PLAN),checks=checks,
         device_scope='MuJoCo and legacy MJRL natural-gradient policy on CPU; value baseline on CUDA',
         action_contract='Unclipped residual for Gaussian likelihood; bounded residual plus reference for env.step; native scaling once',
         heldout_used=False,long_training=False)
    study.write_json(output/'preflight.json',preflight)
    if not args.train:
        print('PREFLIGHT',json.dumps(preflight),flush=True);return
    if not all(c['task_pass'] and c['half_task_pass'] for c in checks):
        raise RuntimeError('Nominal standard/half-step task gate failed; not starting short training')
    # Keep exploration comparable to BC output scale, not legacy std=exp(-2).
    std=np.clip(policy.model.out_scale.detach().numpy()*.2,1e-4,.005)
    policy.min_log_std=-10.
    policy.log_std.data=torch.as_tensor(np.log(std),dtype=torch.float32)
    policy.set_param_values(policy.get_param_values())
    stride=plan['dapg']['demo_stride']
    if checkpoint.get('contact_guard'):
        from fromrealhand.contact_guard import ContactGuard
        from fromrealhand.multivideo import conditioned_features
        demo_paths=[]
        for video,entry in zip(parent['videos'],nominal):
            demonstration=demos[video['name']+'/'+entry['name']];boxes=[]
            class Expert:
                def __init__(self,exp):
                    self.exp=exp;self.guard=ContactGuard(exp,checkpoint['contact_guard'])
                    self.x=[];self.y=[];boxes.append(self)
                def __len__(self): return video['horizon']
                def __getitem__(self,step):
                    e=self.exp.env;d=e.sim.data
                    self.x.append(conditioned_features(e._get_observations(),d.qpos,d.qvel,step,e.control_timestep,
                                      self.exp.duration,video['control']['time_scale'],video['id']))
                    action=demonstration['actions'][step]
                    self.y.append(action-checkpoint['references'][video['name']][step]-self.guard.correction())
                    return action.copy()
            _,replay=run_case(video,dict(geometry=entry['geometry']),None,seed=entry['seed'],action_factory=Expert)
            if np.max(np.abs(replay['observations']-demonstration['observations']))>1e-8:
                raise RuntimeError('Guard-aware expert labels differ from physical replay')
            demo_paths.append(dict(observations=np.asarray(boxes[0].x)[::stride],actions=np.asarray(boxes[0].y)[::stride]))
    else:
        data=np.load(study.RUN/chosen['method']/'input.npz')
        demo_paths=[dict(observations=data['features'][::stride],actions=data['labels'][::stride])]
    baseline=MLPBaseline(SimpleNamespace(observation_dim=policy.n),epochs=2,use_gpu=True)
    agent=DAPG(None,policy,baseline,demo_paths=demo_paths,
               FIM_invert_args={'iters':plan['dapg']['cg_iterations'],'damping':1e-3},
               lam_0=.01,lam_1=.95,seed=plan['seed'],save_logs=True)
    log=[]
    for iteration in range(1,plan['dapg']['iterations']+1):
        paths=[];reports=[]
        for video in parent['videos']:
            choices=[e for e in admission['reports'] if e['video_id']==video['id']]
            entry=choices[(iteration-1)%len(choices)]
            report,_,trajectory=sample(checkpoint,video,entry)
            paths.append(trajectory);reports.append(dict(video=video['name'],case=entry['name'],report=report,**study.gates(report,plan)))
        compute_returns(paths,.995)
        compute_advantages(paths,baseline,.995,.97)
        before=policy.get_param_values().copy()
        stats=agent.train_from_paths(paths)
        errors=baseline.fit(paths,return_errors=True)
        params=policy.get_param_values()
        if not np.isfinite(params).all() or not np.isfinite(stats).all() or not np.isfinite(errors).all():
            raise RuntimeError('Nonfinite short-training update')
        row=dict(iteration=iteration,sampled_frames=sum(len(p['actions']) for p in paths),
                 returns=stats,baseline_errors=list(errors),parameter_delta_l2=float(np.linalg.norm(params-before)),
                 kl=float(agent.logger.log['kl_dist'][-1]),reports=reports)
        log.append(row);study.write_json(output/'iterations.json',log)
        if iteration in (1,5,20):
            with (output/('iteration_%03d.pickle'%iteration)).open('xb') as stream: pickle.dump(checkpoint,stream)
        print('DAPG',json.dumps({k:v for k,v in row.items() if k!='reports'}),flush=True)
    post=[]
    for video,entry in zip(parent['videos'],nominal):
        report,demo=study.execute(checkpoint,video,entry)
        post.append(dict(video=video['name'],report=report,**study.gates(report,plan)))
        with (output/(video['name']+'_rollout.pkl')).open('xb') as stream: pickle.dump({'video_faithful':demo},stream)
    study.write_json(output/'summary.json',dict(completed_iterations=len(log),
        nonzero_updates=sum(r['parameter_delta_l2']>0 for r in log),finite=True,
        policy_sha256=digest(output/'iteration_020.pickle'),preflight=preflight,nominal_post_training=post,
        heldout_evaluated=False,deploy_or_replace_bc=False,long_training=False,
        scope='Interface smoke only: 2 full trajectories/iteration, development distribution; not generalization evidence'))
    print('DAPG_COMPLETE',len(log),'iterations',flush=True)


if __name__=='__main__': main()
