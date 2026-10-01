#!/usr/bin/env python3
"""GPU direct/residual BC; select only on preregistered development rollouts."""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from v10_common import ROOT,protocol,digest,reference_demo,run_case,full_gate,failure_labels
from fromrealhand.multivideo import phase_diagnostics
from fromrealhand.policy_learning import model_to_device, lift_success
from mjrl.policies.gaussian_mlp import MLP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/processed/dual_video_v10/training_input')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/dual_video_v10/learning')
    args=parser.parse_args()
    study,sha=protocol();plan=study['learning']
    meta=json.loads((args.input/'metadata.json').read_text())
    assert meta['protocol_sha256']==sha and digest(args.input/'dataset.npz')==meta['dataset_sha256']
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this experiment')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    data=np.load(args.input/'dataset.npz');x=data['features'];weights=data['weights']
    references={v['name']:reference_demo(v)['actions'] for v in study['videos']}
    clocks={}
    for v in study['videos']:
        g=np.load(ROOT/v['paths']['geometry'])
        clocks[v['name']]=[.01,float((g['source_frames'][-1]-g['source_frames'][0])/g['fps']),v['control']['time_scale']]
    # All admitted training scenarios are development-only selection cases.
    eval_cases=meta['trajectories']
    source_demos={p:pickle.loads(Path(p).read_bytes()) for p in set(e['dataset'] for e in eval_cases)}
    selection_protocol=dict(protocol_sha256=sha,training_input_sha256=digest(args.input/'metadata.json'),
        evaluation_cases=[dict(video=e['video'],name=e['name'],geometry_sha256=e['geometry_sha256'],seed=e['seed']) for e in eval_cases],
        selection='Equal-video mean full success, then equal-video mean goal distance, then earlier epoch',
        lr=.001,input_scale_floor=.001,output_scale_floor=.001,device=torch.cuda.get_device_name(0))
    (args.output/'selection_protocol.json').write_text(json.dumps(selection_protocol,indent=2)+'\n')
    frozen={}
    for method in plan['methods']:
        folder=args.output/method;folder.mkdir()
        y=data['actions'] if method=='direct_bc' else data['residuals']
        policy=MLP(SimpleNamespace(observation_dim=x.shape[1],action_dim=30),hidden_sizes=tuple(plan['hidden_sizes']),seed=plan['seed'],init_log_std=-2.)
        shift=np.average(x,axis=0,weights=weights)
        scale=np.maximum(np.sqrt(np.average((x-shift)**2,axis=0,weights=weights)),.001)
        out_shift=np.average(y,axis=0,weights=weights)
        out_scale=np.maximum(np.sqrt(np.average((y-out_shift)**2,axis=0,weights=weights)),.001)
        for model in (policy.model,policy.old_model): model.set_transformations(shift,scale,out_shift,out_scale)
        model_to_device(policy.model,'cuda')
        optimizer=torch.optim.Adam(policy.model.parameters(),lr=.001)
        features=torch.as_tensor(x,dtype=torch.float32,device='cuda')
        targets=torch.as_tensor(y,dtype=torch.float32,device='cuda')
        w=torch.as_tensor(weights,dtype=torch.float32,device='cuda')
        rng=np.random.RandomState(plan['seed']);losses=[];comparisons=[]
        for epoch in range(1,max(plan['epochs'])+1):
            order=rng.choice(len(x),size=len(x),replace=True,p=weights/weights.sum())
            for start in range(0,len(order),plan['batch_size']):
                ids=torch.as_tensor(order[start:start+plan['batch_size']],device='cuda')
                optimizer.zero_grad()
                loss=torch.nn.functional.mse_loss(policy.model(features[ids]),targets[ids])
                loss.backward();optimizer.step()
            with torch.no_grad():
                mse=float((((policy.model(features)-targets)**2).mean(1)*w).sum().cpu())
            losses.append(dict(epoch=epoch,balanced_action_mse=mse))
            if epoch%10==0: print(json.dumps(dict(method=method,epoch=epoch,rmse=float(np.sqrt(mse)))),flush=True)
            if epoch not in plan['epochs']: continue
            model_to_device(policy.model,'cpu')
            policy.set_param_values(policy.get_param_values())
            checkpoint=dict(policy=policy,method=method,epoch=epoch,references=references,clocks=clocks,protocol_sha256=sha)
            path=folder/('epoch_%03d.pickle'%epoch)
            with path.open('xb') as stream: pickle.dump(checkpoint,stream)
            reports=[]
            for entry in eval_cases:
                video=study['videos'][entry['video_id']]
                assert digest(entry['geometry'])==entry['geometry_sha256']
                case_dir=folder/('epoch_%03d'%epoch)/video['name']/entry['name']
                report,demo=run_case(video,dict(geometry=entry['geometry']),None,case_dir,seed=entry['seed'],checkpoint=checkpoint)
                expert=source_demos[entry['dataset']][entry['name']]
                phases=phase_diagnostics(demo,references[video['name']],video,np.load(entry['geometry']),expert)
                result=dict(video=video['name'],case=entry['name'],report=report,phases=phases,full_pass=full_gate(report),
                            lift_pass=lift_success(report),failure_labels=failure_labels(report),rollout=str(case_dir/'diagnostic_rollout.pkl'))
                reports.append(result)
                (case_dir/'phases.json').write_text(json.dumps(phases,indent=2)+'\n')
                print(json.dumps(dict(method=method,epoch=epoch,video=video['name'],case=entry['name'],passed=result['full_pass'],goal_mm=report['final_distance_m']*1000,failures=result['failure_labels'])),flush=True)
            fraction=float(np.mean([np.mean([r['full_pass'] for r in reports if r['video']==v['name']]) for v in study['videos']]))
            distance=float(np.mean([np.mean([r['report']['final_distance_m'] for r in reports if r['video']==v['name']]) for v in study['videos']]))
            summary=dict(method=method,epoch=epoch,policy=str(path),policy_sha256=digest(path),full_fraction=fraction,mean_goal_m=distance,
                         full_count=sum(r['full_pass'] for r in reports),case_count=len(reports),offline_action_rmse=float(np.sqrt(mse)),reports=reports)
            comparisons.append(summary)
            (folder/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
            (folder/'losses.json').write_text(json.dumps(losses,indent=2)+'\n')
            model_to_device(policy.model,'cuda')
        model_to_device(policy.model,'cpu')
        best=min(comparisons,key=lambda r:(-r['full_fraction'],r['mean_goal_m'],r['epoch']))
        frozen[method]={k:v for k,v in best.items() if k!='reports'}
    result=dict(protocol_sha256=sha,training_input_sha256=digest(args.input/'metadata.json'),
                selection_protocol_sha256=digest(args.output/'selection_protocol.json'),policies=frozen,
                heldout_evaluated=False,long_training_started=False)
    (args.output/'frozen_policies.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FROZEN',json.dumps(result),flush=True)


if __name__=='__main__': main()
