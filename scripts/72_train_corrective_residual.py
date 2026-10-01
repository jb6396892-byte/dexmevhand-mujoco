#!/usr/bin/env python3
"""Phase-balanced residual BC and admitted on-policy corrective labels."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from v10_common import ROOT,digest,run_case,full_gate,failure_labels,surface
from v11_common import RUN,protocol
from fromrealhand.multivideo import trajectory_arrays,phase_diagnostics
from fromrealhand.corrective_learning import CorrectiveActions,phase_weights
from fromrealhand.policy_learning import model_to_device,lift_success
from mjrl.policies.gaussian_mlp import MLP


def balance(entries,phase_mass):
    xs,ys,weights=[],[],[]
    for vid,x,y in entries:
        count=sum(v==vid for v,_,_ in entries)
        xs.append(x);ys.append(y);weights.append(.5/count*phase_weights(len(x),phase_mass))
    if {v for v,_,_ in entries}!={0,1}: raise ValueError('Both videos must have admitted labels')
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(weights)


def collect(video,entry,demo,refs,limits,checkpoint,gain,beta,folder):
    box=[]
    def factory(exp):
        controller=CorrectiveActions(exp,video,demo,refs[video['name']],limits[video['name']],checkpoint,gain,beta)
        box.append(controller);return controller
    report,rollout=run_case(video,dict(geometry=entry['geometry']),None,folder,seed=entry['seed'],action_factory=factory)
    c=box[0]
    arrays=dict(features=np.asarray(c.features),labels=np.asarray(c.labels),feedback=np.asarray(c.feedback),
                hand_state_error_norm_mixed_units=np.asarray(c.qerrors))
    return report,rollout,arrays


def evaluate(checkpoint,label,folder,cases,demos,parent,preferred):
    path=folder/(label+'.pickle')
    with path.open('xb') as stream: pickle.dump(checkpoint,stream)
    reports=[]
    for entry in cases:
        video=parent['videos'][entry['video_id']];key=video['name']+'/'+entry['name']
        dest=folder/label/key
        report,rollout=run_case(video,dict(geometry=entry['geometry']),None,dest,seed=entry['seed'],checkpoint=checkpoint)
        phases=phase_diagnostics(rollout,checkpoint['references'][video['name']],video,np.load(entry['geometry']),demos.get(key))
        row=dict(video=video['name'],case=entry['name'],full_pass=full_gate(report),preferred=bool(full_gate(report) and report['max_hand_scene_penetration_m']<=preferred),
                 lift_pass=lift_success(report),report=report,phases=phases,failure_labels=failure_labels(report),rollout=str(dest/'diagnostic_rollout.pkl'))
        reports.append(row)
        print('EVAL',json.dumps(dict(model=label,video=video['name'],case=entry['name'],passed=row['full_pass'],
                                    depth_mm=report['max_hand_scene_penetration_m']*1000,goal_mm=report['final_distance_m']*1000)),flush=True)
    mean=lambda key:float(np.mean([np.mean([r[key] for r in reports if r['video']==v['name']]) for v in parent['videos']]))
    result=dict(label=label,policy=str(path),policy_sha256=digest(path),full_fraction=mean('full_pass'),preferred_fraction=mean('preferred'),
                mean_goal_m=float(np.mean([np.mean([r['report']['final_distance_m'] for r in reports if r['video']==v['name']]) for v in parent['videos']])),
                full_count=sum(r['full_pass'] for r in reports),case_count=len(reports),reports=reports)
    (folder/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def rank(row): return (-row['full_fraction'],-row['preferred_fraction'],row['mean_goal_m'])


def feedback_phases(arrays,bins):
    times=np.arange(len(arrays['feedback']))*.01
    result={}
    for name,lo,hi in zip(['prepare','approach','closure','lift','transport_hold'],bins[:-1],bins[1:]):
        ids=(times>=lo)&(times<(hi if hi is not None else np.inf))
        values=arrays['feedback'][ids]
        if len(values):
            result[name]=dict(frames=int(ids.sum()),mean_abs_action=float(np.mean(np.abs(values))),
                              nonzero_fraction=float(np.mean(np.linalg.norm(values,axis=1)>1e-6)))
    return result


def fit(policy,x,y,weights,epochs,seed,lr,folder,callback=None,checkpoint_epochs=()):
    model_to_device(policy.model,'cuda');optimizer=torch.optim.Adam(policy.model.parameters(),lr=lr)
    features=torch.as_tensor(x,dtype=torch.float32,device='cuda');targets=torch.as_tensor(y,dtype=torch.float32,device='cuda')
    w=torch.as_tensor(weights,dtype=torch.float32,device='cuda');rng=np.random.RandomState(seed);losses=[]
    for epoch in range(1,epochs+1):
        order=rng.choice(len(x),size=len(x),replace=True,p=weights/weights.sum())
        for start in range(0,len(order),128):
            ids=torch.as_tensor(order[start:start+128],device='cuda');optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(policy.model(features[ids]),targets[ids]);loss.backward();optimizer.step()
        with torch.no_grad(): mse=float((((policy.model(features)-targets)**2).mean(1)*w).sum().cpu())
        losses.append(dict(epoch=epoch,weighted_mse=mse))
        if epoch%10==0: print('TRAIN',json.dumps(dict(run=folder.name,epoch=epoch,rmse=float(np.sqrt(mse)))),flush=True)
        if callback is not None and epoch in checkpoint_epochs:
            model_to_device(policy.model,'cpu');policy.set_param_values(policy.get_param_values())
            callback(epoch,policy)
            model_to_device(policy.model,'cuda')
    model_to_device(policy.model,'cpu');policy.set_param_values(policy.get_param_values())
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'losses.json').write_text(json.dumps(losses,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experts',type=Path,default=RUN/'experts')
    parser.add_argument('--output',type=Path,default=RUN/'learning')
    args=parser.parse_args();study,parent,sha=protocol();plan=study['learning']
    admission=json.loads((args.experts/'admission.json').read_text())
    if admission['protocol_sha256']!=sha or not admission['completed']: raise ValueError('Incomplete expert development')
    for name,key in [('demonstrations.pkl','demo_sha256'),('references.pkl','reference_sha256')]:
        if digest(args.experts/name)!=admission[key]: raise ValueError('Changed expert data')
    demos=pickle.loads((args.experts/'demonstrations.pkl').read_bytes());ref_demos=pickle.loads((args.experts/'references.pkl').read_bytes())
    refs={name:d['actions'] for name,d in ref_demos.items()};limits={};clocks={}
    for video in parent['videos']:
        exp=surface.SurfaceExperiment(ROOT/video['paths']['geometry'])
        try:
            conv=-exp.model.actuator_biasprm[:,1]/(exp.model.actuator_gainprm[:,0]*exp.env.act_rng)
            bounds=plan['residual_q_equivalent_limits']
            limits[video['name']]=conv*np.r_[np.full(3,bounds[0]),np.full(3,bounds[1]),np.full(24,bounds[2])]
            clocks[video['name']]=[.01,exp.duration,video['control']['time_scale']]
        finally: exp.env.close()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    cases=admission['reports'];base=[];clipped=0;total=0
    for entry in cases:
        if not entry['admitted']: continue
        video=parent['videos'][entry['video_id']];key=video['name']+'/'+entry['name']
        x,_,y=trajectory_arrays(demos[key],refs[video['name']],video,np.load(entry['geometry']))
        clipped+=int(np.sum(np.abs(y)>limits[video['name']]));total+=y.size
        base.append((video['id'],x,np.clip(y,-limits[video['name']],limits[video['name']])))
    x,y,w=balance(base,plan['phase_mass'])
    np.savez_compressed(args.output/'phase_bc_input.npz',features=x,labels=y,weights=w)
    (args.output/'input.json').write_text(json.dumps(dict(protocol_sha256=sha,expert_sha256=admission['demo_sha256'],
         reference_sha256=admission['reference_sha256'],frames=len(x),trajectories=len(base),label_clip_fraction=clipped/total,
         feature_dim=x.shape[1],device=torch.cuda.get_device_name(0),heldout_used=False,
         runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda),
         phase_mass=plan['phase_mass'],phase_bins_control_s=plan['phase_bins_control_s']),indent=2)+'\n')
    old=pickle.loads((ROOT/'data/processed/dual_video_v10/learning/residual_bc/epoch_050.pickle').read_bytes())
    pilot_names={'first/video_seed_0','first/former_test_cup_y_plus','second/nominal','second/former_test_cup_x_plus'}
    pilot=[]
    for gain in study['expert']['tracking_gain_candidates']:
        rows=[]
        for entry in cases:
            key=entry['video']+'/'+entry['name']
            if key not in pilot_names or not entry['admitted']: continue
            video=parent['videos'][entry['video_id']]
            r,_,_=collect(video,entry,demos[key],refs,limits,old,gain,.5,args.output/'teacher_pilot'/str(gain)/key)
            rows.append(r)
        row=dict(gain=gain,full_count=sum(full_gate(r) for r in rows),preferred_count=sum(full_gate(r) and r['max_hand_scene_penetration_m']<=study['safety']['preferred_penetration_m'] for r in rows),
                 mean_goal_m=float(np.mean([r['final_distance_m'] for r in rows])),reports=rows)
        pilot.append(row);print('PILOT',json.dumps({k:v for k,v in row.items() if k!='reports'}),flush=True)
    teacher=min(pilot,key=lambda r:(-r['full_count'],-r['preferred_count'],r['mean_goal_m']))['gain']
    (args.output/'teacher_pilot.json').write_text(json.dumps(dict(selected_gain=teacher,results=pilot),indent=2)+'\n')
    policy=MLP(SimpleNamespace(observation_dim=84,action_dim=30),hidden_sizes=tuple(plan['hidden_sizes']),seed=plan['seed'],init_log_std=-2.)
    shift=np.average(x,axis=0,weights=w);scale=np.maximum(np.sqrt(np.average((x-shift)**2,axis=0,weights=w)),.001)
    out=np.average(y,axis=0,weights=w);outscale=np.maximum(np.sqrt(np.average((y-out)**2,axis=0,weights=w)),.001)
    for model in (policy.model,policy.old_model): model.set_transformations(shift,scale,out,outscale)
    def checkpoint(model,label):
        return dict(policy=model,method='residual_bc',references=refs,clocks=clocks,residual_limits=limits,
                    protocol_sha256=sha,training_label=label,teacher_gain=teacher)
    comparisons=[];preferred=study['safety']['preferred_penetration_m']
    def callback(epoch,model):
        if epoch in plan['phase_bc_epochs']:
            label='phase_bc_%03d'%epoch
            comparisons.append(evaluate(checkpoint(model,label),label,args.output,cases,demos,parent,preferred))
            (args.output/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    fit(policy,x,y,w,max(plan['phase_bc_epochs']),plan['seed'],.001,args.output/'phase_bc_training',callback,plan['phase_bc_epochs'])
    best=min(comparisons,key=rank);current=pickle.loads(Path(best['policy']).read_bytes());corrections=[];collections=[]
    for iteration,beta in enumerate(plan['dagger_betas'],1):
        rows=[]
        for entry in cases:
            if not entry['admitted']: continue
            video=parent['videos'][entry['video_id']];key=video['name']+'/'+entry['name']
            dest=args.output/('collection_%d'%iteration)/key
            r,_,arrays=collect(video,entry,demos[key],refs,limits,current,teacher,beta,dest)
            np.savez_compressed(dest/'correction_labels.npz',**arrays)
            accepted=full_gate(r)
            if accepted: corrections.append((video['id'],arrays['features'],arrays['labels']))
            row=dict(video=video['name'],case=entry['name'],admitted_labels=accepted,report=r,
                     mean_abs_feedback_action=float(np.mean(np.abs(arrays['feedback']))),
                     nonzero_feedback_fraction=float(np.mean(np.linalg.norm(arrays['feedback'],axis=1)>1e-6)),
                     phase_feedback=feedback_phases(arrays,plan['phase_bins_control_s']))
            rows.append(row)
            print('COLLECT',json.dumps(dict(round=iteration,video=video['name'],case=entry['name'],accepted=accepted,feedback=row['mean_abs_feedback_action'])),flush=True)
        collections.append(dict(round=iteration,beta=beta,rows=rows))
        (args.output/'collections.json').write_text(json.dumps(collections,indent=2)+'\n')
        if not corrections or {v for v,_,_ in corrections}!={0,1}:
            print('No admitted two-video correction data; stop correction training',flush=True);break
        cx,cy,cw=balance(corrections,plan['phase_mass'])
        xx=np.concatenate([x,cx]);yy=np.concatenate([y,cy]);ww=np.r_[w*.5,cw*.5]
        label='dagger_%d'%iteration
        fit(current['policy'],xx,yy,ww,plan['dagger_epochs'],plan['seed']+iteration,.0003,args.output/(label+'_training'))
        current['training_label']=label
        comparisons.append(evaluate(current,label,args.output,cases,demos,parent,preferred))
        (args.output/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
        # Continue DAgger from the most recently trained model, but choose final policy on all development scores.
    best=min(comparisons,key=rank)
    result=dict(protocol_sha256=sha,selected={k:v for k,v in best.items() if k!='reports'},teacher_gain=teacher,
                expert_sha256=admission['demo_sha256'],reference_sha256=admission['reference_sha256'],
                correction_trajectories=len(corrections),correction_frames=sum(len(a) for _,a,_ in corrections),
                heldout_evaluated=False,long_training_started=False)
    (args.output/'frozen_policy.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FROZEN',json.dumps(result),flush=True)


if __name__=='__main__': main()
