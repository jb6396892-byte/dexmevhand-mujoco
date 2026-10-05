#!/usr/bin/env python3
"""Package complete physical successes; fit only train statistics, never train a policy."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from hierarchy_common import ROOT,read,write
from fromrealhand.skill_inputs import fit_normalization,SkillSequenceDataset
from fromrealhand.tabletop.training_inputs import SKILLS,FEATURE_DIM,FEATURE_VERSION


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    protocol=read(a.collection/'protocol.json'); collection=read(a.collection/'collection.json')
    required={(v,s) for v in protocol['videos'] for s in
        protocol['development_train_seeds']+protocol['development_validation_seeds']}
    episodes={}; failures=[]
    for record in collection['records']:
        key=(record['video'],record['seed']); run=Path(record['output'])
        if key not in required or key in episodes: raise ValueError('Unexpected or duplicate scene')
        report=read(run/'report.json')
        if not record['passed'] or not report.get('positive_demonstration'):
            failures.append(dict(video=key[0],seed=key[1],reason=report['reason'])); continue
        if (not report['live_vision'] or report['state_writes_during_execution'] or report['object_forces_applied']
                or report['max_penetration_m']>protocol['limits']['penetration_m']):
            raise ValueError('Physical provenance not accepted')
        if report['completed']!=list(SKILLS): raise ValueError('Missing stage confirmation')
        for name,sha in read(run/'sources.json').items():
            if digest(ROOT/name)!=sha: raise ValueError('Controller changed after collection: '+name)
        truth=np.asarray(report['ground_truth_after_stop'])[:3,3]
        goal=np.asarray(read(run/'goal.json')['goal_world_m'])
        if np.linalg.norm(truth-goal)>protocol['limits']['goal_distance_m']:
            failures.append(dict(video=key[0],seed=key[1],reason='post_stop_actual_goal_error')); continue
        with np.load(str(run/'demonstration.npz'),allow_pickle=False) as saved:
            arrays={k:saved[k].copy() for k in saved.files}
        n=len(arrays['inputs'])
        if arrays['inputs'].shape!=(n,FEATURE_DIM) or any(len(x)!=n or not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError('Invalid feature shapes/values')
        if set(arrays['phases'].tolist())!=set(range(4)): raise ValueError('Incomplete skill sequence')
        split='train' if key[1] in protocol['development_train_seeds'] else 'validation'
        if split!=record['split']: raise ValueError('Episode split changed')
        episodes[key]=(record,arrays)
    missing=sorted(required-set(episodes))
    readiness=dict(ready_for_small_scale_generalization_training=False,training_started=False,
        accepted_episodes=len(episodes),required_episodes=len(required),missing=missing,failures=failures,
        independent_generalization_claim=False,final_holdout_executed=False,feature_version=FEATURE_VERSION,
        checkpoint_compatible=False,packaging_checks_complete=False)
    write(a.output/'readiness.json',readiness)
    if missing:
        print(readiness); return
    stats=fit_normalization([x['inputs'] for r,x in episodes.values() if r['split']=='train'])
    write(a.output/'normalization.json',stats); records=[]
    for (video,seed),(record,arrays) in sorted(episodes.items()):
        run=Path(record['output'])
        for phase,skill in enumerate(SKILLS):
            selected=arrays['phases']==phase
            file=a.output/('%s-%s-%s.npz'%(video,seed,skill))
            np.savez_compressed(str(file),**{k:v[selected] for k,v in arrays.items()})
            records.append(dict(video=video,seed=seed,skill=skill,split=record['split'],artifact=file.name,
                sha256=digest(file),frames=int(selected.sum()),source_report_sha256=digest(run/'report.json'),
                source_inputs_sha256=digest(run/'demonstration.npz'),source=str(run)))
    index=dict(feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,records=records,
        normalization='normalization.json',normalization_sha256=digest(a.output/'normalization.json'),
        state_timing='pre_action',object_input='rgbd_only',sequence_boundary='video_scene_skill',training=False)
    write(a.output/'index.json',index)
    checks={}
    for split in ('train','validation'):
        dataset=SkillSequenceDataset(a.output/'index.json',split); batch=dataset.sample_batch(16,8,0)
        checks[split]=dict(frames=len(dataset),groups=len(dataset.groups),batch_shape=list(batch['inputs'].shape),
            finite=bool(np.isfinite(batch['inputs']).all()))
    import torch
    from types import SimpleNamespace
    from mjrl.policies.gaussian_mlp import MLP
    from fromrealhand.routed_residual import RoutedNetwork
    policy=MLP(SimpleNamespace(observation_dim=FEATURE_DIM,action_dim=30),hidden_sizes=(64,64),seed=0)
    routed=RoutedNetwork(policy.model,8)
    x=torch.as_tensor(batch['inputs'].reshape(-1,FEATURE_DIM),dtype=torch.float32)
    phase=batch['phases']; video=np.asarray([0 if r['video']=='first' else 1 for r in batch['records']])[:,None]
    routes=torch.as_tensor(np.eye(8)[(video*4+phase).ravel()],dtype=torch.float32)
    with torch.no_grad():
        shared_output=policy.model(x); routed_output=routed(x,routes)
    if shared_output.shape!=(128,30) or routed_output.shape!=(128,30): raise ValueError('Network shape mismatch')
    if not torch.isfinite(shared_output).all() or not torch.isfinite(routed_output).all():
        raise ValueError('Nonfinite dry forward')
    checks['untrained_forward_only']=dict(shared_shape=list(shared_output.shape),routed_shape=list(routed_output.shape),
        route_count=8,optimizer_created=False,backward_called=False,checkpoints_saved=False,
        weights_are_random=True,policy_success_not_tested=True)
    write(a.output/'loader_checks.json',checks)
    readiness.update(ready_for_small_scale_generalization_training=True,packaging_checks_complete=True)
    write(a.output/'readiness.json',readiness); print(readiness)


if __name__=='__main__': main()
