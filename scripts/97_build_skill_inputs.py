#!/usr/bin/env python3
"""Stage 4.3: build read-only, reference-conditioned inputs, not skill policies."""
import argparse
import json
import pickle
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.skill_inputs import skill_features,fit_normalization,SkillSequenceDataset,FEATURE_DIM
from fromrealhand.skills import SKILLS
import hashlib


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'data/processed/stage4_pipeline_v2')
    args = parser.parse_args()
    build = args.run/'build'
    manifest = read(build/'manifest.json')
    receipt = read(build/'receipt.json')
    if not receipt['passed'] or receipt['manifest_sha256']!=digest(build/'manifest.json'):
        raise ValueError('Passed 4.2 data admission is required')
    config = manifest['config']
    protocol = read(args.run/'protocol.json')
    references = ROOT/config['references']
    if digest(references)!=protocol['sha256'][config['references']]:
        raise ValueError('Frozen per-video reference changed')
    refs = pickle.loads(references.read_bytes())
    folder = args.run/'inputs'
    folder.mkdir(exist_ok=False)
    records,train = [],[]
    for entry in manifest['trajectories']:
        if digest(entry['geometry'])!=entry['geometry_sha256']: raise ValueError('Geometry changed')
        with np.load(entry['geometry']) as geometry:
            for segment in entry['segments']:
                path = build/segment['artifact']
                if digest(path)!=segment['sha256']: raise ValueError('Skill data changed')
                piece = pickle.loads(path.read_bytes())
                reference = refs[entry['video']]['actions']
                lo,hi = segment['start'],segment['stop']
                inputs = np.stack([skill_features(obs,state['qpos'],state['qvel'],lo+i,
                    entry,segment,geometry,reference) for i,(obs,state) in
                    enumerate(zip(piece['observations'],piece['sim_data']))])
                actions = np.asarray(piece['actions'])
                target = actions-reference[lo:hi]
                if len(inputs)!=hi-lo or not np.isfinite(target).all(): raise ValueError('Input timing mismatch')
                name = str(len(records)).zfill(3)+'.npz'
                np.savez_compressed(folder/name,inputs=inputs,actions=actions,residual_targets=target,
                    reference_actions=reference[lo:hi],global_steps=np.arange(lo,hi),
                    local_steps=np.arange(hi-lo),source_frames=np.asarray([r['source_frame'] for r in piece['metrics']]))
                records.append(dict(trajectory=entry['trajectory'],video=entry['video'],skill=segment['skill'],
                    split=entry['split'],geometry_group=entry['geometry_group'],frames=hi-lo,
                    start=lo,stop=hi,artifact=name,sha256=digest(folder/name)))
                if entry['split']=='train': train.append(inputs)
    train_groups={r['geometry_group'] for r in records if r['split']=='train'}
    val_groups={r['geometry_group'] for r in records if r['split']=='validation'}
    if train_groups & val_groups: raise ValueError('A scene group crossed the split')
    write(folder/'normalization.json',fit_normalization(train))
    index=dict(feature_dim=FEATURE_DIM,action_dim=30,records=records,normalization='normalization.json',
        normalization_sha256=digest(folder/'normalization.json'),source_manifest_sha256=digest(build/'manifest.json'),
        feature_slices=dict(native_observation=[0,39],qvel=[39,75],object_rotation=[75,81],source_phase=[81,82],
            video_id=[82,84],skill_id=[84,88],local_phase=[88,89],commanded_goal=[89,92],frozen_reference_action=[92,122]),
        validation_scope='Internal scene-group validation from the same two videos; not an independent heldout set',
        padding='Repeat final frame; mask=False; loss must ignore padded positions',
        reference_scope='Per-video frozen original reference, not each trajectory target action')
    write(folder/'index.json',index)
    checks = {}
    for split in ('train','validation'):
        dataset = SkillSequenceDataset(folder/'index.json',split)
        batch = dataset.sample_batch(32,config['sequence_length'],config['seed'])
        checks[split] = dict(frames=len(dataset),segments=len(dataset.entries),
            strata=len(dataset.groups),input_shape=list(batch['inputs'].shape),finite=bool(np.isfinite(batch['inputs']).all()),
            residual_reconstruction_error=float(np.max(np.abs(batch['reference_actions']+
                batch['residual_targets']-batch['actions']))),valid_sample_frames=int(batch['mask'].sum()))
    # Exercise the actual Torch interface without starting a new skill-training run.
    import torch
    torch.manual_seed(config['seed'])
    net=torch.nn.Linear(FEATURE_DIM,30)
    prediction=net(torch.from_numpy(batch['inputs']))
    mask=torch.from_numpy(batch['mask']).float()
    loss=(((prediction-torch.from_numpy(batch['residual_targets']).float())**2)*mask).sum()/(mask.sum()*30)
    passed=all(c['finite'] and c['strata']==8 and c['residual_reconstruction_error']<1e-7 for c in checks.values())
    passed=bool(passed and torch.isfinite(loss))
    write(folder/'receipt.json',dict(passed=passed,index_sha256=digest(folder/'index.json'),
        checks=checks,torch_forward_loss=float(loss.detach()),optimizer_steps=0,
        train_validation_groups_disjoint=True,normalization_fitted_on_train_only=True))
    print(json.dumps(read(folder/'receipt.json')),flush=True)
    if not passed: raise SystemExit(1)


if __name__=='__main__': main()
