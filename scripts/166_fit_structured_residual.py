#!/usr/bin/env python3
"""Fit a small physically structured BC residual, using train episodes only."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from hierarchy_common import ROOT,read,write
from fromrealhand.skill_inputs import SkillSequenceDataset
from fromrealhand.tabletop.structured_control import fit_diagonal_residual
from fromrealhand.tabletop.training_inputs import FEATURE_DIM,FEATURE_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/tabletop-dual-v4-training.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); cfg=read(a.config)
    if not cfg['training_enabled'] or not torch.cuda.is_available():
        raise ValueError('Authorized GPU training required')
    index=Path(cfg['dataset']); meta=read(index)
    if (meta['feature_version']!=FEATURE_VERSION
            or not read(index.parent/'readiness.json')['ready_for_small_scale_generalization_training']):
        raise ValueError('Unadmitted dataset')
    train=SkillSequenceDataset(index,'train'); validation=SkillSequenceDataset(index,'validation')
    a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    x=np.concatenate([v['inputs'] for v in train.arrays])
    y=np.concatenate([v['residual_actions'] for v in train.arrays])
    gain,fit=fit_diagonal_residual(x,y)
    vx=np.concatenate([v['inputs'] for v in validation.arrays])
    vy=np.concatenate([v['residual_actions'] for v in validation.arrays])
    error=vx[:,109:139]*gain-vy
    saved=dict(mode='structured-residual',feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,
        gain=gain.tolist(),normalization=read(index.parent/meta['normalization']),
        index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        training_method='diagonal least-squares BC on raw joint-error features',
        physical_policy_success_not_tested=True)
    torch.save(saved,str(a.output/'candidate.pt'))
    receipt=dict(training_started=True,checkpoint_saved=True,device='cuda',
        train_frames=len(train),validation_frames=len(validation),
        train=fit,development_validation_mse=float(np.mean(error**2)),
        development_validation_max_error=float(np.max(np.abs(error))),
        gains=gain.tolist(),parameters=30,nonzero_parameters=int(np.count_nonzero(gain)),
        training_wall_s=time.monotonic()-started,
        independent_test_used_for_fit=False,simulator_gains_used_for_fit=False,
        note='No learned constant action or object truth; joint errors change online.')
    write(a.output/'receipt.json',receipt); print(receipt)


if __name__=='__main__': main()
