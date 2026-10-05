#!/usr/bin/env python3
"""Guarded visual BC entrypoint. Default: forward checks only, no optimizer."""
import argparse
import hashlib
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from hierarchy_common import ROOT,read,write
from fromrealhand.skill_inputs import SkillSequenceDataset
from fromrealhand.tabletop.training_inputs import FEATURE_DIM,FEATURE_VERSION
from fromrealhand.policy_learning import model_to_device
from fromrealhand.routed_residual import RoutedNetwork
from mjrl.policies.gaussian_mlp import MLP


def tensors(batch,device,residual):
    x=torch.as_tensor(batch['inputs'].reshape(-1,FEATURE_DIM),dtype=torch.float32,device=device)
    target='residual_actions' if residual else 'actions'
    y=torch.as_tensor(batch[target].reshape(-1,30),dtype=torch.float32,device=device)
    video=np.asarray([0 if r['video']=='first' else 1 for r in batch['records']])[:,None]
    route=np.eye(8)[(video*4+batch['phases']).ravel()]
    return x,y,torch.as_tensor(route,dtype=torch.float32,device=device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/tabletop-dual-v3-pretraining.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['shared','routed-residual'],default='shared')
    p.add_argument('--start-training',action='store_true')
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--seed',type=int,default=0)
    a=p.parse_args(); cfg=read(a.config)
    if a.start_training and not cfg['training_enabled']:
        raise ValueError('Training remains locked by project configuration; explicit approval is required')
    if a.epochs<1: raise ValueError('Positive epochs required')
    index=Path(cfg['dataset']); meta=read(index); readiness=read(index.parent/'readiness.json')
    if not readiness['ready_for_small_scale_generalization_training'] or meta['feature_version']!=FEATURE_VERSION:
        raise ValueError('Dataset not admitted or feature version mismatch')
    train=SkillSequenceDataset(index,'train'); validation=SkillSequenceDataset(index,'validation')
    policy=MLP(SimpleNamespace(observation_dim=FEATURE_DIM,action_dim=30),hidden_sizes=tuple(cfg['hidden_sizes']),seed=a.seed)
    residual=a.mode=='routed-residual'; network=RoutedNetwork(policy.model,8) if residual else policy.model
    device='cuda' if a.start_training else 'cpu'
    if a.start_training and not torch.cuda.is_available(): raise RuntimeError('GPU training requested but unavailable')
    if residual: network.to_runtime(device)
    else: model_to_device(network,device)
    forward=lambda x,r: network(x,r) if residual else network(x)
    batch=validation.sample_batch(16,1,a.seed); x,y,r=tensors(batch,device,residual)
    with torch.no_grad(): predicted=forward(x,r)
    if predicted.shape!=y.shape or not torch.isfinite(predicted).all(): raise ValueError('Invalid policy forward')
    a.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(mode=a.mode,feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,
        parameters=sum(v.numel() for v in network.parameters()),device=device,
        train_frames=len(train),validation_frames=len(validation),forward_shape=list(predicted.shape),
        index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),normalization_sha256=meta['normalization_sha256'],
        training_started=False,optimizer_created=False,backward_called=False,checkpoint_saved=False,
        independent_test_executed=False,physical_policy_success_not_tested=True)
    write(a.output/'receipt.json',receipt)
    if not a.start_training:
        print(receipt); return
    optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    receipt.update(training_started=True,optimizer_created=True)
    write(a.output/'receipt.json',receipt)
    started=time.monotonic()
    losses=[]; steps=math.ceil(len(train)/128)
    for epoch in range(a.epochs):
        network.train(); total=0.
        for iteration in range(steps):
            batch=train.sample_batch(128,1,a.seed+epoch*steps+iteration)
            x,y,r=tensors(batch,device,residual); optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(forward(x,r),y)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); optimizer.step(); total+=float(loss.detach().cpu())
        network.eval(); x,y,r=tensors(validation.sample_batch(512,1,0),device,residual)
        with torch.no_grad(): val=float(torch.mean((forward(x,r)-y)**2).cpu())
        losses.append(dict(epoch=epoch+1,train_mse=total/steps,development_validation_mse=val))
        write(a.output/'losses.json',losses); print(losses[-1],flush=True)
    state={k:v.detach().cpu() for k,v in network.state_dict().items()}
    torch.save(dict(state_dict=state,mode=a.mode,feature_version=FEATURE_VERSION,feature_dim=FEATURE_DIM,
        hidden_sizes=cfg['hidden_sizes'],normalization=read(index.parent/meta['normalization']),
        index_sha256=receipt['index_sha256'],physical_policy_success_not_tested=True),str(a.output/'candidate.pt'))
    receipt.update(backward_called=True,checkpoint_saved=True,completed_epochs=a.epochs,
        training_wall_s=time.monotonic()-started,optimizer_steps=a.epochs*steps)
    write(a.output/'receipt.json',receipt)


if __name__=='__main__': main()
