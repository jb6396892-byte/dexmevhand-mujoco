#!/usr/bin/env python3
"""GPU behavior-cloning ablation with CPU closed-loop physics at checkpoints."""
import argparse
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

surface = import_module('33_optimize_surface_grasp')
from fromrealhand.policy_learning import policy_features, model_to_device, StudentActions, lift_success
from mjrl.policies.gaussian_mlp import MLP
from check_codex_budget import latest_usage


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--features', choices=['original', 'state', 'phase'], default='original')
    p.add_argument('--epochs', type=int, nargs='+', default=[5, 25, 100, 300])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 2, 5])
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--scale-floor', type=float, default=.001)
    p.add_argument('--residual', action='store_true', help='Learn corrections to a fixed nominal action trajectory')
    p.add_argument('--quota-stop', type=float, default=85.)
    p.add_argument('--candidate', type=Path, default=surface.ROOT/'data/processed/seq_dexycb_001/scene_fidelity_v3/approach_feedback/admission.json')
    p.add_argument('--demo', type=Path, default=surface.ROOT/'data/demonstrations/relocate-mug-video-faithful-v3.pkl')
    p.add_argument('--admission', type=Path, default=surface.ROOT/'data/processed/seq_dexycb_001/scene_fidelity_v3/verified_export/admission.json')
    args = p.parse_args()
    source = json.loads(args.candidate.read_text())
    admission = json.loads(args.admission.read_text())
    raw = args.demo.read_bytes()
    if not admission.get('training_ready') or hashlib.sha256(raw).hexdigest() != admission['demo_sha256']:
        raise ValueError('Demonstration admission or hash failed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    args.output.mkdir(parents=True, exist_ok=False)
    demos = list(pickle.loads(raw).values())
    exp = surface.SurfaceExperiment(source['geometry'])
    time_scale = source['best']['time_scale']
    x = np.stack([policy_features(obs, state['qpos'], state['qvel'], i, exp.env.control_timestep,
                                  exp.duration, time_scale, args.features)
                  for demo in demos for i, (obs, state) in enumerate(zip(demo['observations'], demo['sim_data']))])
    y = np.concatenate([demo['actions'] for demo in demos])
    reference = np.asarray(demos[0]['actions'])
    if args.residual:
        y = y-np.tile(reference, (len(demos), 1))
    policy = MLP(SimpleNamespace(observation_dim=x.shape[1], action_dim=30), hidden_sizes=(64, 64), seed=200, init_log_std=-2.)
    shift, scale = x.mean(0), np.maximum(x.std(0), args.scale_floor)
    output_shift, output_scale = y.mean(0), np.maximum(y.std(0), .001)
    for model in (policy.model, policy.old_model):
        model.set_transformations(shift, scale, output_shift, output_scale)
    model_to_device(policy.model, 'cuda')
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=.001)
    features = torch.as_tensor(x, dtype=torch.float32, device='cuda')
    targets = torch.as_tensor(y, dtype=torch.float32, device='cuda')
    checkpoints, losses = [], []
    rng = np.random.RandomState(200)
    stopped = False
    try:
        for epoch in range(1, max(args.epochs)+1):
            quota = latest_usage()
            if quota and quota['used_percent'] >= args.quota_stop:
                stopped = True
                break
            order = rng.permutation(len(x))
            for start in range(0, len(order), args.batch_size):
                ids = torch.as_tensor(order[start:start+args.batch_size], device='cuda')
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(policy.model(features[ids]), targets[ids])
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                mse = float(torch.mean((policy.model(features)-targets)**2).cpu())
            losses.append(dict(epoch=epoch, action_mse=mse))
            (args.output/'losses.json').write_text(json.dumps(losses, indent=2)+'\n')
            if epoch not in args.epochs:
                continue
            model_to_device(policy.model, 'cpu')
            policy.set_param_values(policy.get_param_values())
            checkpoint = dict(policy=policy, feature_mode=args.features, time_scale=time_scale, duration=exp.duration,
                              dt=exp.env.control_timestep, epoch=epoch, source_candidate=str(args.candidate.resolve()))
            if args.residual:
                checkpoint['action_reference'] = reference.copy()
            path = args.output/('policy_bc_%d.pickle'%epoch)
            with path.open('xb') as stream:
                pickle.dump(checkpoint, stream)
            reports = []
            for seed in args.seeds:
                actions = StudentActions(checkpoint, exp, len(demos[0]['actions']))
                report, _ = exp.run_surface(time_scale, 0., 0., seed=seed, saved_actions=actions)
                reports.append(report)
            summary = dict(epoch=epoch, offline_action_rmse=float(np.sqrt(mse)), policy=str(path.resolve()), reports=reports,
                           lift_count=sum(lift_success(r) for r in reports),
                           physical_count=sum(r['surface_physics_passed'] for r in reports),
                           full_count=sum(r['surface_physics_passed'] and r['fidelity_passed'] for r in reports), quota=latest_usage())
            checkpoints.append(summary)
            (args.output/'comparison.json').write_text(json.dumps(dict(feature_mode=args.features, observation_dim=x.shape[1],
                source_demo_sha256=admission['demo_sha256'], normalization_floor=args.scale_floor, residual=args.residual,
                batch_size=args.batch_size, device='cuda', checkpoints=checkpoints), indent=2)+'\n')
            print(json.dumps({k: summary[k] for k in ('epoch', 'offline_action_rmse', 'lift_count', 'physical_count', 'full_count', 'quota')}), flush=True)
            model_to_device(policy.model, 'cuda')
        (args.output/'completion.json').write_text(json.dumps(dict(completed_epochs=len(losses), quota=latest_usage(),
            stopped_for_quota=stopped, training_type='supervised behavior cloning; no RL'))+'\n')
    finally:
        model_to_device(policy.model, 'cpu')
        exp.env.close()


if __name__ == '__main__':
    main()
