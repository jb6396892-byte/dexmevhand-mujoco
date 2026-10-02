#!/usr/bin/env python3
"""Controlled routed-residual development comparison; never reads old holdouts."""
import argparse
import copy
import importlib
import json
import pickle
from types import SimpleNamespace
import numpy as np
import torch
from v10_common import ROOT, digest, run_case
from fromrealhand.multivideo import phase_diagnostics
from fromrealhand.routed_residual import RoutedNetwork, phase_routes, student_actions
from mjrl.policies.gaussian_mlp import MLP

study = importlib.import_module('77_run_v12_study')
PLAN = ROOT/'configs/v13-study.json'
RUN = ROOT/'data/processed/dual_video_v13'
write_json = study.write_json


def inputs():
    plan = json.loads(PLAN.read_text())
    old, parent, admission, demos, refs, base = study.load_inputs()
    data = np.load(study.RUN/'aligned_bc/input.npz')
    return plan, old, parent, admission, demos, refs, base, data


def routes_for(mode, parent, plan):
    return {v['name']: (np.tile(np.eye(2)[v['id']], (v['horizon'], 1)) if mode == 'video' else
            phase_routes(v['horizon'], np.load(ROOT/v['paths']['geometry']),
                         v['control']['time_scale'], plan['phase_blend_half_width_steps']))
            for v in parent['videos']}


def batch_routes(checkpoint, admission):
    return np.concatenate([checkpoint['routes'][e['video']] for e in admission['reports']])


def fit(network, x, y, w, routes, epochs, seed, lr, folder):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this controlled BC study')
    torch.set_num_threads(1)
    network.to_runtime('cuda')
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    x, y, routes = [torch.as_tensor(a, dtype=torch.float32, device='cuda') for a in (x, y, routes)]
    weight = torch.as_tensor(w, dtype=torch.float32, device='cuda')
    rng = np.random.RandomState(seed)
    losses = []
    for epoch in range(1, epochs+1):
        ids = rng.choice(len(x), size=len(x), replace=True, p=w/w.sum())
        for start in range(0, len(ids), 128):
            j = torch.as_tensor(ids[start:start+128], device='cuda')
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(network(x[j], routes[j]), y[j])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mse = float((((network(x, routes)-y)**2).mean(1)*weight).sum().cpu())
        losses.append(dict(epoch=epoch, weighted_mse=mse))
        if not np.isfinite(mse):
            raise RuntimeError('Nonfinite training loss')
        if epoch % 10 == 0:
            print('TRAIN', folder.name, epoch, float(np.sqrt(mse)), flush=True)
    network.to_runtime('cpu')
    write_json(folder/'losses.json', losses)


def execute(checkpoint, video, entry, half=False, actions=None):
    return run_case(video, dict(geometry=entry['geometry']), actions, seed=entry['seed'], half=half,
                    action_factory=(lambda exp: student_actions(checkpoint, exp, video)) if actions is None else None)


def train(mode):
    plan, old, parent, admission, demos, refs, base, data = inputs()
    folder = RUN/mode
    folder.mkdir(parents=True, exist_ok=False)
    x, y, w = [data[k] for k in ('features', 'labels', 'weights')]
    policy = MLP(SimpleNamespace(observation_dim=84, action_dim=30),
                 hidden_sizes=tuple(plan['hidden_sizes']), seed=plan['seed'], init_log_std=-2.)
    baseline = pickle.loads((ROOT/plan['baseline']).read_bytes())
    model = baseline['policy'].model
    policy.model.set_transformations(*[getattr(model, k).numpy() for k in ('in_shift', 'in_scale', 'out_shift', 'out_scale')])
    network = RoutedNetwork(policy.model, 2 if mode == 'video' else 5)
    checkpoint = {k: copy.deepcopy(baseline[k]) for k in ('method', 'references', 'clocks', 'residual_limits')}
    checkpoint.update(routing_mode=mode, network=network, routes=routes_for(mode, parent, plan),
                      protocol_sha256=digest(PLAN), training_label='v13_'+mode)
    routes = batch_routes(checkpoint, admission)
    if len(routes) != len(x):
        raise ValueError('Routing/data length mismatch')
    write_json(folder/'input.json', dict(protocol_sha256=digest(PLAN), input_sha256=digest(study.RUN/'aligned_bc/input.npz'),
        frames=len(x), sampled_frames=len(x)*plan['epochs'], batch_size=128, seed=plan['seed'],
        parameter_count=sum(p.numel() for p in network.parameters()),
        routing_mode=mode, capacity_matched=False, heldout_used=False,
        route_mass=(routes*w[:, None]).sum(0).tolist(), gpu=torch.cuda.get_device_name(0)))
    fit(network, x, y, w, routes, plan['epochs'], plan['seed'], plan['learning_rate'], folder/'training')
    with (folder/'policy.pickle').open('xb') as stream:
        pickle.dump(checkpoint, stream)
    print('TRAINED', mode, digest(folder/'policy.pickle'), flush=True)


def develop(mode):
    plan, old, parent, admission, demos, refs, base, data = inputs()
    path = ROOT/plan['baseline'] if mode == 'shared' else RUN/mode/'policy.pickle'
    checkpoint = pickle.loads(path.read_bytes())
    folder = RUN/mode
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder/'development.json'
    previous = json.loads(dest.read_text()) if dest.exists() else None
    if previous and (previous['policy_sha256'] != digest(path) or previous['protocol_sha256'] != digest(PLAN)):
        raise ValueError('Changed candidate/protocol')
    rows = previous['reports'] if previous else []
    for entry in admission['reports']:
        video = parent['videos'][entry['video_id']]
        key = video['name']+'/'+entry['name']
        if any(r['video'] == video['name'] and r['case'] == entry['name'] for r in rows):
            continue
        report, demo = execute(checkpoint, video, entry)
        row = dict(video=video['name'], case=entry['name'], report=report, **study.gates(report, old))
        row['phases'] = phase_diagnostics(demo, refs[video['name']]['actions'], video, np.load(entry['geometry']), demos[key])
        if video['id'] == 1 or entry['name'] == 'video_seed_0':
            output = folder/'rollouts'/video['name']/(entry['name']+'.pkl')
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('xb') as stream:
                pickle.dump({'video_faithful': demo}, stream)
        rows.append(row)
        write_json(dest, dict(protocol_sha256=digest(PLAN), policy_sha256=digest(path), policy=str(path),
                             reports=rows, summary=study.summarize(rows), heldout_used=False))
        print('DEV', mode, video['name'], entry['name'], row['task_pass'], row['strict_pass'],
              round(report['final_distance_m']*1000, 3), round(report['max_hand_scene_penetration_m']*1000, 3), flush=True)
    print('SUMMARY', mode, json.dumps(study.summarize(rows)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'develop', 'compare'])
    parser.add_argument('--mode', choices=['shared', 'video', 'phase', 'base_finetune', 'corrected'])
    args = parser.parse_args()
    if args.command == 'train':
        if args.mode not in ('video', 'phase'):
            parser.error('Train video or phase only')
        train(args.mode)
    elif args.command == 'develop':
        if args.mode is None:
            parser.error('A mode is required')
        develop(args.mode)
    else:
        rows = []
        for mode in ('shared', 'video', 'phase'):
            r = json.loads((RUN/mode/'development.json').read_text())
            if r['summary']['count'] != 35 or r['policy_sha256'] != digest(r['policy']):
                raise ValueError('Incomplete/changed candidate')
            rows.append(dict(mode=mode, path=r['policy'], sha256=r['policy_sha256'], **r['summary']))
        selected = min(rows, key=lambda r: (-r['task_fraction'], -r['strict_fraction'], r['mean_goal_m']))
        write_json(RUN/'comparison.json', dict(protocol_sha256=digest(PLAN), candidates=rows,
                   selected=selected, heldout_used=False, capacity_matched=False))
        print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
