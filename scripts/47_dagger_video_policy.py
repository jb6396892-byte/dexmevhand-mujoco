#!/usr/bin/env python3
"""Aggregate online expert labels; evaluate each student without expert control."""
import argparse
import json
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

finger = import_module('37_optimize_finger_reference')
from fromrealhand.policy_learning import policy_features, model_to_device, StudentActions, lift_success
from check_codex_budget import latest_usage


class LabelExperiment(finger.CorrectedExperiment):
    selector = None

    def run_video(self, *args, **kwargs):
        kwargs['action_selector'] = self.selector
        return super().run_video(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--betas', type=float, nargs='+', default=[.8, .5, .2, 0.])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--quota-stop', type=float, default=85.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.checkpoint.open('rb') as stream:
        checkpoint = pickle.load(stream)
    source = json.loads(Path(checkpoint['source_candidate']).read_text())
    b = source['best']
    with (finger.surface.ROOT/'data/demonstrations/relocate-mug-video-faithful-v3.pkl').open('rb') as stream:
        demos = list(pickle.load(stream).values())
    exp = LabelExperiment(source['geometry'])
    exp.correction = np.asarray(b['joint_correction'])
    exp.closure_lead = b.get('closure_lead', 0.)
    exp.feedback_weights = b.get('feedback_weights')
    exp.approach_gain = b.get('approach_gain', 0.)
    exp.approach_root_gain = b.get('approach_root_gain')
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    observations = [policy_features(obs, state['qpos'], state['qvel'], i, exp.env.control_timestep,
                                   exp.duration, b['time_scale'], checkpoint['feature_mode'])
                    for demo in demos for i, (obs, state) in enumerate(zip(demo['observations'], demo['sim_data']))]
    reference = checkpoint.get('action_reference', np.zeros_like(demos[0]['actions']))
    labels = list(np.concatenate([demo['actions']-reference for demo in demos]))
    policy = checkpoint['policy']
    rng = np.random.RandomState(300)
    rounds = []
    try:
        for iteration, beta in enumerate(args.betas, 1):
            quota = latest_usage()
            if quota and quota['used_percent'] >= args.quota_stop:
                break
            actions = StudentActions(checkpoint, exp, len(demos[0]['actions']))
            new_observations, new_labels = [], []

            def select(step, observation, expert_action):
                new_observations.append(actions.features(step))
                new_labels.append(expert_action-reference[step])
                return np.clip(beta*expert_action+(1.-beta)*actions[step], -1., 1.)

            exp.selector = select
            collection = [exp.run_surface(**kwargs, seed=seed)[0] for seed in (0, 2, 5)]
            exp.selector = None
            observations.extend(new_observations)
            labels.extend(new_labels)
            np.savez_compressed(args.output/('labels_%d.npz'%iteration), observations=new_observations, actions=new_labels)
            x = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device='cuda')
            y = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device='cuda')
            model_to_device(policy.model, 'cuda')
            optimizer = torch.optim.Adam(policy.model.parameters(), lr=.0003)
            epochs_done = 0
            for epoch in range(args.epochs):
                quota = latest_usage()
                if quota and quota['used_percent'] >= args.quota_stop:
                    break
                ids = rng.permutation(len(observations))
                for start in range(0, len(ids), 128):
                    batch = torch.as_tensor(ids[start:start+128], device='cuda')
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(policy.model(x[batch]), y[batch])
                    loss.backward()
                    optimizer.step()
                epochs_done += 1
            with torch.no_grad():
                mse = float(torch.mean((policy.model(x)-y)**2).cpu())
            model_to_device(policy.model, 'cpu')
            policy.set_param_values(policy.get_param_values())
            checkpoint['dagger_round'] = iteration
            path = args.output/('policy_dagger_%d.pickle'%iteration)
            with path.open('xb') as stream:
                pickle.dump(checkpoint, stream)
            reports = [exp.run_surface(**kwargs, seed=seed, saved_actions=actions)[0] for seed in (0, 2, 5)]
            result = dict(iteration=iteration, beta=beta, epochs=epochs_done, sample_count=len(observations),
                          aggregated_action_rmse=float(np.sqrt(mse)), policy=str(path.resolve()), collection=collection, reports=reports,
                          lift_count=sum(lift_success(r) for r in reports), physical_count=sum(r['surface_physics_passed'] for r in reports),
                          full_count=sum(r['surface_physics_passed'] and r['fidelity_passed'] for r in reports), quota=latest_usage())
            rounds.append(result)
            (args.output/'dagger.json').write_text(json.dumps(dict(rounds=rounds, source_checkpoint=str(args.checkpoint.resolve()),
                scope='Expert-labeled learner states are supervised correction data, not admitted successful demonstrations'), indent=2)+'\n')
            print(json.dumps({k: result[k] for k in ('iteration', 'beta', 'epochs', 'sample_count', 'aggregated_action_rmse',
                  'lift_count', 'physical_count', 'full_count', 'quota')}), flush=True)
            del x, y
    finally:
        model_to_device(policy.model, 'cpu')
        exp.env.close()


if __name__ == '__main__':
    main()
