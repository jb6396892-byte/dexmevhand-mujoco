#!/usr/bin/env python3
"""Preflight by default; --train starts only the admitted 20-iteration curriculum."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.paths import configure_runtime_paths
configure_runtime_paths()
from fromrealhand.verified_curriculum import load_admitted_demos, install_verified_factory, make_verified_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--cfg', type=Path, default=ROOT/'configs/dapg-mug-physics-verified-smoke.yaml')
    parser.add_argument('--admission', type=Path, default=ROOT/'data/processed/seq_dexycb_001/physical_grasp_verified_v1/admission.json')
    args = parser.parse_args()
    from tpi.core.config import cfg, assert_cfg
    cfg.merge_from_file(str(args.cfg))
    assert_cfg()
    if cfg.NUM_ITER != 20 or cfg.NUM_CPU != 1:
        raise ValueError('This initial curriculum entry requires NUM_ITER=20 and NUM_CPU=1')
    os.environ['FROMREALHAND_ADMISSION'] = str(args.admission.resolve())
    os.environ['FROMREALHAND_VERIFIED_DEMO'] = cfg.DEMO_FILE
    demos, admission = load_admitted_demos()
    install_verified_factory()
    environment = make_verified_environment(cfg.ENV_NAME)
    reset_errors = []
    for seed in range(6):
        environment.seed(seed)
        obs = environment.reset()
        expected = environment.env.demonstrations[environment.env.demo_index]['observations'][0]
        reset_errors.append(float(np.max(np.abs(obs-expected))))
    if max(reset_errors) > 1e-8:
        raise RuntimeError('Curriculum reset observations differ from verified demonstrations')
    from mjrl.samplers.base_sampler import do_rollout

    class RecordedActions:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.index = 0

        def get_action(self, observation):
            demo = self.wrapped.env.demonstrations[self.wrapped.env.demo_index]
            action = demo['actions'][self.index]
            self.index += 1
            return action, {'evaluation': action.copy()}

    sampled = do_rollout(1, RecordedActions(environment), env=environment, pegasus_seed=200)[0]
    expected = environment.env.demonstrations[environment.env.demo_index]
    sampling_error = float(np.max(np.abs(sampled['observations']-expected['observations'])))
    if len(sampled['actions']) != 1000 or sampling_error > 1e-8:
        raise RuntimeError('Training sampler does not reproduce the admitted action trajectory')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to claim GPU training readiness')
    x = torch.randn(16, 16, device='cuda', requires_grad=True)
    (x @ x).sum().backward()
    torch.cuda.synchronize()
    result = dict(training_ready=True, trajectory_count=len(demos), horizon=environment.horizon,
                  reset_max_error=max(reset_errors), sampling_max_observation_error=sampling_error,
                  cuda_device=torch.cuda.get_device_name(0),
                  device_scope='MuJoCo and policy use CPU; upstream value baseline uses GPU',
                  curriculum=admission['scope'], training_started=bool(args.train))
    print(json.dumps(result, indent=2), flush=True)
    (args.admission.parent/'training_preflight.json').write_text(json.dumps(result, indent=2)+'\n')
    if not args.train:
        return
    prop = Path(cfg.DEMO_FILE).stem
    job = 'dapg_%s_%s_%s_%s_%s_seed%s' % (cfg.ENV_NAME, prop, cfg.DAPG_LAM0,
                                         cfg.DEMO_RATIO, cfg.JOB_NAME, cfg.RNG_SEED)
    if (Path(cfg.JOB_DIR)/job).exists():
        raise FileExistsError('Refusing to overwrite existing training directory: '+job)
    cfg.freeze()
    sys.path.insert(0, '/home/smgbro/dexmv-sim/examples')
    from train import train
    train()


if __name__ == '__main__':
    main()
