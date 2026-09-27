#!/usr/bin/env python3
"""Optimize video control under explicit near-surface contact requirements."""
import argparse
import hashlib
import json
import pickle
from importlib import import_module
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
video = import_module('30_validate_video_faithful')


def surface_gate(report):
    return bool(report['physics_passed']
                and report.get('max_hand_scene_penetration_m', float('inf')) <= .001
                and report.get('initial_hand_scene_penetration_m', float('inf')) <= .0005
                and report['max_loaded_gap_m'] <= .0005
                and report['max_penetration_m'] <= .001
                and report['max_joint_violation_rad'] <= .02
                and report['tail_slip_m'] <= .005
                and report['saturation'] < .01
                and report['finite']
                and report['tail_finger_contact_fraction']['thumb'] >= .8
                and report['tail_finger_contact_fraction']['index'] >= .8)


class SurfaceExperiment(video.VideoExperiment):
    def __init__(self, geometry, margin=.0002):
        super().__init__(geometry)
        self.margin = margin
        self.audit = []
        self.contact_peaks = {}
        original = self.env._pre_action

        def audited(action, policy_step=False):
            self.sample()
            return original(action, policy_step)

        self.env._pre_action = audited

    def sample(self):
        e, m, d = self.env, self.model, self.env.sim.data
        cs = video.contact_details(e)
        scene_depth = max([max(0., -float(c.dist)) for c in d.contact[:d.ncon]
                           if m.geom_id2name(int(c.geom1)) in self.hand_geoms
                           or m.geom_id2name(int(c.geom2)) in self.hand_geoms] or [0.])
        loaded = [c['distance_m'] for c in cs if c['normal_force_n'] > .01]
        for c in d.contact[:d.ncon]:
            pair = sorted(m.geom_id2name(int(i)) or str(int(i)) for i in (c.geom1, c.geom2))
            if not self.hand_geoms.intersection(pair):
                continue
            key = '|'.join(pair)
            depth = max(0., -float(c.dist))
            if depth > self.contact_peaks.get(key, {}).get('penetration_m', -1.):
                self.contact_peaks[key] = dict(geoms=pair, time_s=float(d.time), penetration_m=depth)
        violation = np.maximum(m.jnt_range[:30, 0]-d.qpos[:30], d.qpos[:30]-m.jnt_range[:30, 1])
        palm = m.body_name2id('palm')
        relative = (d.body_xpos[e.obj_bid]-d.body_xpos[palm]) @ d.body_xmat[palm].reshape(3, 3)
        self.audit.append(dict(time=float(d.time), gap=max(loaded or [0.]),
                               scene_penetration=scene_depth,
                               penetration=max([max(0., -c['distance_m']) for c in cs] or [0.]),
                               joint=max(0., float(violation.max())), relative=relative.tolist(),
                               finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())))

    def run_surface(self, scale, close, gain, seed=0, saved_actions=None, output=None):
        # Reset only these contact settings; retain original meshes, mass and friction.
        m = self.model
        ids = [i for i in range(m.ngeom) if m.geom_id2name(i) in self.hand_geoms | self.mug_geoms]
        m.geom_margin[ids] = self.margin
        m.geom_gap[ids] = 0.
        self.audit = []
        self.contact_peaks = {}
        report, actions = self.run_video(scale, close, seed=seed, saved_actions=saved_actions, cartesian_gain=gain)
        self.sample()
        tail = [x for x in self.audit if x['time'] >= self.audit[-1]['time']-1.]
        relative = np.array([x['relative'] for x in tail])
        report.update(max_loaded_gap_m=max(x['gap'] for x in self.audit),
                      max_hand_scene_penetration_m=max(x['scene_penetration'] for x in self.audit),
                      initial_hand_scene_penetration_m=self.audit[0]['scene_penetration'],
                      max_penetration_m=max(x['penetration'] for x in self.audit),
                      max_joint_violation_rad=max(x['joint'] for x in self.audit),
                      tail_slip_m=float(np.linalg.norm(relative-relative[0], axis=1).max()),
                      finite=all(x['finite'] for x in self.audit), margin_m=self.margin)
        report['surface_physics_passed'] = surface_gate(report)
        report['scene_contact_peaks'] = sorted(self.contact_peaks.values(), key=lambda x: -x['penetration_m'])
        self.last_demo['physics_model'] = {key: getattr(m, key).copy() for key in ('geom_margin', 'geom_gap')}
        if output is not None:
            output.mkdir(parents=True, exist_ok=False)
            (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            (output/'substeps.json').write_text(json.dumps(self.audit)+'\n')
            with (output/'diagnostic_rollout.pkl').open('xb') as f:
                pickle.dump({'video_faithful': self.last_demo}, f)
        return report, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, default=ROOT/'data/processed/seq_dexycb_001/video_faithful_v1/retarget_control/geometry.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--closures', type=float, nargs='+', default=[.1, .15, .2])
    parser.add_argument('--gains', type=float, nargs='+', default=[80., 160., 240.])
    parser.add_argument('--scale', type=float, default=4.)
    parser.add_argument('--initial-correction', type=Path)
    parser.add_argument('--search-seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--quota-stop', type=float, default=85.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    exp = SurfaceExperiment(args.geometry)
    if args.initial_correction:
        exp.env.close()
        exp = import_module('37_optimize_finger_reference').CorrectedExperiment(args.geometry)
        exp.correction = np.asarray(json.loads(args.initial_correction.read_text())['joint_correction'])
    results = []
    stop_search = False
    from check_codex_budget import latest_usage
    try:
        for close in args.closures:
            for gain in args.gains:
                quota = latest_usage()
                if results and quota and quota['used_percent'] >= args.quota_stop:
                    stop_search = True
                    break
                report, _ = exp.run_surface(args.scale, close, gain)
                probes = [report] + [exp.run_surface(args.scale, close, gain, seed=seed)[0] for seed in args.search_seeds if seed != 0]
                report['search_score'] = max(import_module('37_optimize_finger_reference').score(r) for r in probes)
                report['search_seeds'] = sorted(set([0]+args.search_seeds))
                results.append(report)
                (args.output/'search.json').write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps(report), flush=True)
            if stop_search:
                break
        best = min(results, key=lambda r: r['search_score'])
        close, gain = best['close'], best['cartesian_gain']
        best, actions = exp.run_surface(args.scale, close, gain, output=args.output/'best')
        original = exp.last_demo['observations'].copy()
        replay, _ = exp.run_surface(args.scale, close, gain, saved_actions=actions, output=args.output/'replay')
        error = float(np.max(np.abs(original-exp.last_demo['observations'])))
        holdouts = [exp.run_surface(args.scale, close, gain, seed=seed, output=args.output/('seed_%d'%seed))[0] for seed in range(1, 6)]
        manifest = json.loads((args.geometry.parent/'manifest.json').read_text())
        unchanged = all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest for name, digest in manifest['protected_files'].items())
        passed = unchanged and error < 1e-8 and all(r['surface_physics_passed'] for r in [best, replay]+holdouts)
        result = dict(surface_physics_passed=bool(passed), training_ready=bool(passed and not manifest['failed_optimizer_frames'] and all(r['fidelity_passed'] for r in [best]+holdouts)),
                      scene_audit_version=2, quota=latest_usage(),
                      stop_reason='quota_reserve' if stop_search else 'search_complete',
                      search_seeds=sorted(set([0]+args.search_seeds)),
                      best=best, replay=replay, replay_observation_error=error, holdouts=holdouts,
                      baseline_unchanged=unchanged, geometry=str(args.geometry.resolve()),
                      scope='Native mesh collision physics; +/-2mm XY initial perturbations; not hardware validation')
        (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
        print('ADMISSION', json.dumps(result), flush=True)
    finally:
        exp.env.close()


if __name__ == '__main__':
    main()
