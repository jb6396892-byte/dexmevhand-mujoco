#!/usr/bin/env python3
"""Search bounded hand-root feedback, then verify held-out physics and replay."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from importlib import import_module
import numpy as np

finger = import_module('37_optimize_finger_reference')
from fromrealhand.transport_control import TransportCorrection
from fromrealhand.trajectory_control import world_to_root_delta
from fromrealhand.video_fidelity import source_clock
from check_codex_budget import latest_usage


class TransportExperiment(finger.CorrectedExperiment):
    transport_kp = 0.
    transport_ki = 0.
    max_shift = .08
    max_speed = .03

    def run_video(self, *args, **kwargs):
        controller = TransportCorrection(self.transport_kp, self.transport_ki, self.max_shift, self.max_speed)
        scales = args[0] if args else kwargs['time_scale']
        active_steps = 0

        def select(step, observation, action):
            nonlocal active_steps
            e, m, d = self.env, self.model, self.env.sim.data
            dt = e.control_timestep
            clock = float(source_clock(step*dt, self.duration, scales))
            contacts = finger.surface.video.contact_details(e)
            loaded = {c['hand'][2:4] for c in contacts if c['normal_force_n'] > .05}
            bottom = self.contact_stats()[3]
            ready = ('th' in loaded and 'ff' in loaded and bottom > .015 and clock > self.grasp_time+.35)
            error = self.pcurve(clock)-d.body_xpos[e.obj_bid]
            shift = controller.update(error, ready, dt)
            active_steps += int(np.linalg.norm(shift) > 1e-10)
            jac = d.get_body_jacp('palm').reshape(3, m.nv)[:, :3]
            delta = world_to_root_delta(jac, shift)
            reference = self.qcurve(clock)[:3]
            delta = np.clip(reference+delta, m.jnt_range[:3, 0], m.jnt_range[:3, 1])-reference
            kp = -m.actuator_biasprm[:3, 1]
            action[:3] += kp*delta/(m.actuator_gainprm[:3, 0]*e.act_rng[:3])
            return np.clip(action, -1., 1.)

        if kwargs.get('saved_actions') is None and (self.transport_kp or self.transport_ki):
            kwargs['action_selector'] = select
        report, actions = super().run_video(*args, **kwargs)
        report.update(transport_kp=self.transport_kp, transport_ki=self.transport_ki,
                      transport_max_shift_m=self.max_shift, transport_max_speed_m_s=self.max_speed,
                      transport_final_shift_m=controller.shift.tolist(), transport_active_steps=active_steps)
        return report, actions


def configure(e, b):
    e.correction = np.asarray(b['joint_correction'])
    e.closure_lead = b.get('closure_lead', 0.)
    e.feedback_weights = b.get('feedback_weights')
    e.approach_gain = b.get('approach_gain', 0.)
    e.approach_root_gain = b.get('approach_root_gain')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kp', type=float, nargs='+', default=[0., .5, 1., 2.])
    p.add_argument('--ki', type=float, nargs='+', default=[0., .5])
    p.add_argument('--quota-stop', type=float, default=85.)
    args = p.parse_args()
    source = json.loads(args.candidate.read_text())
    b = source['best']
    args.output.mkdir(parents=True, exist_ok=False)
    exp = TransportExperiment(source['geometry'])
    configure(exp, b)
    kwargs = dict(scale=b['time_scale'], close=b['close'], gain=b['cartesian_gain'])
    results = []
    try:
        for kp, ki in itertools.product(args.kp, args.ki):
            quota = latest_usage()
            if quota and quota['used_percent'] >= args.quota_stop:
                break
            exp.transport_kp, exp.transport_ki = kp, ki
            report, _ = exp.run_surface(**kwargs)
            results.append(report)
            (args.output/'search.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps({k:report[k] for k in ('transport_kp','transport_ki','hold_s','final_distance_m','surface_physics_passed','fidelity_passed','max_hand_scene_penetration_m')}), flush=True)
        if not results:
            return
        best = min(results, key=lambda r:(not r['surface_physics_passed'], r['final_distance_m']))
        exp.transport_kp, exp.transport_ki = best['transport_kp'], best['transport_ki']
        best, actions = exp.run_surface(**kwargs, output=args.output/'best')
        before = exp.last_demo['observations'].copy()
        replay, _ = exp.run_surface(**kwargs, saved_actions=actions, output=args.output/'replay')
        error = float(np.max(np.abs(before-exp.last_demo['observations'])))
        reports = []
        for seed in range(20, 25):
            quota = latest_usage()
            if quota and quota['used_percent'] >= args.quota_stop:
                break
            report, _ = exp.run_surface(**kwargs, seed=seed, output=args.output/('seed_%d'%seed))
            reports.append(report)
            (args.output/'progress.json').write_text(json.dumps(reports, indent=2)+'\n')
        half = []
        quota = latest_usage()
        if len(reports) == 5 and (quota is None or quota['used_percent'] < args.quota_stop):
            dt = exp.model.opt.timestep/2
            exp.model.opt.timestep = dt; exp.env.model_timestep = dt
            half = [exp.run_surface(**kwargs, saved_actions=actions)[0], exp.run_surface(**kwargs)[0]]
        manifest = json.loads((Path(source['geometry']).parent/'manifest.json').read_text())
        unchanged = all(hashlib.sha256((finger.surface.ROOT/name).read_bytes()).hexdigest() == digest for name,digest in manifest['protected_files'].items())
        all_reports = [best,replay]+reports+half
        complete = len(reports) == 5 and len(half) == 2
        physical = bool(complete and unchanged and error < 1e-8 and all(r['surface_physics_passed'] for r in all_reports))
        result = dict(best=best, replay=replay, holdouts=reports, half_timestep=half,
                      source_candidate=str(args.candidate.resolve()), geometry=source['geometry'],
                      surface_physics_passed=physical, precision_20mm_passed=bool(physical and all(r['final_distance_m'] <= .02 for r in all_reports)),
                      training_ready=bool(physical and not manifest['failed_optimizer_frames'] and all(r['fidelity_passed'] for r in all_reports)),
                      replay_observation_error=error, baseline_unchanged=unchanged, completed=complete, quota=latest_usage())
        (args.output/'admission.json').write_text(json.dumps(result, indent=2)+'\n')
        print('COMPLETE',json.dumps({k:result[k] for k in ('surface_physics_passed','precision_20mm_passed','training_ready','completed')}),flush=True)
    finally:
        exp.env.close()


if __name__ == '__main__':
    main()
