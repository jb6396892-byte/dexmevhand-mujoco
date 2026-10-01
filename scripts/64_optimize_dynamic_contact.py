#!/usr/bin/env python3
"""Optimize bounded actions through free-cup dynamics, without policy training."""
import argparse
import hashlib
import json
import pickle
import time
from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

surface = import_module('33_optimize_surface_grasp')
from fromrealhand.video_fidelity import TIP_INDICES, object_relative, source_clock
from fromrealhand.validation import validate_demo_file
from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
from mujoco_py import functions


def action_envelope(count, start, transition):
    phase = np.clip((np.arange(count)-start)/float(transition), 0., 1.)
    return phase**3*(10.-15.*phase+6.*phase**2)


def corrected_actions(original, parameters, scales, conversion, start, transition):
    envelope = action_envelope(len(original), start, transition)
    return original+envelope[:, None]*(parameters*scales*conversion)[None, :]


class ShootingProblem:
    def __init__(self, source, demo, args):
        self.args, self.demo, self.source = args, demo, source
        self.e = surface.video.VideoExperiment(source['geometry'])
        self.env, self.m, self.d = self.e.env, self.e.model, self.e.env.sim.data
        e, m, d = self.env, self.m, self.d
        e.pack_mujoco_model(demo['model_data'][0])
        for key, value in demo['physics_model'].items():
            getattr(m, key)[:] = value
        self.physics = {k:getattr(m,k).copy() for k in ('geom_margin','geom_gap','geom_friction','body_mass','jnt_range')}
        self.hands = {i for i in range(m.ngeom) if m.geom_id2name(i) in self.e.hand_geoms}
        self.mugs = {i for i in range(m.ngeom) if m.geom_id2name(i) in self.e.mug_geoms}
        self.rings = [i for i in self.hands if (m.geom_id2name(i) or '').startswith('C_rf') and m.geom_type[i] == 3]
        self.tips = [m.site_name2id('S_'+f+'tip') for f in ('th','ff','mf','rf','lf')]
        e.sim.forward()
        vertices = []
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] != e.obj_bid or m.geom_type[gid] != 7 or m.geom_group[gid] != 1:
                continue
            mid = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            world = v @ d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
            vertices.append((world-d.body_xpos[e.obj_bid]) @ d.body_xmat[e.obj_bid].reshape(3,3))
        self.tree = cKDTree(np.vstack(vertices))
        self.scales = np.array([.01]*3+[.12]*27)
        self.conversion = -m.actuator_biasprm[:,1]/(m.actuator_gainprm[:,0]*e.act_rng)
        self.max_depth = 0.

        def audited(action, policy_step=False):
            self.audit_depth()
            return YCBRelocate._pre_action(e, action, policy_step)

        e._pre_action = audited
        e.reset(); e.sim.reset()
        e.pack_mujoco_model(demo['model_data'][0])
        e.pack(demo['sim_data'][0]); e.sim.forward()
        for action in demo['actions'][:args.start]:
            e.step(action)
        self.checkpoint = e.sim.get_state()
        self.warmstart = d.qacc_warmstart.copy()
        self.ctrl = d.ctrl.copy()
        self.clock = e.cur_time
        self.prefix_depth = self.max_depth
        self.evaluations, self.best_cost = 0, float('inf')
        self.history = []
        self.cached = None
        self.best_x = None
        self.start_time = time.monotonic()

    def audit_depth(self):
        depth = max([max(0., -float(c.dist)) for c in self.d.contact[:self.d.ncon]
                     if int(c.geom1) in self.hands or int(c.geom2) in self.hands] or [0.])
        self.max_depth = max(self.max_depth, depth)

    def measure(self, step):
        e, m, d = self.env, self.m, self.d
        force = dict.fromkeys(('th','ff','mf','rf','lf'), 0.)
        native_ring = []
        for i, c in enumerate(d.contact[:d.ncon]):
            a, b = int(c.geom1), int(c.geom2)
            gid = a if a in self.hands and b in self.mugs else b if b in self.hands and a in self.mugs else None
            if gid is None:
                continue
            finger = m.geom_id2name(gid)[2:4]
            if finger in force:
                f = np.zeros(6); functions.mj_contactForce(m, d, i, f)
                force[finger] += max(0., float(f[0]))
            if finger == 'rf':
                native_ring.append(float(c.dist))
        if native_ring:
            gap = min(native_ring)
        else:
            points, radii = [], []
            for gid in self.rings:
                axis = d.geom_xmat[gid].reshape(3,3)[:,2]
                points.extend(d.geom_xpos[gid]+np.linspace(-m.geom_size[gid,1],m.geom_size[gid,1],11)[:,None]*axis)
                radii.extend([m.geom_size[gid,0]]*11)
            local = (np.asarray(points)-d.body_xpos[e.obj_bid]) @ d.body_xmat[e.obj_bid].reshape(3,3)
            gap = float(np.min(self.tree.query(local)[0]-radii))
        clock = float(source_clock(step*e.control_timestep,self.e.duration,self.source['best']['time_scale']))
        reference = np.eye(4); reference[:3,3] = self.e.pcurve(clock); reference[:3,:3] = self.e.rcurve(clock).as_matrix()
        targets = object_relative(self.e.jcurve(clock)[TIP_INDICES],reference)
        actual = (d.site_xpos[self.tips]-d.body_xpos[e.obj_bid]) @ d.body_xmat[e.obj_bid].reshape(3,3)
        error = actual-targets
        goal = d.body_xpos[e.obj_bid]-self.e.geometry['object_poses'][-1,:3,3]
        palm = m.body_name2id('palm')
        relative = (d.body_xpos[e.obj_bid]-d.body_xpos[palm]) @ d.body_xmat[palm].reshape(3,3)
        return dict(gap=gap, forces=force, tip_error=error, goal=goal, relative=relative,
                    cup_speed=float(np.linalg.norm(d.qvel[30:33])), cup_height=float(d.body_xpos[e.obj_bid,2]))

    def action_sequence(self, x):
        return corrected_actions(self.demo['actions'],x,self.scales,self.conversion,self.args.start,self.args.transition)

    def evaluate(self, x):
        if self.cached is not None and np.array_equal(x,self.cached[0]):
            return self.cached[1]
        e, m, d = self.env, self.m, self.d
        # Restore solver memory as well as state so trial order does not affect derivatives.
        e.sim.set_state(self.checkpoint)
        d.ctrl[:] = self.ctrl
        e.sim.forward(); d.qacc_warmstart[:] = self.warmstart
        e.timestep = self.args.start; e.cur_time = self.clock; e.done = False
        self.max_depth = self.prefix_depth
        raw = self.action_sequence(x)
        rows = []
        for step in range(self.args.start,len(raw)):
            e.step(np.clip(raw[step],-1.,1.))
            if step >= len(raw)-150 and step % 10 == 9:
                rows.append(self.measure(step))
        self.audit_depth()
        residual = []
        for row in rows:
            residual.append(self.args.ring_weight*(row['gap']+.00015))
            residual.extend(self.args.force_weight*np.maximum(0., .1-np.array([row['forces'][f] for f in ('th','ff','mf','rf')])))
            residual.extend((.2*row['tip_error']).ravel())
            error = np.linalg.norm(row['tip_error'],axis=1)
            residual.extend(2*np.maximum(0.,error-.023))
            residual.append(2*max(0.,float(np.mean(error))-.013))
            residual.extend(self.args.goal_weight*row['goal'])
            residual.append(.01*max(0.,row['cup_speed']-.01))
        saturation = float(np.mean(np.abs(raw)>=.999))
        residual.extend([20*max(0.,self.max_depth-.0008), .1*max(0.,saturation-.005)])
        residual.extend(.0003*x)
        result = np.asarray(residual)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite dynamic trajectory')
        cost = float(result @ result)
        last = rows[-10:]
        report = dict(evaluation=self.evaluations,cost=cost,max_penetration_m=self.max_depth,
                      final_distance_m=float(np.linalg.norm(rows[-1]['goal'])), ring_gap_m=rows[-1]['gap'],
                      ring_contact_fraction=float(np.mean([r['forces']['rf']>.01 for r in last])),
                      min_support_contact_fraction=min(float(np.mean([r['forces'][f]>.01 for r in last])) for f in ('th','ff','mf')),
                      saturation=saturation,elapsed_s=time.monotonic()-self.start_time)
        self.evaluations += 1
        self.history.append(report)
        if cost < self.best_cost:
            self.best_cost, self.best_x = cost, x.copy()
            (self.args.output/'best.json').write_text(json.dumps(dict(parameters=x.tolist(),**report),indent=2)+'\n')
            np.save(self.args.output/'best_actions.npy',np.clip(raw,-1.,1.))
            print('BEST',json.dumps(report),flush=True)
        if self.evaluations % 10 == 0:
            (self.args.output/'progress.json').write_text(json.dumps(self.history,indent=2)+'\n')
            print('EVAL',json.dumps(report),flush=True)
        self.cached = (x.copy(),result.copy())
        return result

    def jacobian(self,x):
        base = self.evaluate(x).copy()
        cols = []
        for i in range(len(x)):
            xp = x.copy()
            dx = self.args.fd_step if x[i]+self.args.fd_step <= self.args.bound else -self.args.fd_step
            xp[i] += dx
            cols.append((self.evaluate(xp)-base)/dx)
        return np.column_stack(cols)


def verify(source,demo,actions,output,demo_path=None):
    actions = np.asarray(actions)
    if actions.shape != np.asarray(demo['actions']).shape or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.:
        raise ValueError('Verification requires finite normalized actions with the original horizon')
    if demo_path is not None and demo_path.exists():
        raise FileExistsError(demo_path)
    e = surface.SurfaceExperiment(source['geometry'])
    e.env.pack_mujoco_model(demo['model_data'][0])
    b = source['best']
    kwargs = dict(scale=b['time_scale'],close=b['close'],gain=b['cartesian_gain'])
    try:
        nominal,_ = e.run_surface(**kwargs,saved_actions=actions,output=output/'nominal')
        accepted_demo = e.last_demo
        reports = [nominal]
        replay_error = None
        passed = lambda r:r['surface_physics_passed'] and r['fidelity_passed'] and r['final_distance_m'] <= .02
        if passed(nominal):
            original = e.last_demo['observations'].copy()
            replay,_ = e.run_surface(**kwargs,saved_actions=actions,output=output/'saved_replay')
            replay_error = float(np.max(np.abs(original-e.last_demo['observations'])))
            reports.append(replay)
            e.model.opt.timestep /= 2; e.env.model_timestep = e.model.opt.timestep
            half,_ = e.run_surface(**kwargs,saved_actions=actions,output=output/'half_timestep')
            reports.append(half)
        ready = bool(len(reports)==3 and replay_error < 1e-8 and all(passed(r) for r in reports))
        manifest = json.loads((Path(source['geometry']).parent/'manifest.json').read_text())
        protected = {name:hashlib.sha256((surface.ROOT/name).read_bytes()).hexdigest()==digest
                     for name,digest in manifest['protected_files'].items()}
        finite = all(np.isfinite(np.asarray(accepted_demo[k])).all() for k in ('observations','actions','rewards'))
        ready = bool(ready and all(protected.values()) and not manifest['failed_optimizer_frames'] and finite)
        result = dict(training_ready=ready,scope='Nominal scene plus exact replay and half timestep only',
                      training_started=False,generalization_started=False,replay_observation_error=replay_error,reports=reports,
                      geometry=source['geometry'],protected_files_unchanged=protected,finite_demo=finite,
                      action_array_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                      geometry_sha256=hashlib.sha256(Path(source['geometry']).read_bytes()).hexdigest(),
                      nominal_rollout_sha256=hashlib.sha256((output/'nominal/diagnostic_rollout.pkl').read_bytes()).hexdigest())
        if ready and demo_path is not None:
            demo_path.parent.mkdir(parents=True,exist_ok=True)
            with demo_path.open('xb') as stream:
                pickle.dump({'seq_dexycb_002_dynamic_v9':accepted_demo},stream)
            stats = validate_demo_file(demo_path)
            result.update(demo=str(demo_path.resolve()),demo_sha256=hashlib.sha256(demo_path.read_bytes()).hexdigest(),
                          data_format_passed=True,trajectory_count=len(stats),horizon=stats[0].length)
        (output/'admission.json').write_text(json.dumps(result,indent=2)+'\n')
        print('ADMISSION',json.dumps(dict(training_ready=ready,nominal_physics=nominal['surface_physics_passed'],nominal_fidelity=nominal['fidelity_passed'],ring=nominal['tail_finger_contact_fraction']['ring'])),flush=True)
        return result
    finally:
        e.env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',type=int,default=850)
    p.add_argument('--transition',type=int,default=300)
    p.add_argument('--max-nfev',type=int,default=20)
    p.add_argument('--fd-step',type=float,default=.01)
    p.add_argument('--bound',type=float,default=2.)
    p.add_argument('--ring-weight',type=float,default=6.)
    p.add_argument('--goal-weight',type=float,default=1.)
    p.add_argument('--force-weight',type=float,default=.02)
    p.add_argument('--method',choices=['least-squares','powell'],default='least-squares')
    p.add_argument('--max-evaluations',type=int,default=1500)
    p.add_argument('--initial',type=Path)
    p.add_argument('--verify-actions',type=Path)
    p.add_argument('--demo',type=Path,help='Export a single nominal-scene demonstration only after all verification gates pass')
    args = p.parse_args()
    source = json.loads(args.candidate.read_text())
    demo = pickle.loads(args.rollout.read_bytes())['video_faithful']
    if not 0 <= args.start < len(demo['actions'])-150 or args.transition <= 0:
        p.error('Invalid transition interval')
    if not all(np.isfinite(v) and v > 0 for v in (args.fd_step,args.bound,args.ring_weight,args.goal_weight,args.force_weight)):
        p.error('Weights, bounds and finite-difference step must be positive and finite')
    if args.max_nfev < 1 or args.max_evaluations < 1 or args.fd_step > 2*args.bound:
        p.error('Invalid optimization limits')
    args.output.mkdir(parents=True,exist_ok=False)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rollout_sha256=hashlib.sha256(args.rollout.read_bytes()).hexdigest())
    (args.output/'configuration.json').write_text(json.dumps(config,indent=2)+'\n')
    if args.verify_actions:
        verify(source,demo,np.load(args.verify_actions),args.output,args.demo)
        return
    problem = ShootingProblem(source,demo,args)
    try:
        x = np.zeros(30) if args.initial is None else np.asarray(json.loads(args.initial.read_text())['parameters'])
        if x.shape != (30,) or not np.isfinite(x).all() or np.max(np.abs(x)) > args.bound:
            raise ValueError('Initial parameters must be finite, bounded and 30-dimensional')
        if args.method == 'powell':
            def cost(values):
                residual = problem.evaluate(values)
                return float(residual @ residual)
            solve = minimize(cost,x,method='Powell',bounds=[(-args.bound,args.bound)]*len(x),
                             options=dict(maxfev=args.max_evaluations,maxiter=args.max_nfev,xtol=.002,ftol=1e-5,disp=True,direc=np.eye(len(x))*.3))
        else:
            solve = least_squares(problem.evaluate,x,jac=problem.jacobian,bounds=(-args.bound,args.bound),
                                  max_nfev=args.max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-7,verbose=2)
        unchanged = all(np.array_equal(getattr(problem.m,k),v) for k,v in problem.physics.items())
        (args.output/'optimization.json').write_text(json.dumps(dict(success=bool(solve.success),message=solve.message,
            evaluations=problem.evaluations,physics_parameters_unchanged=unchanged),indent=2)+'\n')
        (args.output/'progress.json').write_text(json.dumps(problem.history,indent=2)+'\n')
        if not unchanged:
            raise RuntimeError('Physics parameters changed during shooting')
    except KeyboardInterrupt:
        (args.output/'progress.json').write_text(json.dumps(problem.history,indent=2)+'\n')
        (args.output/'optimization.json').write_text(json.dumps(dict(success=False,interrupted=True,
            evaluations=problem.evaluations,training_started=False,generalization_started=False,
            physics_parameters_unchanged=all(np.array_equal(getattr(problem.m,k),v) for k,v in problem.physics.items())),indent=2)+'\n')
        return
    finally:
        problem.env.close()
    verify(source,demo,np.load(args.output/'best_actions.npy'),args.output,args.demo)


if __name__ == '__main__':
    main()
