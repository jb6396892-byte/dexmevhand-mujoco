#!/usr/bin/env python3
"""View saved actions in free-object physics, with finger-specific measurements."""
import argparse
import csv
import json
import pickle
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.paths import configure_runtime_paths
from fromrealhand.video_fidelity import FINGERS, FINGER_NAMES, HandLandmarks, fidelity_metrics, source_clock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['baseline', 'video'], default='video')
    parser.add_argument('--rollout', type=Path)
    parser.add_argument('--geometry', type=Path, default=ROOT/'data/processed/seq_dexycb_001/video_faithful_v1/retarget_control/geometry.npz')
    parser.add_argument('--episodes', type=int, default=0)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--camera-distance', type=float, default=.55)
    parser.add_argument('--output', type=Path, help='Render one audited headless replay to this new directory')
    args = parser.parse_args()
    if args.speed <= 0 or args.episodes < 0 or args.camera_distance <= 0:
        parser.error('speed must be positive and episodes nonnegative')
    if args.rollout is None:
        args.rollout = (ROOT/'data/demonstrations/relocate-mug-physics-verified-v1.pkl' if args.version == 'baseline'
                        else ROOT/'data/processed/seq_dexycb_001/video_faithful_v1/refined/best/diagnostic_rollout.pkl')
    with args.rollout.open('rb') as stream:
        demos = pickle.load(stream)
    demo = demos['physics_seed_0'] if args.version == 'baseline' else demos['video_faithful']
    actions = np.asarray(demo['actions'])
    if actions.ndim != 2 or actions.shape[1] != 30 or not np.isfinite(actions).all():
        raise ValueError('Expected finite 30-dimensional saved actions')
    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
    contacts = import_module('21_diagnose_geometry').contacts
    env = YCBRelocate(has_renderer=args.output is None, object_name='mug', object_scale=.8,
                      friction=(1, .5, .01), solref='-6000 -300', randomness_scale=.25)
    landmarks = HandLandmarks(env.sim.model)
    geometry = np.load(args.geometry)
    steps_file = args.rollout.parent/'steps.csv'
    source_frames = None
    if args.version == 'video' and steps_file.exists():
        with steps_file.open() as stream:
            source_frames = [int(round(float(row['source_frame']))) for row in csv.DictReader(stream)]
    elif args.version == 'video' and (args.rollout.parent/'summary.json').exists():
        summary = json.loads((args.rollout.parent/'summary.json').read_text())
        if 'time_scale' in summary:
            frames, fps = geometry['source_frames'], float(geometry['fps'])
            clocks = source_clock(np.arange(len(actions))*env.control_timestep,
                                  float((frames[-1]-frames[0])/fps), summary['time_scale'])
            source_frames = np.rint(frames[0]+clocks*fps).astype(int).tolist()
    if args.output is not None:
        import cv2
        from mujoco_py import MjRenderContextOffscreen
        args.output.mkdir(parents=True, exist_ok=False)
        context = MjRenderContextOffscreen(env.sim)
    print('Saved-action physics replay: %s; %d steps. Ctrl+C closes the window.' % (args.version, len(actions)), flush=True)
    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            env.reset()
            env.sim.reset()
            env.pack_mujoco_model(demo['model_data'][0])
            for key, values in demo.get('physics_model', {}).items():
                if key not in ('geom_margin', 'geom_gap'):
                    raise ValueError('Unsupported physics model field: '+key)
                getattr(env.sim.model, key)[:] = values
            env.pack(demo['sim_data'][0])
            env.sim.forward()
            start = time.monotonic()
            forces, images = [], []
            selected = set(np.linspace(0, len(actions)-1, 8).round().astype(int))
            max_replay_error = 0.
            camera = context.cam if args.output is not None else env.viewer.viewer.cam
            camera.type = 0
            camera.lookat[:] = geometry['object_poses'][[0, -1], :3, 3].mean(axis=0)
            camera.distance = args.camera_distance
            camera.azimuth, camera.elevation = 135., -30.
            for step, action in enumerate(actions):
                max_replay_error = max(max_replay_error, float(np.max(np.abs(env._get_observations()-demo['observations'][step]))))
                env.step(action)
                current = dict.fromkeys(FINGERS, 0.)
                for contact in contacts(env):
                    finger = contact['hand'][2:4]
                    if finger in current:
                        current[finger] += max(0., contact['normal_force_n'])
                forces.append([current[f] for f in FINGERS])
                if args.output is None:
                    if step % 2 == 0:
                        env.render()
                    time.sleep(max(0., start+(step+1)*env.control_timestep/args.speed-time.monotonic()))
                elif step in selected:
                    context.render(640, 480, camera_id=-1)
                    rendered = cv2.cvtColor(context.read_pixels(640, 480, depth=False)[::-1], cv2.COLOR_RGB2BGR)
                    cv2.putText(rendered, '%s physics step %d' % (args.version, step), (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
                    if source_frames is not None:
                        frame = source_frames[step]
                        source = cv2.imread(str(ROOT/'data/real_data/relocate_mug/seq_dexycb_001/rgb'/('%06d.jpg' % frame)))
                        cv2.putText(source, 'Source video frame %d' % frame, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                        rendered = np.hstack([source, rendered])
                    cv2.imwrite(str(args.output/('step_%04d.jpg' % step)), rendered)
                    images.append(rendered)
            pose = np.eye(4)
            pose[:3, 3] = env.sim.data.body_xpos[env.obj_bid]
            pose[:3, :3] = env.sim.data.body_xmat[env.obj_bid].reshape(3, 3)
            metrics = fidelity_metrics(landmarks.read(env.sim.data), pose, geometry['human_joints'][-1], geometry['object_poses'][-1])
            tail = np.asarray(forces[-100:])
            metrics.update(version=args.version, replay_max_observation_error=max_replay_error,
                           tail_finger_force_n=dict(zip(FINGER_NAMES, tail.mean(0).tolist())),
                           tail_finger_contact_fraction=dict(zip(FINGER_NAMES, (tail>.01).mean(0).tolist())))
            print(json.dumps(metrics, indent=2), flush=True)
            if args.output is not None:
                cv2.imwrite(str(args.output/'keyframes.jpg'), np.vstack(images))
                np.save(args.output/'finger_forces.npy', np.asarray(forces))
                (args.output/'comparison.json').write_text(json.dumps(metrics, indent=2)+'\n')
                break
            until = time.monotonic()+2.
            while time.monotonic() < until:
                env.render()
                time.sleep(1./30.)
            episode += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == '__main__':
    main()
