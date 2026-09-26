#!/usr/bin/env python3
"""Open a live MuJoCo window for the verified, free-object grasp rollout."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fromrealhand.paths import configure_runtime_paths
from fromrealhand.verified_curriculum import load_admitted_demos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, choices=range(6))
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed relative to simulated time (default: 1.0)")
    parser.add_argument("--render-every", type=int, default=2,
                        help="Render every N control steps (default: 2, about 50 FPS)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of replays; 0 repeats until Ctrl+C")
    parser.add_argument("--hold-seconds", type=float, default=2.0,
                        help="Keep the final state visible before replaying")
    args = parser.parse_args()
    if args.speed <= 0 or args.render_every < 1 or args.episodes < 0 or args.hold_seconds < 0:
        parser.error("speed and render-every must be positive; episodes and hold-seconds must be nonnegative")

    os.environ["FROMREALHAND_ADMISSION"] = str(
        ROOT / "data/processed/seq_dexycb_001/physical_grasp_verified_v1/admission.json")
    os.environ["FROMREALHAND_VERIFIED_DEMO"] = str(
        ROOT / "data/demonstrations/relocate-mug-physics-verified-v1.pkl")
    demonstrations, _ = load_admitted_demos()
    demonstration = demonstrations["physics_seed_%d" % args.seed]

    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

    env = YCBRelocate(
        has_renderer=True,
        object_name="mug",
        object_scale=0.8,
        friction=(1, 0.5, 0.01),
        solref="-6000 -300",
        randomness_scale=0.25,
    )
    actions = np.asarray(demonstration["actions"])
    if actions.shape != (1000, 30):
        raise ValueError("Expected 1000 normalized, 30-dimensional MuJoCo actions")
    print("MuJoCo physics window: seed %d, %d steps, %.1fx speed; Ctrl+C to close." %
          (args.seed, len(actions), args.speed), flush=True)

    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            env.reset()
            env.sim.reset()
            env.pack_mujoco_model(demonstration["model_data"][0])
            env.pack(demonstration["sim_data"][0])
            env.sim.forward()
            env.render()
            start = time.monotonic()
            for step, action in enumerate(actions):
                env.step(action)
                if step % args.render_every == 0 or step == len(actions) - 1:
                    env.render()
                deadline = start + (step + 1) * env.control_timestep / args.speed
                time.sleep(max(0.0, deadline - time.monotonic()))
            cup = env.sim.data.body_xpos[env.obj_bid]
            target = env.sim.data.body_xpos[env.target_object_bid]
            print("Replay %d: cup z=%.4f m, target distance=%.4f m" %
                  (episode + 1, cup[2], np.linalg.norm(cup - target)), flush=True)
            until = time.monotonic() + args.hold_seconds
            while time.monotonic() < until:
                env.render()
                time.sleep(1.0 / 30.0)
            episode += 1
    except KeyboardInterrupt:
        print("Window closed by Ctrl+C.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
