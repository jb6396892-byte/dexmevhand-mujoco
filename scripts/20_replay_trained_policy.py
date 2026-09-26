#!/usr/bin/env python3
"""Render a trained mug policy acting in the actual MuJoCo dynamics."""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fromrealhand.paths import configure_runtime_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.fps <= 0 or args.hold_seconds < 0:
        parser.error("steps and fps must be positive; hold-seconds must be nonnegative")

    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate

    with args.policy.open("rb") as stream:
        policy = pickle.load(stream)
    env = YCBRelocate(
        has_renderer=True,
        object_name="mug",
        object_scale=0.8,
        friction=(1, 0.5, 0.01),
        solref="-6000 -300",
        randomness_scale=0.25,
        use_visual_obs=False,
    )
    env.seed(args.seed)
    observation = env.reset()
    initial_height = float(env.sim.data.body_xpos[env.obj_bid, 2])
    max_height = initial_height
    contact_steps = 0
    contact_lift_streak = 0
    max_contact_lift_streak = 0
    reward_sum = 0.0
    elapsed_steps = 0

    try:
        env.render()
        start = time.monotonic()
        for step in range(args.steps):
            action = policy.get_action(observation)[1]["evaluation"]
            observation, reward, _, _ = env.step(action)
            reward_sum += float(reward)
            height = float(env.sim.data.body_xpos[env.obj_bid, 2])
            contact = bool(env.check_contact(env.body_geom_names, env.robot_geom_names))
            contact_steps += int(contact)
            contact_lift_streak = contact_lift_streak + 1 if contact and height > initial_height + 0.015 else 0
            max_contact_lift_streak = max(max_contact_lift_streak, contact_lift_streak)
            max_height = max(max_height, height)
            elapsed_steps += 1
            env.render()
            time.sleep(max(0.0, start + (step + 1) / args.fps - time.monotonic()))
        hold_until = time.monotonic() + args.hold_seconds
        while time.monotonic() < hold_until:
            env.render()
            time.sleep(0.04)
    except KeyboardInterrupt:
        pass

    result = {
        "policy": str(args.policy.resolve()),
        "seed": args.seed,
        "steps": elapsed_steps,
        "reward_sum": reward_sum,
        "initial_object_height_m": initial_height,
        "max_object_height_m": max_height,
        "contact_steps": contact_steps,
        "contact_lift_max_streak": max_contact_lift_streak,
        "grasp_reproduced": max_contact_lift_streak >= 3,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
