#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from mujoco_py import MjRenderContextOffscreen

from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved mug policy and capture offscreen frames.")
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--sample-count", type=int, default=8)
    args = parser.parse_args()

    with args.policy.open("rb") as stream:
        policy = pickle.load(stream)
    env = YCBRelocate(has_renderer=False, object_name="mug", object_scale=0.8, friction=(1, 0.5, 0.01))
    env.seed(200)
    observation = env.reset()
    context = MjRenderContextOffscreen(env.sim)
    env.sim.add_render_context(context)
    camera_id = env.sim.model.camera_name2id("frontview")
    sampled = set(np.linspace(0, args.steps - 1, min(args.sample_count, args.steps)).round().astype(int))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rewards = []
    heights = []
    contacts = 0
    for step in range(args.steps):
        action = policy.get_action(observation)[1]["evaluation"]
        observation, reward, _, _ = env.step(action)
        rewards.append(float(reward))
        heights.append(float(env.sim.data.body_xpos[env.obj_bid, 2]))
        contacts += int(env.check_contact(env.body_geom_names, env.robot_geom_names))
        if step in sampled:
            context.render(640, 480, camera_id=camera_id)
            rgb = context.read_pixels(640, 480, depth=False)[::-1]
            cv2.imwrite(str(args.output_dir / f"{step:04d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    summary = {
        "policy": str(args.policy.resolve()),
        "steps": args.steps,
        "reward_sum": float(np.sum(rewards)),
        "object_height_initial_m": heights[0],
        "object_height_final_m": heights[-1],
        "object_height_max_m": float(np.max(heights)),
        "contact_steps": contacts,
        "frames_saved": len(sampled),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
