#!/usr/bin/env python3
"""Check whether a saved relocation demonstration reproduces in MuJoCo dynamics."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from mjrl.utils.get_environment import get_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--env-name", default="relocate-mug-0.8")
    args = parser.parse_args()

    with args.demo.open("rb") as stream:
        trajectories = pickle.load(stream)
    if len(trajectories) != 1:
        parser.error("expected exactly one trajectory")
    demo = next(iter(trajectories.values()))
    env = get_environment(args.env_name).env
    env.seed(args.seed)
    env.reset()
    reset_object = env.sim.data.body_xpos[env.obj_bid].copy()
    reset_palm = env.sim.data.site_xpos[env.S_grasp_sid].copy()
    reset_target = env.sim.data.body_xpos[env.target_object_bid].copy()
    env.pack_mujoco_model(demo["model_data"][0])
    env.sim.forward()

    true_observations = []
    forced_contacts = 0
    forced_heights = []
    for state in demo["sim_data"]:
        env.pack(state)
        env.sim.forward()
        true_observations.append(env._get_observations().copy())
        forced_contacts += int(env.check_contact(env.body_geom_names, env.robot_geom_names))
        forced_heights.append(float(env.sim.data.body_xpos[env.obj_bid, 2]))

    initial = demo["sim_data"][0]
    env.pack(initial)
    env.sim.forward()
    demo_object = env.sim.data.body_xpos[env.obj_bid].copy()
    demo_palm = env.sim.data.site_xpos[env.S_grasp_sid].copy()
    demo_target = env.sim.data.body_xpos[env.target_object_bid].copy()

    dynamic_contacts = 0
    dynamic_heights = []
    contact_lift_streak = 0
    max_contact_lift_streak = 0
    step_error = None
    for i, action in enumerate(demo["actions"]):
        try:
            env.step(action)
        except Exception as exc:
            step_error = "step {}: {}: {}".format(i, type(exc).__name__, exc)
            break
        contact = env.check_contact(env.body_geom_names, env.robot_geom_names)
        height = float(env.sim.data.body_xpos[env.obj_bid, 2])
        dynamic_contacts += int(contact)
        dynamic_heights.append(height)
        contact_lift_streak = contact_lift_streak + 1 if contact and height > demo_object[2] + 0.015 else 0
        max_contact_lift_streak = max(max_contact_lift_streak, contact_lift_streak)

    qpos = np.stack([state["qpos"] for state in demo["sim_data"]])
    root_outside = []
    for i in range(6):
        joint = env.mjpy_model.actuator_trnid[i, 0]
        column = env.mjpy_model.jnt_qposadr[joint]
        low, high = env.mjpy_model.actuator_ctrlrange[i]
        root_outside.append(float(np.mean((qpos[:, column] < low - 1e-6) | (qpos[:, column] > high + 1e-6))))

    true_observations = np.asarray(true_observations)
    obs_error = np.abs(demo["observations"] - true_observations)
    result = {
        "steps": len(demo["actions"]),
        "reset_object_xyz_m": reset_object.tolist(),
        "demo_object_xyz_m": demo_object.tolist(),
        "environment_name": args.env_name,
        "reset_palm_xyz_m": reset_palm.tolist(),
        "demo_palm_xyz_m": demo_palm.tolist(),
        "reset_target_xyz_m": reset_target.tolist(),
        "demo_target_xyz_m": demo_target.tolist(),
        "observation_max_abs_error": float(obs_error.max()),
        "root_joint_outside_ctrl_fraction": root_outside,
        "forced_state_contact_steps": forced_contacts,
        "forced_state_max_object_height_m": max(forced_heights),
        "dynamic_contact_steps": dynamic_contacts,
        "dynamic_contact_lift_max_streak": max_contact_lift_streak,
        "dynamic_max_object_height_m": max(dynamic_heights) if dynamic_heights else None,
        "dynamic_step_error": step_error,
    }
    result["dynamic_grasp_reproduced"] = bool(max_contact_lift_streak >= 3 and step_error is None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["dynamic_grasp_reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
