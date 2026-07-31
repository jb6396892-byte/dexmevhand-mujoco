from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fromrealhand.pose_io import load_pickle


REQUIRED_KEYS = ("observations", "actions", "rewards", "sim_data", "model_data")


@dataclass
class TrajectoryStats:
    trajectory_id: str
    length: int
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    reward_sum: float
    action_saturation_fraction: float


def validate_demo_file(path: str | Path) -> list[TrajectoryStats]:
    demo = load_pickle(path)
    if not isinstance(demo, dict):
        raise TypeError(f"demo root must be dict[trajectory_id] = trajectory, got {type(demo).__name__}")
    if not demo:
        raise ValueError("demo contains no trajectories")

    stats: list[TrajectoryStats] = []
    for trajectory_id, traj in demo.items():
        if not isinstance(traj, dict):
            raise TypeError(f"trajectory {trajectory_id} must be a dict")
        missing = [key for key in REQUIRED_KEYS if key not in traj]
        if missing:
            raise KeyError(f"trajectory {trajectory_id} missing keys: {missing}")

        observations = np.asarray(traj["observations"])
        actions = np.asarray(traj["actions"])
        rewards = np.asarray(traj["rewards"])
        if observations.ndim != 2:
            raise ValueError(f"{trajectory_id}: observations must be 2D, got {observations.shape}")
        if actions.ndim != 2:
            raise ValueError(f"{trajectory_id}: actions must be 2D, got {actions.shape}")
        if rewards.ndim != 1:
            raise ValueError(f"{trajectory_id}: rewards must be 1D, got {rewards.shape}")
        if not (len(observations) == len(actions) == len(rewards)):
            raise ValueError(
                f"{trajectory_id}: length mismatch obs={len(observations)} actions={len(actions)} rewards={len(rewards)}"
            )
        if len(traj["sim_data"]) != len(observations):
            raise ValueError(f"{trajectory_id}: sim_data length mismatch")
        if len(traj["model_data"]) < 1:
            raise ValueError(f"{trajectory_id}: model_data is empty")

        saturation = float(np.mean(np.abs(actions) >= 0.999)) if actions.size else 0.0
        stats.append(
            TrajectoryStats(
                trajectory_id=str(trajectory_id),
                length=len(actions),
                observation_shape=tuple(observations.shape[1:]),
                action_shape=tuple(actions.shape[1:]),
                reward_sum=float(np.sum(rewards)),
                action_saturation_fraction=saturation,
            )
        )
    return stats


def format_stats(stats: list[TrajectoryStats]) -> str:
    lines = [
        "trajectory_id,length,observation_shape,action_shape,reward_sum,action_saturation_fraction",
    ]
    for item in stats:
        lines.append(
            f"{item.trajectory_id},{item.length},{item.observation_shape},{item.action_shape},"
            f"{item.reward_sum:.6f},{item.action_saturation_fraction:.6f}"
        )
    return "\n".join(lines)
