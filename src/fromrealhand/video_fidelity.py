"""Object-relative, finger-specific measurements for video imitation."""
import numpy as np

FINGERS = ("th", "ff", "mf", "rf", "lf")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
TIP_INDICES = np.array([4, 8, 12, 16, 20])


def object_relative(points, pose):
    points, pose = np.asarray(points), np.asarray(pose)
    return (points - pose[:3, 3]) @ pose[:3, :3]


def shifted_tip_targets(local, offset, phase):
    """Return control targets without modifying the human fidelity reference."""
    local, offset = np.asarray(local), np.asarray(offset, dtype=float)
    if (local.shape != (5, 3) or offset.shape != (5, 3)
            or not np.isfinite(local).all() or not np.isfinite(offset).all()
            or not np.isfinite(phase) or np.any(np.linalg.norm(offset, axis=1) > .025+1e-10)):
        raise ValueError('Finite 5x3 tip targets and offsets bounded to 25mm required')
    return local+np.clip(phase, 0., 1.)*offset


def finger_directions(points):
    chains = np.asarray(points)[1:].reshape(5, 4, 3)
    bones = np.diff(chains, axis=1)
    return bones / np.maximum(np.linalg.norm(bones, axis=-1, keepdims=True), 1e-9)


def fidelity_metrics(robot_points, robot_pose, human_points, human_pose):
    robot = object_relative(robot_points, robot_pose)
    human = object_relative(human_points, human_pose)
    errors = np.linalg.norm(robot[TIP_INDICES] - human[TIP_INDICES], axis=-1)
    dots = np.sum(finger_directions(robot) * finger_directions(human), axis=-1)
    angles = np.rad2deg(np.arccos(np.clip(dots, -1, 1)))
    return dict(tip_error_m=errors.tolist(), mean_tip_error_m=float(errors.mean()),
                max_tip_error_m=float(errors.max()),
                finger_direction_error_deg=angles.mean(axis=-1).tolist())


def source_clock(time, duration, time_scale=3., warmup=.5):
    if duration <= 0 or time_scale <= 0:
        raise ValueError("duration and time_scale must be positive")
    phase = np.clip((np.asarray(time) - warmup) / (duration * time_scale), 0., 1.)
    # Smooth retiming preserves every source frame and starts/stops at zero speed.
    return duration * (6 * phase**5 - 15 * phase**4 + 10 * phase**3)


class HandLandmarks:
    def __init__(self, model):
        self.body_ids = [model.body_name2id("palm")]
        for finger in FINGERS:
            self.body_ids.extend(model.body_name2id(finger + part)
                                 for part in ("proximal", "middle", "distal"))
        self.tip_ids = [model.site_name2id("S_" + finger + "tip") for finger in FINGERS]

    def read(self, data):
        points = [data.body_xpos[self.body_ids[0]].copy()]
        for i, tip in enumerate(self.tip_ids):
            points.extend(data.body_xpos[self.body_ids[1 + 3*i:4 + 3*i]].copy())
            points.append(data.site_xpos[tip].copy())
        return np.asarray(points)
