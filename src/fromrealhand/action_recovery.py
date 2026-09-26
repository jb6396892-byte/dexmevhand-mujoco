"""Convert requested actuator force into the normalized Adroit action."""
import numpy as np


def force_to_action(force, qpos, qvel, gain, bias, midpoint, span):
    gain = np.asarray(gain)
    span = np.asarray(span)
    if np.any(gain == 0) or np.any(span <= 0):
        raise ValueError('actuator gains and control spans must be nonzero')
    actuator_bias = bias[:, 0] + bias[:, 1] * qpos + bias[:, 2] * qvel
    control = (force - actuator_bias) / gain
    return np.clip((control - midpoint) / span, -1.0, 1.0)
