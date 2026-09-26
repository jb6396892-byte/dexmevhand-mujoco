"""Smooth phase transitions and world-to-root translation mapping."""
import numpy as np


def smooth_blend(t):
    t = np.clip(t, 0., 1.)
    return t**3 * (10. - 15.*t + 6.*t*t)


def world_to_root_delta(translation_jacobian, displacement):
    axes = np.asarray(translation_jacobian)
    displacement = np.asarray(displacement)
    if axes.shape != (3, 3) or displacement.shape != (3,):
        raise ValueError('Expected a 3x3 translation Jacobian and a 3-vector')
    if not np.isfinite(axes).all() or not np.isfinite(displacement).all():
        raise ValueError('Non-finite root translation input')
    return np.linalg.solve(axes, displacement)
