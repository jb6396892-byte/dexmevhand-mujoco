"""Conservative F1 test motions; obstacle planning remains an F3 task."""
import numpy as np
from scipy.interpolate import BPoly


def rest_to_rest(start, target, config):
    start, target = np.asarray(start), np.asarray(target)
    delta = np.abs(target-start)
    # Bounds of the quintic smoothstep's first three derivatives.
    duration = max(1., float(np.max(1.875*delta/np.asarray(config['max_velocity_m_s']))),
                   float(np.max(np.sqrt(5.774*delta/np.asarray(config['max_acceleration_m_s2'])))),
                   float(np.max(np.cbrt(60*delta/np.asarray(config['max_jerk_m_s3'])))))*1.25
    return duration, BPoly.from_derivatives([0., duration],
                    [[start, np.zeros(3), np.zeros(3)], [target, np.zeros(3), np.zeros(3)]])


def braking(position, velocity, acceleration, duration=2.0):
    position, velocity = np.asarray(position), np.asarray(velocity)
    target = position+velocity*duration/2
    curve = BPoly.from_derivatives([0., duration],
        [[position, velocity, np.asarray(acceleration)], [target, np.zeros(3), np.zeros(3)]])
    return duration, curve
