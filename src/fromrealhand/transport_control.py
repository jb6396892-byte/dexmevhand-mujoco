"""Bounded, contact-gated Cartesian PI correction for a grasped object."""
import numpy as np


def limit_norm(value, limit):
    value = np.asarray(value, dtype=float)
    return value*min(1., limit/max(float(np.linalg.norm(value)), 1e-12))


class TransportCorrection:
    def __init__(self, kp, ki, max_shift=.08, max_speed=.03, settle_s=.2):
        if not np.isfinite([kp, ki, max_shift, max_speed, settle_s]).all() or min(kp, ki, settle_s) < 0 or min(max_shift, max_speed) <= 0:
            raise ValueError('Invalid transport controller parameters')
        self.kp, self.ki = kp, ki
        self.max_shift, self.max_speed, self.settle_s = max_shift, max_speed, settle_s
        self.integral = np.zeros(3)
        self.shift = np.zeros(3)
        self.loaded_s = 0.

    def update(self, error, contact_ready, dt):
        error = np.asarray(error, dtype=float)
        if error.shape != (3,) or not np.isfinite(error).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid transport error or timestep')
        self.loaded_s = self.loaded_s+dt if contact_ready else 0.
        if contact_ready and self.loaded_s+1e-12 >= self.settle_s:
            self.integral = limit_norm(self.integral+self.ki*error*dt, self.max_shift)
            target = limit_norm(self.kp*error+self.integral, self.max_shift)
            self.shift += limit_norm(target-self.shift, self.max_speed*dt)
        return self.shift.copy()
