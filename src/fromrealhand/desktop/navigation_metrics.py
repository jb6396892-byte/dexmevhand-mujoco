"""Separate translational travel limits from angular joint limits for telemetry."""
import numpy as np


def joint_limit_metrics(qpos,joint_ranges,translation_ranges):
    q=np.asarray(qpos)[:30];ranges=np.asarray(joint_ranges)[:30]
    travel=np.asarray(translation_ranges)
    angular=np.maximum(ranges[3:,0]-q[3:],q[3:]-ranges[3:,1])
    translation=np.maximum(travel[:,0]-q[:3],q[:3]-travel[:,1])
    return dict(joint_violation_rad=max(0.,float(angular.max())),
                root_translation_violation_m=max(0.,float(translation.max())))
