"""SI units; platform commands and legacy normalized actions stay separate."""
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Tuple

Vec3 = Tuple[float, float, float]


class Phase(str, Enum):
    PLAN = 'plan'
    TRANSIT = 'transit'
    ALIGN = 'align'
    GRASP = 'grasp'
    LIFT = 'lift'
    TRANSPORT = 'transport'
    HOLD = 'hold'
    STOP = 'stop'


@dataclass(frozen=True)
class ObjectPose:
    name: str
    world_position_m: Vec3
    world_quaternion_wxyz: Tuple[float, float, float, float]


@dataclass(frozen=True)
class TaskRequest:
    seed: int
    video: str
    objects: Tuple[ObjectPose, ...]
    goal_world_m: Vec3


@dataclass(frozen=True)
class CompositeCommand:
    platform_joint_target_m: Vec3
    local_normalized_action: Tuple[float, ...]

    def __post_init__(self):
        if (len(self.platform_joint_target_m) != 3
                or not all(isfinite(x) for x in self.platform_joint_target_m)):
            raise ValueError('Expected three finite platform joint targets in meters')
        if (len(self.local_normalized_action) != 30
                or not all(isfinite(x) and abs(x) <= 1 for x in self.local_normalized_action)):
            raise ValueError('Expected 30 normalized local actions, not a combined 33-vector')


@dataclass(frozen=True)
class PlanningResult:
    status: str
    reason: str
    platform_waypoints_m: Tuple[Vec3, ...] = ()


@dataclass(frozen=True)
class Telemetry:
    phase: Phase
    platform_position_m: Vec3
    platform_velocity_m_s: Vec3
    goal_error_m: float
    max_penetration_m: float
    reason: str


class ScaffoldOnlyError(RuntimeError):
    pass
