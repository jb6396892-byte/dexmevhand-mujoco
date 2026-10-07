"""Validate project assumptions without importing MuJoCo, Torch, or Qt."""
import json
from math import isfinite
from pathlib import Path
from .contracts import ScaffoldOnlyError


def _vector(value, length):
    if (not isinstance(value, list) or len(value) != length
            or any(type(x) not in (int, float) or not isfinite(x) for x in value)):
        raise ValueError('Invalid finite numeric vector')
    return value


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config['status'] != 'scaffold_only' or config['execution_enabled'] is not False:
        raise ValueError('This entry point is only for framework inspection')
    if config['table']['central_clear_corridor'] is not False:
        raise ValueError('Whole-table sampling must not reserve the old central corridor')
    if any(x <= 0 for x in _vector(config['table']['size_xy_m'], 2)):
        raise ValueError('Invalid table size')
    platform = config['platform']
    if platform['mechanism'] != 'xyz_gantry':
        raise ValueError('Expected the selected XYZ gantry mechanism')
    for field in ('joint_names', 'actuator_names'):
        names = platform[field]
        if len(names) != 3 or len(set(names)) != 3 or not all(isinstance(n, str) and n for n in names):
            raise ValueError('Platform names must be three unique nonempty strings')
    lo = _vector(platform['provisional_joint_min_m'], 3)
    hi = _vector(platform['provisional_joint_max_m'], 3)
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError('Invalid provisional platform range')
    for field in ('provisional_max_velocity_m_s', 'provisional_max_acceleration_m_s2',
                  'provisional_max_jerk_m_s3'):
        if any(x <= 0 for x in _vector(platform[field], 3)):
            raise ValueError('Motion limits must be positive')
    policy = config['local_policy']
    if (policy['action_dim'], policy['feature_dim'], policy['mapping']) != (30, 139, 'named_joints_and_actuators'):
        raise ValueError('Legacy adapter contract changed')
    if config['objects']['resample_after_planning_failure'] is not False:
        raise ValueError('Planning failure must not trigger hidden resampling')
    return config


def inspect(config):
    return dict(status='scaffold_only', configuration_valid=True,
                physics_executed=False, training_started=False, runtime_ready=False,
                table_size_xy_m=config['table']['size_xy_m'],
                mechanism=config['platform']['mechanism'],
                control_channels=dict(platform_metric=3, local_normalized=30),
                available_separately=['F1 gantry and named mapping', 'F1 whole-table layout sampler'],
                f1_entrypoint='scripts/177_launch_whole_table_f1.sh',
                missing=['collision_and_time_parameterized_planner', 'local_policy_frame_adapter',
                         'physical_supervisor_and_qt_integration', 'new_frozen_evaluation'])


def create_runtime(config):
    raise ScaffoldOnlyError('Whole-table grasp runtime is not implemented; F1 platform demo is a separate entrypoint')
