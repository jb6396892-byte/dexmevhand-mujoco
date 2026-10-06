"""Deterministic task sampling, separated from physics success and evaluation."""
import numpy as np

DISTRACTORS = ('banana', 'sugar_box', 'mustard_bottle', 'tomato_soup_can')


def validate_goal(goal, config):
    value = np.asarray(goal, dtype=float)
    if (value.shape != (3,) or not np.isfinite(value).all()
            or np.any(value < config['goal_min_m']) or np.any(value > config['goal_max_m'])):
        raise ValueError('Target outside registered random-task workspace')
    return value.copy()


def validate_cup(cup_xy, config):
    value=np.asarray(cup_xy,dtype=float)
    if (value.shape!=(2,) or not np.isfinite(value).all()
            or np.any(value<config['cup_xy_min_m']) or np.any(value>config['cup_xy_max_m'])):
        raise ValueError('Initial cup position outside registered workspace')
    return value.copy()


def sample(seed, config, goal=None, count=None, cup_xy=None):
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError('Invalid scene seed')
    rng = np.random.RandomState(seed)
    cup = rng.uniform(config['cup_xy_min_m'], config['cup_xy_max_m'])
    if cup_xy is not None: cup=validate_cup(cup_xy,config)
    target = rng.uniform(config['goal_min_m'], config['goal_max_m'])
    if goal is not None: target = validate_goal(goal, config)
    low, high = config['distractor_count_range']
    n = int(rng.randint(low, high+1))
    if count is not None:
        if not isinstance(count, (int, np.integer)) or not low <= count <= high:
            raise ValueError('Invalid distractor count')
        n = int(count)
    names = list(rng.permutation(DISTRACTORS)[:n])
    objects = [dict(name='mug', xy=cup.tolist(), yaw_deg=float(config['cup_yaw_deg']))]
    for name in names:
        for _ in range(2000):
            xy = np.array([rng.choice([-1, 1])*rng.uniform(*config['distractor_x_abs_range_m']),
                           rng.uniform(*config['distractor_y_range_m'])])
            if all(np.linalg.norm(xy-np.asarray(o['xy'])) >= config['distractor_min_separation_m'] for o in objects):
                break
        else: raise ValueError('Cannot sample separated objects; do not discard this episode')
        objects.append(dict(name=str(name), xy=xy.tolist(), yaw_deg=float(rng.uniform(-25, 25))))
    variant = int(rng.randint(len(config['table_colors'])))
    return dict(seed=int(seed), objects=objects, goal_world_m=target.tolist(), distractor_count=n,
                table_variant=variant, table_rgba=config['table_colors'][variant],
                cup_model='025_mug', object_scale=.8, version=config['version'])
