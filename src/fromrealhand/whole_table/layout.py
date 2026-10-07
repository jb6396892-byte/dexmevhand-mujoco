"""Full-table sampling with mesh-derived, conservative projected bounds."""
import numpy as np

OBJECTS = ('mug', 'banana', 'sugar_box', 'mustard_bottle', 'tomato_soup_can')


def rotated_bounds(vertices, yaw_deg):
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = np.asarray(vertices) @ rotation.T
    return points.min(axis=0), points.max(axis=0)


def center_bounds(vertices, yaw_deg, config):
    lo, hi = rotated_bounds(vertices, yaw_deg)
    half = np.asarray(config['table_size_xy_m'])/2
    edge = config['edge_clearance_m']
    return -half+edge-lo[:2], half-edge-hi[:2]


def rectangles_clear(a, b, gap):
    return bool(np.any(a[1]+gap <= b[0]) or np.any(b[1]+gap <= a[0]))


def sample(seed, catalog, config, count=None, cup_xy=None):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid seed')
    rng = np.random.RandomState(seed)
    n = int(rng.randint(0, 5)) if count is None else count
    if type(n) is not int or not 0 <= n <= 4:
        raise ValueError('Expected 0 to 4 distractors')
    names = ['mug']+list(rng.permutation(OBJECTS[1:])[:n])
    objects, bounds = [], []
    for name in names:
        yaw = 0.0 if name == 'mug' else float(rng.uniform(-180, 180))
        lower, upper = center_bounds(catalog[name], yaw, config)
        if np.any(lower > upper): raise ValueError('Object does not fit on table')
        lo, hi = rotated_bounds(catalog[name], yaw)
        for attempt in range(2000):
            xy = rng.uniform(lower, upper)
            if name == 'mug' and cup_xy is not None:
                xy = np.asarray(cup_xy, dtype=float)
                if (xy.shape != (2,) or not np.isfinite(xy).all()
                        or np.any(xy < lower) or np.any(xy > upper)):
                    raise ValueError('Cup footprint extends beyond supported table area')
            rect = (lo[:2]+xy, hi[:2]+xy)
            if all(rectangles_clear(rect, other, config['object_gap_m']) for other in bounds):
                break
        else:
            raise ValueError('No legal initial layout; no physics/planning resampling')
        objects.append(dict(name=str(name), xy=xy.tolist(), yaw_deg=yaw,
                            placement_attempts=attempt+1, bounds_xy=[r.tolist() for r in rect]))
        bounds.append(rect)
    return dict(seed=seed, objects=objects, distractor_count=n,
                sampling='full_table_mesh_projected_aabb', central_clear_corridor=False)
