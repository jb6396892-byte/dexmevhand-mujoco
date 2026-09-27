#!/usr/bin/env python3
"""Compare loaded collision contacts with the rendered hand and mug meshes."""
import argparse
import json
import pickle
import sys
from pathlib import Path
from importlib import import_module
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.paths import configure_runtime_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configure_runtime_paths()
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
    with args.rollout.open('rb') as f:
        demo = pickle.load(f)['video_faithful']
    e = YCBRelocate(has_renderer=False, object_name='mug', object_scale=.8,
                    friction=(1, .5, .01), solref='-6000 -300', randomness_scale=.25)
    try:
        e.reset(); e.sim.reset()
        e.pack_mujoco_model(demo['model_data'][0])
        for key in ('geom_margin', 'geom_gap'):
            getattr(e.sim.model, key)[:] = demo['physics_model'][key]
        e.pack(demo['sim_data'][0]); e.sim.forward()
        for action in demo['actions']:
            e.step(action)
        m, d = e.sim.model, e.sim.data

        def closest_visual(body_id, point):
            candidates = []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != body_id or m.geom_type[g] != 7 or m.geom_group[g] != 1:
                    continue
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
                f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]]
                mesh = trimesh.Trimesh(vertices=v @ d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g], faces=f, process=False)
                pts, dist, _ = trimesh.proximity.closest_point_naive(mesh, np.array([point]))
                candidates.append((float(dist[0]), pts[0]))
            if not candidates:
                raise ValueError('No visual mesh for body '+m.body_id2name(body_id))
            return min(candidates, key=lambda x: x[0])

        rows = []
        for c in import_module('21_diagnose_geometry').contacts(e):
            if c['normal_force_n'] <= .01:
                continue
            hand = m.geom_name2id(c['hand'])
            hd, hp = closest_visual(m.geom_bodyid[hand], c['position'])
            od, op = closest_visual(e.obj_bid, c['position'])
            rows.append(dict(c, contact_to_hand_visual_m=hd, contact_to_mug_visual_m=od,
                             nearest_visual_points_distance_m=float(np.linalg.norm(hp-op))))
        result = dict(contacts=rows, scope='Final-state contact-neighborhood visual mesh distances; not global signed mesh penetration',
                      max_nearby_visual_separation_m=max((x['nearest_visual_points_distance_m'] for x in rows), default=None))
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        e.close()


if __name__ == '__main__':
    main()
