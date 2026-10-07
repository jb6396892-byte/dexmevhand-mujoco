#!/usr/bin/env python3
"""Test whole-table placement and empty-hand platform positioning; no grasping."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from hierarchy_common import ROOT, write
from fromrealhand.whole_table.layout import sample, center_bounds
from fromrealhand.whole_table.scene import PlatformScene, object_catalog
from fromrealhand.whole_table.motion import rest_to_rest, braking
from fromrealhand.whole_table.platform_checks import move, screenshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/whole-table-f1.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layout-seeds', type=int, nargs='+')
    a = p.parse_args(); config = json.loads(a.config.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    catalog = object_catalog()
    write(a.output/'mesh-bounds.json', {n: dict(min_m=v.min(0).tolist(), max_m=v.max(0).tolist()) for n,v in catalog.items()})
    rows = []
    seeds = config['layout_seeds'] if a.layout_seeds is None else a.layout_seeds
    for seed in seeds:
        layout = sample(seed, catalog, config, count=seed % 5)
        scene = PlatformScene(layout, config)
        objects = scene.object_report()
        rows.append(dict(seed=seed, layout=layout, objects=objects,
            passed=all(r['supported'] for r in objects) and scene.peak_penetration <= config['max_penetration_m'],
            low_velocity=all(r['low_velocity'] for r in objects),
            max_penetration_m=scene.peak_penetration, initial_hand_site_m=scene.initial_site.tolist()))
        write(a.output/'layouts.json', rows)
        print('layout', seed, rows[-1]['passed'], flush=True)
    # Dedicated endpoint checks use an empty-hand hover, not a grasp policy.
    scene = PlatformScene(sample(101, catalog, config, count=0), config)
    (a.output/'platform-scene.xml').write_text(scene.xml)
    write(a.output/'joint-map.json', scene.mapping)
    moves = []
    for x in (-.40, 0., .40):
        for y in (-.375, 0., .375):
            row = move(scene, [x, y, .30]); row['case'] = 'grid'
            moves.append(row); write(a.output/'moves.json', moves)
            print('grid', x, y, row['passed'], row['final_joint_error_m'], flush=True)
    for axis in range(3):
        for side in ('min', 'max'):
            target = np.array(config['home_m'])
            target[axis] = config['joint_'+side+'_m'][axis]+(.005 if side == 'min' else -.005)
            row = move(scene, target); row.update(case='limit', axis=axis, side=side)
            moves.append(row); write(a.output/'moves.json', moves)
            print('limit', axis, side, row['passed'], flush=True)
    # Prove invalid setpoints are rejected without advancing simulation time.
    rejected = []
    for target in ([.601, 0, .3], [-.601, 0, .3], [0, .601, .3], [0, -.601, .3],
                   [0, 0, -.001], [0, 0, .451], [float('nan'), 0, .3]):
        before = scene.sim.data.time
        try: scene.step(target)
        except ValueError: rejected.append(scene.sim.data.time == before)
        else: rejected.append(False)
    reset_move = move(scene, config['home_m'])
    duration, curve = rest_to_rest(scene.position(), [.35, .25, .4], config)
    for step in range(int(duration/config['timestep_s']/2)):
        transition_time = (step+1)*config['timestep_s']
        scene.step(curve(transition_time))
    # Keep servo reference position, velocity and acceleration continuous.
    # Restarting from measured q would abruptly erase the servo's tracking lag.
    brake_duration, brake_curve = braking(curve(transition_time),
        curve.derivative(1)(transition_time), curve.derivative(2)(transition_time))
    brake = move(scene, brake_curve(brake_duration), curve_spec=(brake_duration, brake_curve))
    brake['final_speed_m_s'] = np.abs(scene.sim.data.qvel[scene.vids]).tolist()
    brake['passed'] = brake['passed'] and max(brake['final_speed_m_s']) < .001
    write(a.output/'braking.json', brake)
    # Check extreme legal object centers against actual settled geometry.
    lower, upper = center_bounds(catalog['mug'], 0, config)
    corners = []
    for i, xy in enumerate(([lower[0], lower[1]], [lower[0], upper[1]], [upper[0], lower[1]], [upper[0], upper[1]])):
        corner_scene = PlatformScene(sample(200+i, catalog, config, count=4, cup_xy=xy), config)
        objects = corner_scene.object_report()
        corners.append(dict(cup_xy=xy, objects=objects,
            max_penetration_m=corner_scene.peak_penetration,
            low_velocity=all(o['low_velocity'] for o in objects),
            passed=all(o['supported'] for o in objects) and corner_scene.peak_penetration <= config['max_penetration_m']))
    write(a.output/'corners.json', corners)
    rendered_scene = PlatformScene(sample(9, catalog, config, count=4), config)
    images = dict(scene=screenshot(rendered_scene, a.output/'whole-table.png'))
    image_motion = move(rendered_scene, [.38, -.35, .35])
    images['corner'] = screenshot(rendered_scene, a.output/'platform-corner.png')
    files = list((ROOT/'src/fromrealhand/whole_table').glob('*.py'))+[a.config, Path(__file__).resolve()]
    summary = dict(scope='F1 platform and layout, NOT grasp success',
        passed=all(r['passed'] for r in rows+moves+corners) and all(rejected) and brake['passed'] and reset_move['passed'] and image_motion['passed'],
        layouts_passed=sum(r['passed'] for r in rows), layouts_total=len(rows),
        low_velocity_layouts=sum(r['low_velocity'] for r in rows),
        low_velocity_is_additional_diagnostic=True,
        grid_passed=sum(r['passed'] for r in moves if r['case']=='grid'), grid_total=9,
        limits_passed=sum(r['passed'] for r in moves if r['case']=='limit'), limits_total=6,
        invalid_targets_rejected=sum(rejected), invalid_targets_total=len(rejected),
        corners=corners, braking=brake, reset_move=reset_move, screenshot_motion=image_motion, screenshots=images,
        max_position_error_m=max(r['final_joint_error_m'] for r in moves),
        max_site_error_m=max(r['grasp_site_error_m'] for r in moves),
        max_penetration_m=max([r['max_penetration_m'] for r in rows+moves+corners+[brake, reset_move, image_motion]]),
        no_execution_pose_writes=True, local_hand='relative joint equalities, no grasp policy',
        training_started=False, grasp_success_rate=None,
        sources={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    write(a.output/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('passed','layouts_passed','grid_passed','limits_passed','max_position_error_m','max_penetration_m')}, indent=2))
    if not summary['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
