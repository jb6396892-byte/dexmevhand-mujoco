#!/usr/bin/env python3
"""Two-video translated local grasp and optional continuous navigation handoff."""
import argparse
import json
from pathlib import Path
import numpy as np
from hierarchy_common import ROOT,write
from fromrealhand.tabletop.random_task import RandomTask
from fromrealhand.whole_table.local_adapter import translate_initial_scene,MotionBridge
from fromrealhand.whole_table.navigation_runner import navigate,capture

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--video',choices=['first','second'],required=True)
p.add_argument('--shift',type=float,nargs=2,default=[.2,0])
p.add_argument('--transit',action='store_true')
p.add_argument('--carry-goal',type=float,nargs=3)
p.add_argument('--checkpoint',type=Path,default=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4/structured-bc/candidate.pt'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads((ROOT/'configs/adroit-navigation-v1.json').read_text())
task=RandomTask(ROOT,a.video,a.checkpoint,ROOT/'configs/tabletop-random-v6.json')
delta=np.r_[a.shift,0.];receipt=dict(video=a.video,shift_m=delta.tolist(),transit_requested=a.transit)
def setup(env,mesh,scene,poses):
    transformed=translate_initial_scene(env,delta,poses,scene)
    mug=env.sim.model.body_name2id('mug_0')
    vertices=mesh['vertices']@env.sim.data.body_xmat[mug].reshape(3,3).T+env.sim.data.body_xpos[mug]
    limits=np.array(cfg['table_size_xy_m'])/2-cfg['edge_clearance_m']
    if np.any(np.abs(vertices[:,:2])>limits):
        raise ValueError('Translated cup footprint outside tabletop support')
    # The layout dictionary is shared with the task, so its goal remains in world coordinates.
    scene['random_layout']['goal_world_m']=(np.array(scene['random_layout']['goal_world_m'])+delta).tolist()
    for item in scene['random_layout']['objects']:
        if item['name']=='mug':item['xy']=(np.array(item['xy'])+delta[:2]).tolist()
    bridge=MotionBridge(env,cfg)
    receipt['pregrasp_center_m']=bridge.position().tolist()
    if a.transit:
        scene['initial_hand_repositioned']=True
        goal=bridge.position().copy()
        env.sim.data.qpos[bridge.tcols]=bridge.to_joints(cfg['home_m'])
        env.sim.data.qvel[:30]=0;env.sim.forward()
        bridge.target=np.array(cfg['home_m']);bridge.hand_envelope=bridge.hand_shapes()-bridge.position()
        for _ in range(int(cfg['settling_s']/cfg['timestep_s'])):bridge.step(bridge.target)
        capture(bridge,a.output/'navigation-start.png')
        try:
            receipt['navigation']=navigate(bridge,goal)
        except ValueError as error:
            from fromrealhand.whole_table.navigation import inflated_boxes,segments_clear
            receipt['navigation_failure']=str(error)
            receipt['goal_margin_conflicts']=[]
            for obstacle in bridge.obstacles():
                for part in bridge.hand_envelope:
                    if not segments_clear(goal,goal,inflated_boxes(part,[obstacle],cfg['clearance_m']))[0]:
                        receipt['goal_margin_conflicts'].append(dict(obstacle=obstacle,hand_part_at_goal=(part+goal).tolist()))
            raise
        finally:
            bridge.restore();write(a.output/'adapter.json',receipt)
    else:bridge.restore()
    write(a.output/'adapter.json',receipt)
    return transformed
def finish(env,controller,rows,initial,goal):
    bridge=MotionBridge(env,cfg,held_action=controller.last_action)
    capture(bridge,a.output/'lift.png');bridge.restore()
    if a.carry_goal:
        from fromrealhand.whole_table.loaded_motion import carry
        result=carry(env,controller.last_action,np.asarray(a.carry_goal),cfg)
        write(a.output/'carry.json',result)
        bridge=MotionBridge(env,cfg,held_action=controller.last_action)
        capture(bridge,a.output/'carry.png');bridge.restore()
        if not result['passed']:raise RuntimeError('carry_acceptance_failed')
        return result
    return dict(scope='local grasp and lift; no long-distance carry',learned_grasp=True)
try:
    report=task.run(30,a.output/'local',goal=[0,0,.16],count=0,cup_xy=[0,0],stop_skill='lift',
                    scene_adapter=setup,completion=finish)
    write(a.output/'summary.json',dict(passed=report['passed'],reason=report['reason'],completed=report['completed'],
        video=a.video,shift_m=delta.tolist(),continuous_navigation=a.transit,
        initial_hand_repositioned_only_before_execution=a.transit,training_started=False))
    print(json.dumps({k:report[k] for k in ('passed','reason','completed','final')},indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)
finally:task.close()
