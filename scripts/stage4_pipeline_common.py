"""Artifact export and actual env.step validation for the version-2 skill pipeline."""
import pickle
from pathlib import Path
import numpy as np
from stage4_common import ROOT, digest, experiment, measure, restore_once, save_pickle
from fromrealhand.skill_pipeline import GuardedSkill


def export_pieces(folder,entry,result,source_sha):
    dest = folder/entry['trajectory']
    dest.mkdir(parents=True,exist_ok=False)
    for segment in entry['segments']:
        lo,hi = segment['start'],segment['stop']
        piece = {name:result['demo'][name][lo:hi] for name in ('observations','actions','rewards','sim_data')}
        piece.update(model_data=result['demo']['model_data'],physics_model=result['demo']['physics_model'],
            initial_snapshot=result['snapshots'][lo],terminal_snapshot=result['snapshots'][hi],
            terminal_observation=result['demo']['observations'][hi] if hi<entry['horizon'] else result['terminal_observation'],
            initial_metrics=result['initial'] if lo==0 else result['rows'][lo-1],
            metrics=result['rows'][lo:hi],expected_post_states=result['post_states'][lo:hi],
            metadata=dict(segment,trajectory=entry['trajectory'],video=entry['video'],dt=entry['dt'],
                          source_demo_sha256=source_sha,action_semantics='normalized_once'))
        path = dest/(segment['skill']+'.pkl')
        save_pickle(path,piece)
        segment.update(artifact=str(path.relative_to(folder)),sha256=digest(path))


def load_pieces(folder,entry):
    if digest(entry['geometry']) != entry['geometry_sha256']:
        raise ValueError('Geometry hash changed')
    pieces = []
    for segment in entry['segments']:
        path = Path(folder)/segment['artifact']
        if digest(path)!=segment['sha256']:
            raise ValueError('Skill artifact changed')
        pieces.append(pickle.loads(path.read_bytes()))
    return pieces


def replay(entry,pieces,config,offset=None):
    exp = experiment(entry['geometry'],pieces[0])
    expected = np.concatenate([p['expected_post_states'] for p in pieces])
    errors,rows,contracts = [],[],[]
    audit_index = [0]
    perturbed = offset is not None and bool(np.any(offset))
    def read():
        row = measure(exp,audit_index[0])
        audit_index[0] = len(exp.audit)
        return row
    try:
        restore_once(exp.env,pieces[0]['initial_snapshot'])
        if perturbed:
            # A new test episode's initial condition, never a mid-execution correction.
            exp.env.sim.data.qpos[30:32] += np.asarray(offset)
            exp.env.sim.data.qacc_warmstart[:] = 0.
            exp.env.sim.forward()
        row = read()
        for piece in pieces:
            guard = GuardedSkill(piece['metadata']['skill'],config,row)
            if guard.status != 'failed':
                for action in piece['actions']:
                    exp.env.step(action)
                    row = read()
                    current = np.r_[exp.env.sim.data.qpos,exp.env.sim.data.qvel]
                    errors.append(float(np.max(np.abs(current-expected[len(rows)]))))
                    rows.append(row)
                    guard.update(row)
                    if guard.status in ('failed','timeout'):
                        break
            contracts.append(dict(skill=guard.skill,status=guard.status,reason=guard.reason,steps=guard.steps))
            if guard.status != 'success':
                break
        complete = len(rows)==len(expected)
        error = max(errors or [0.])
        passed = bool(complete and all(c['status']=='success' for c in contracts)
                      and (perturbed or error <= config['replay_state_tolerance']))
        return dict(trajectory=entry['trajectory'],skills=[p['metadata']['skill'] for p in pieces],
            offset_m=[0.,0.] if offset is None else list(offset),passed=passed,complete=complete,
            state_replay_error=None if perturbed else error,state_comparison_required=not perturbed,
            steps=len(rows),expected_steps=len(expected),contracts=contracts,
            max_penetration_m=max([r['scene_penetration_m'] for r in rows]+[row['scene_penetration_m']]),
            final_bottom_m=row['bottom_m'],final_goal_distance_m=row['target_distance_m'],
            initialization_count=1,state_writes_during_execution=0)
    finally:
        exp.env.close()
