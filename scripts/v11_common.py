"""v11 protocol and data boundaries; no heldout simulation during development."""
import json
import pickle
from pathlib import Path
from v10_common import ROOT, digest, protocol as parent_protocol

RUN = ROOT/'data/processed/dual_video_v11'


def protocol():
    parent,sha=parent_protocol()
    path=ROOT/'configs/v11-study.json'
    study=json.loads(path.read_text())
    if sha != study['parent_protocol_sha256']:
        raise ValueError('Changed parent protocol')
    old=ROOT/'data/processed/dual_video_v10/learning/residual_bc/epoch_050.pickle'
    if digest(old) != study['parent_residual_sha256']:
        raise ValueError('Changed frozen parent policy')
    return study,parent,digest(path)


def heldout_cases(study):
    spec=study['heldout']['axis_cases_per_video'];cases=[]
    for axis in range(2):
        for value in spec['cup_xy_offsets_m']:
            delta=[0.,0.,0.];delta[axis]=value
            cases.append(dict(name='cup_%d_%+.5f'%(axis,value),cup_offset_m=delta,cup_yaw_deg=0.,goal_offset_m=[0.,0.,0.]))
        for value in spec['goal_xy_offsets_m']:
            delta=[0.,0.,0.];delta[axis]=value
            cases.append(dict(name='goal_%d_%+.5f'%(axis,value),cup_offset_m=[0.,0.,0.],cup_yaw_deg=0.,goal_offset_m=delta))
    for value in spec['cup_yaw_deg']:
        cases.append(dict(name='yaw_%+.2f'%value,cup_offset_m=[0.,0.,0.],cup_yaw_deg=value,goal_offset_m=[0.,0.,0.]))
    cases+=study['heldout']['combined_cases_per_video']
    if len(cases)!=study['heldout']['count_per_video']:
        raise ValueError('Heldout count differs from registration')
    return cases


def development_cases(parent):
    meta=json.loads((ROOT/'data/processed/dual_video_v10/training_input/metadata.json').read_text())
    cases=[]
    for e in meta['trajectories']:
        demos=pickle.loads(Path(e['dataset']).read_bytes())
        if digest(e['dataset'])!=e['dataset_sha256'] or digest(e['geometry'])!=e['geometry_sha256']:
            raise ValueError('Changed admitted input')
        cases.append(dict(name=e['name'],video_id=e['video_id'],geometry=e['geometry'],seed=e['seed'],
                          demo=demos[e['name']],origin='v10_admitted'))
    old=json.loads((ROOT/'data/processed/dual_video_v10/heldout/summary.json').read_text())
    definitions={c['name']:c for c in parent['heldout_per_video']}
    for row in old['reports']:
        if row['method']!='residual_bc' or row['full_pass']: continue
        video=next(v for v in parent['videos'] if v['name']==row['video'])
        cases.append(dict(name='former_'+row['case'],video_id=video['id'],seed=0,case=definitions[row['case']],
                          geometry=str(Path(row['rollout']).parents[1]/'geometry.npz'),origin='v10_test_now_development'))
    if len(cases)!=35: raise ValueError('Expected 27 admitted plus 8 known failures')
    return cases
