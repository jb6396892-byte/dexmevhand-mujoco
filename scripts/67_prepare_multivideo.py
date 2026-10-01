#!/usr/bin/env python3
"""Prepare balanced, independently timed dual-video development data, never heldout."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from v10_common import ROOT, protocol, digest, reference_demo
from fromrealhand.multivideo import trajectory_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/dual_video_v10/training_input')
    args = parser.parse_args()
    study,sha = protocol()
    admissions = [ROOT/'data/processed/seq_dexycb_001/learning_v6/demonstrations/admission.json',
                  ROOT/'data/processed/dual_video_v10/development/admission.json']
    xs,ys,rs,weights,video_ids,entries = [],[],[],[],[],[]
    offset = 0
    for video,admission_path in zip(study['videos'],admissions):
        admission = json.loads(admission_path.read_text())
        if video['id'] == 0:
            assert admission['training_ready']
        else:
            assert admission['completed'] and admission['protocol_sha256'] == sha
        demo_path = Path(admission['demo'])
        assert digest(demo_path) == admission['demo_sha256']
        demos = pickle.loads(demo_path.read_bytes())
        reports = {r.get('name',r.get('case',{}).get('name')):r for r in admission['reports']}
        reference = reference_demo(video)['actions']
        for name,demo in demos.items():
            report = reports[name]
            assert report['admitted']
            geometry = Path(report['geometry'])
            if video['id'] == 0:
                assert digest(geometry) == report['geometry_sha256']
            g = np.load(geometry)
            x,y,r = trajectory_arrays(demo,reference,video,g)
            xs.append(x);ys.append(y);rs.append(r)
            weights.append(np.full(len(x),.5/len(demos)/len(x)))
            video_ids.append(np.full(len(x),video['id'],dtype=int))
            seed = report['report']['seed'] if video['id'] == 0 else 0
            entries.append(dict(name=name,video=video['name'],video_id=video['id'],start=offset,end=offset+len(x),
                                geometry=str(geometry),geometry_sha256=digest(geometry),seed=seed,
                                dataset=str(demo_path),dataset_sha256=digest(demo_path)))
            offset += len(x)
    args.output.mkdir(parents=True,exist_ok=False)
    path = args.output/'dataset.npz'
    np.savez_compressed(path,features=np.concatenate(xs),actions=np.concatenate(ys),residuals=np.concatenate(rs),
                        weights=np.concatenate(weights),video_ids=np.concatenate(video_ids))
    metadata = dict(protocol_sha256=sha,dataset_sha256=digest(path),frames=offset,feature_dim=84,
                    feature_layout='39 native observations + 36 velocities + 6 rotation + 1 phase + 2 skill onehot',
                    trajectories=entries,admission_sha256={str(p):digest(p) for p in admissions},
                    heldout_used=False,independent_real_sequences=2,
                    sampling='Equal video mass; equal trajectory mass within each video')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(frames=offset,trajectories=len(entries),per_video={v['name']:sum(e['video']==v['name'] for e in entries) for v in study['videos']})))


if __name__=='__main__': main()
