#!/usr/bin/env python3
"""Build transparent readiness evidence from completed, immutable experiment outputs."""
import argparse
import importlib
import io
import json
import pickle
import shutil
import subprocess
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from v10_common import ROOT,digest
from fromrealhand.multivideo import phase_diagnostics

study=importlib.import_module('86_v14_contact_readiness')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/v14c-study.json')
    args=parser.parse_args()
    plan=args.config.resolve();p=json.loads(plan.read_text())
    run=ROOT/'data/processed'/('dual_video_'+p['version'])
    out=ROOT/'docs/presentation/v14/evidence'
    out.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((run/name).read_text())
    dev=read('development/summary.json');freeze=read('freeze.json')
    held=read('heldout/summary.json');receipt=read('heldout/receipt.json')
    smoke=read('dapg_smoke/summary.json');iterations=read('dapg_smoke/iterations.json')
    cfg_path=ROOT/'configs/v14-dapg.json';cfg=json.loads(cfg_path.read_text())
    limits=p['readiness'];checks={}
    checks['protocol_hashes']=all(x['protocol_sha256']==digest(plan) for x in (dev,freeze,held,receipt))
    checks['frozen_policy_hash']=digest(freeze['policy'])==freeze['policy_sha256']==smoke['frozen_policy_sha256']
    checks['code_hashes']=all(digest(ROOT/name)==sha for name,sha in freeze['code_sha256'].items())
    checks['committed_freeze']=receipt['freeze_sha256']==digest(out/'frozen-policy.json') and bool(receipt['commit'])
    checks['training_config_hash']=smoke['training_config_sha256']==digest(cfg_path)
    checks['development_complete']=len(dev['reports'])==35
    checks['unique_development']=len({(r['video'],r['case']) for r in dev['reports']})==35
    checks['development_each_video']=all(v['task']/v['count']>=limits['development_task_fraction_each_video']
                                        for v in dev['summary']['per_video'].values())
    checks['nominal_standard_and_half']=len(freeze['standard_nominal'])==len(freeze['half_step'])==2 and all(
        r['task_pass'] for r in freeze['standard_nominal']+freeze['half_step'])
    checks['heldout_complete']=len(held['reports'])==4*p['heldout']['count_per_video']
    checks['unique_heldout']=len({(r['video'],r['case'],r['method']) for r in held['reports']})==4*p['heldout']['count_per_video']
    gates,parent,admission,demos,refs,_=study.old.load_inputs()
    checks['gates_recomputed']=all(all(r[k]==v for k,v in study.old.gates(r['report'],gates).items())
                                 for r in dev['reports']+held['reports']+smoke['post_nominal'])
    for video in ('first','second'):
        rows=[r for r in held['reports'] if r['video']==video and r['method']=='predictive']
        checks[video+'_heldout_count']=len(rows)==p['heldout']['count_per_video']
        checks[video+'_heldout_task']=bool(rows) and np.mean([r['task_pass'] for r in rows])>=limits['heldout_task_fraction_each_video']
        checks[video+'_heldout_lift']=bool(rows) and np.mean([r['lift_pass'] for r in rows])>=limits['heldout_stable_lift_fraction_each_video']
    checks['smoke_20_updates']=smoke['completed_iterations']==smoke['nonzero_updates']==limits['finite_nonzero_dapg_updates']==len(iterations)
    checks['smoke_finite_kl']=smoke['finite'] and all(np.isfinite([r['measured_kl'],r['parameter_delta_l2']]).all()
        and r['measured_kl']<=cfg['max_measured_kl'] for r in iterations)
    checks['smoke_post_nominal']=len(smoke['post_nominal'])==2 and all(r['task_pass'] for r in smoke['post_nominal'])
    checks['mean_conversion_exact']=freeze['mean_action_conversion_error']<=1e-8
    audited=dev['reports']+freeze['half_step']+[r for r in held['reports'] if r['method']=='predictive']
    checks['branch_matches_live']=all(r['audit']['prediction_error_max']<=1e-7 for r in audited)
    # Deterministic simulator optimizations must preserve the already evaluated pilot.
    pilot=read('pilot/summary.json');pilot_checks=[]
    for row in pilot['reports']:
        other=next(r for r in dev['reports'] if (r['video'],r['case'])==(row['video'],row['case']))
        errors={k:abs(row['report'][k]-other['report'][k]) for k in
                ('max_hand_scene_penetration_m','final_distance_m')}
        pilot_checks.append(dict(video=row['video'],case=row['case'],errors=errors))
    checks['pilot_development_repeat']=all(max(r['errors'].values())<=1e-10 for r in pilot_checks)
    images=[]
    for folder,step in [('before',753),('after',753),('after',1449)]:
        source=run/'renders'/folder/('step_%04d.jpg'%step)
        pixels=np.asarray(Image.open(str(source)))
        image_out=out/(folder+'_'+source.name)
        shutil.copy2(str(source),str(image_out))
        replay=read('renders/'+folder+'/comparison.json')
        images.append(dict(file=image_out.name,shape=list(pixels.shape),pixel_std=float(pixels.std()),
                           replay=replay,sha256=digest(image_out)))
    checks['screenshots_nonblank']=all(r['pixel_std']>10 for r in images)
    checks['screenshot_physics_replay']=all(r['replay']['replay_max_observation_error']<=1e-8 for r in images)
    phase_rows=[]
    cp=pickle.loads(Path(freeze['policy']).read_bytes())
    for entry in admission['reports']:
        video=parent['videos'][entry['video_id']]
        key=video['name']+'/'+entry['name']
        demo=pickle.loads((run/'development'/key/'rollout.pkl').read_bytes())['video_faithful']
        phases=phase_diagnostics(demo,cp['references'][video['name']],video,np.load(entry['geometry']),demos[key])
        phase_rows.append(dict(video=video['name'],case=entry['name'],phases=phases))
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    print(stream.getvalue()[-1600:])
    checks['tests']=result.wasSuccessful() and len(result.skipped)==0
    subprocess.check_call(['git','diff','--check'],cwd=str(ROOT))
    ready=dict(ready_for_long_training=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},
        failed_checks=[k for k,v in checks.items() if not v],policy_sha256=freeze['policy_sha256'],
        training_config_sha256=digest(cfg_path),protocol_sha256=digest(plan),
        development=dev['summary'],heldout=held['summary'],smoke=smoke,long_training_started=False,
        scope='Controlled long training from frozen BC plus predictive filter, not deployment, new-video generalization, or standalone policy safety.')
    study.write_json(run/'readiness.json',ready)
    for name,value in [('readiness.json',ready),('development.json',dev),('heldout.json',held),
        ('heldout-receipt.json',receipt),('dapg-smoke.json',smoke),('dapg-iterations.json',iterations),
        ('screenshots.json',images),('pilot-repeat.json',pilot_checks),('phases.json',phase_rows),
        ('verification.json',dict(tests=result.testsRun,skipped=len(result.skipped),success=result.wasSuccessful(),
             checks={k:bool(v) for k,v in checks.items()},new_screenshots=len(images),long_training=False))]:
        study.write_json(out/name,value)
    print(json.dumps(dict(ready=ready['ready_for_long_training'],failed=ready['failed_checks'],
                         development=dev['summary'],heldout=held['summary']),indent=2))


if __name__=='__main__': main()
