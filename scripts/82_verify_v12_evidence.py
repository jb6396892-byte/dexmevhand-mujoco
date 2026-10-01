#!/usr/bin/env python3
"""Verify hashes, gates, tests and nonblank rendering before publication."""
import importlib
import io
import json
import subprocess
import unittest
import cv2
import numpy as np

study=importlib.import_module('77_run_v12_study')


def main():
    plan,parent,admission,_,_,_=study.load_inputs()
    root,run=study.ROOT,study.RUN
    receipt=root/'docs/presentation/v12/evidence/frozen-policy.json'
    freeze=json.loads(receipt.read_text());test=json.loads((run/'heldout/summary.json').read_text())
    record=json.loads((run/'heldout/receipt.json').read_text())
    committed=subprocess.check_output(['git','show',record['commit']+':docs/presentation/v12/evidence/frozen-policy.json'],cwd=str(root))
    assert committed==receipt.read_bytes()
    assert record['frozen_sha256']==study.digest(receipt)
    assert freeze['protocol_sha256']==study.digest(study.PLAN)==test['protocol_sha256']
    assert freeze['guard_amendment_sha256']==study.digest(root/'configs/v12-contact-guard-amendment.json')
    for c in freeze['candidates']: assert study.digest(c['path'])==c['sha256']
    previous=json.loads((root/'data/processed/dual_video_v11/learning/phase_bc_150.json').read_text())
    assert study.digest(previous['policy'])==previous['policy_sha256']
    keys={(r['method'],r['video'],r['case']) for r in test['reports']}
    assert len(keys)==len(test['reports'])==80
    for row in test['reports']:
        for key,value in study.gates(row['report'],plan).items(): assert row[key]==value
    for method,summary in test['summary'].items():
        assert summary==study.summarize([r for r in test['reports'] if r['method']==method])
    feature_files=list((run/'reference_data').glob('*/*.json'))
    assert len(feature_files)==35
    assert all(json.loads(p.read_text())['replay_error']==0. for p in feature_files)
    for mode in ('aligned_bc','contact_reference_bc'):
        data=json.loads((run/mode/'input.json').read_text())
        for row in data['phase_audit']: np.testing.assert_allclose(row['phase_mass'],plan['phase_mass'])
    smoke=json.loads((run/'dapg_smoke/summary.json').read_text())
    iterations=json.loads((run/'dapg_smoke/iterations.json').read_text())
    assert smoke['completed_iterations']==len(iterations)==20
    assert smoke['finite'] and not smoke['heldout_evaluated']
    assert all(np.isfinite([r['parameter_delta_l2'],r['kl']]).all() for r in iterations)
    assert all(c['observation_error']==0 and c['action_error']==0 for c in smoke['preflight']['checks'])
    images=[]
    for folder in (run/'renders').iterdir():
        if not folder.is_dir(): continue
        if (folder/'comparison.json').exists():
            assert json.loads((folder/'comparison.json').read_text())['replay_max_observation_error']<1e-8
        for path in folder.glob('step_*.jpg'):
            array=cv2.imread(str(path));assert array is not None and float(array.std())>5.
            images.append(dict(path=str(path.relative_to(root)),std=float(array.std())))
    assert len(images)>=16
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(unittest.defaultTestLoader.discover(str(root/'tests')))
    print(stream.getvalue()[-1500:])
    assert result.wasSuccessful()
    subprocess.check_call(['git','diff','--check'],cwd=str(root))
    verification=dict(date='2026-10-02',tests=result.testsRun,skipped=len(result.skipped),test_success=True,
         unique_heldout_rollouts=len(keys),expert_feature_replays=len(feature_files),
         frozen_policy_hashes_verified=True,strict_and_task_gates_recomputed=True,
         exact_phase_mass_verified=True,dapg_iterations=20,
         nonzero_dapg_updates=smoke['nonzero_updates'],images=images,
         interactive_desktop_window_retested=False,long_training=False)
    study.write_json(root/'docs/presentation/v12/evidence/verification.json',verification)
    print('VERIFIED',json.dumps({k:v for k,v in verification.items() if k!='images'}))


if __name__=='__main__': main()
