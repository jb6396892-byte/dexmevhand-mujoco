#!/usr/bin/env python3
"""Publish compact evidence; retain all failed episodes in aggregate results."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from hierarchy_common import ROOT, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('/media/smgbro/shared/visual_grasp/dual-learn-v4'))
    p.add_argument('--output',type=Path,default=ROOT/'docs/presentation/tabletop_learning_v4/evidence')
    p.add_argument('--qt-results',default='qt-structured-final')
    p.add_argument('--expert-infrastructure-retry',type=Path)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    checks=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],
        cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    (a.output/'regression-output.txt').write_text(
        '\n'.join(line.rstrip() for line in checks.stdout.splitlines())+'\n')
    write(a.output/'software-checks.json',dict(command=[sys.executable,'-m','unittest','discover','-s','tests'],
        returncode=checks.returncode,passed=checks.returncode==0))
    if checks.returncode: raise ValueError('Regression tests failed')
    evaluations={}
    for name in ('dev-shared-60','dev-routed-60','heldout-routed','heldout-expert',
                 'dev-structured','regression-structured','fresh-structured','fresh-expert',
                 'dev-cadence','regression-cadence','final-structured','final-expert'):
        evaluation=read(a.run/name/'evaluation.json')
        if not evaluation['complete']: raise ValueError('Evaluation still running: '+name)
        if name=='heldout-expert' and a.expert_infrastructure_retry:
            retry=read(a.expert_infrastructure_retry)
            if not retry['complete'] or retry['checkpoint'] is not None:
                raise ValueError('Invalid expert infrastructure retry')
            replaced=[]
            for replacement in retry['records']:
                indices=[i for i,r in enumerate(evaluation['records'])
                         if (r['video'],r['seed'])==(replacement['video'],replacement['seed'])]
                if len(indices)!=1: raise ValueError('Retry must match an existing case')
                i=indices[0]; original=evaluation['records'][i]; folder=Path(original['output'])
                report=read(folder/'report.json')
                if (report['steps']!=0 or original['reason']!='Vision worker disconnected'
                        or 'CUDA out of memory' not in (folder/'vision-stderr.log').read_text()):
                    raise ValueError('Only proven zero-action infrastructure failures may be retried')
                replaced.append(dict(original=original,retry=replacement,reason='zero_action_cuda_oom'))
                evaluation['records'][i]=replacement
            evaluation['infrastructure_retries']=replaced
            evaluation['passed']=sum(r['passed'] for r in evaluation['records'])
        evaluations[name]=evaluation
        write(a.output/(name+'.json'),evaluation)
    write(a.output/'freeze.json',read(a.run/'freeze.json'))
    write(a.output/'freeze-structured.json',read(a.run/'freeze-structured.json'))
    write(a.output/'freeze-final.json',read(a.run/'freeze-final.json'))
    stage_errors={}
    for name in ('dev-shared-60','dev-routed-60','dev-structured','dev-cadence'):
        for row in evaluations[name]['records']:
            trace=read(Path(row['output'])/'trace.json')
            stages={}
            for stage in ('reach','grasp','lift','transport'):
                selected=[q for q in trace if q['phase']==stage]
                if selected:
                    errors=[q['expert_action_error_max'] for q in selected]
                    stages[stage]=dict(steps=len(selected),max_action_error=max(errors),
                        mean_action_rms=float(np.mean([q['expert_action_error_rms'] for q in selected])),
                        final_target_distance_m=selected[-1]['target_distance_m'],
                        first_error_over_001_step=next((q['step'] for q in selected if q['expert_action_error_max']>.01),None))
            stage_errors[name+'/'+row['video']]=stages
    write(a.output/'development-stage-errors.json',stage_errors)
    from fromrealhand.tabletop.learned_control import LearnedControl
    training={}
    for name in ('shared-60','routed-60'):
        policy=LearnedControl(a.run/name/'candidate.pt')
        training[name]=dict(receipt=read(a.run/name/'receipt.json'),
            parameters=sum(p.numel() for p in policy.network.parameters()),
            final_loss=read(a.run/name/'losses.json')[-1])
        write(a.output/(name+'-losses.json'),read(a.run/name/'losses.json'))
    write(a.output/'training.json',training)
    write(a.output/'structured-training.json',read(a.run/'structured-bc/receipt.json'))
    quality=read(a.run/a.qt_results/'quality.json'); write(a.output/'qt-quality.json',quality)
    for case in ('first-transport','second-transport','first-lift','second-lift','second-stop','locked','instruction-rejected'):
        write(a.output/('qt-'+case+'.json'),read(a.run/a.qt_results/case/'summary.json'))
    # A small set of actual simulator and Qt images, no synthetic illustrations.
    for source,target in [
            (a.run/'dev-cadence/second-seed-3/grasp.png','second-learned-grasp.png'),
            (a.run/'dev-cadence/second-seed-3/transport.png','second-learned-transport.png'),
            (a.run/a.qt_results/'second-transport/qt-result.png','learned-qt.png')]:
        shutil.copy2(str(source),str(a.output/target))
    summary={name:dict(passed=e['passed'],total=e['total'],
        max_penetration_mm=1000*max(r['max_penetration_m'] or 0 for r in e['records']),
        mean_actual_error_mm=1000*float(np.mean([r['actual_goal_error_m'] for r in e['records'] if r['actual_goal_error_m'] is not None])),
        by_video={v:dict(passed=sum(r['passed'] for r in e['records'] if r['video']==v),
                       total=sum(r['video']==v for r in e['records'])) for v in ('first','second')})
        for name,e in evaluations.items()}
    summary['qt_passed']=all(value for key,value in quality.items() if key!='training_started')
    summary['all_frozen_learning_cases_passed']=evaluations['final-structured']['passed']==evaluations['final-structured']['total']
    summary['regression_passed']=evaluations['regression-cadence']['passed']==evaluations['regression-cadence']['total']
    summary['candidate_development_passed']=evaluations['dev-cadence']['passed']==evaluations['dev-cadence']['total']
    summary['software_checks_passed']=checks.returncode==0
    summary['delivery_passed']=all(summary[key] for key in ('qt_passed','all_frozen_learning_cases_passed',
        'regression_passed','candidate_development_passed','software_checks_passed'))
    summary['selected_model']='structured-bc/candidate.pt'
    summary['previous_mlp_failures_retained']=True
    summary['learning_algorithm']='Supervised behavior cloning; no new reinforcement-learning updates'
    summary['scope']='Same mug/camera; small layout shifts and reference-conditioned goals, not hardware transfer'
    write(a.output/'summary.json',summary)
    print(summary)
    if not summary['delivery_passed']: raise SystemExit(1)


if __name__=='__main__': main()
