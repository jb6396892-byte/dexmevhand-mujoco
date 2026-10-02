#!/usr/bin/env python3
"""Archive stage6 v4 evidence and a shared-storage delivery, never stage3 weights."""
import argparse
import contextlib
import io
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.language_planner.contracts import digest, read, write
from fromrealhand.language_planner.refinement import deployment_matches
from fromrealhand.language_planner.guard_revision import verify_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
    args = parser.parse_args()
    language = args.root.resolve()/'language'
    study = language/'study_v4_guard2'
    verify_revision(ROOT,study)
    acceptance = read(study/'acceptance/summary.json')
    if acceptance['model_acceptance_passed'] and not deployment_matches(acceptance,study,language/'model'):
        raise ValueError('Accepted artifacts changed')
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(
            unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    quality = dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                   skipped=len(result.skipped),passed=result.wasSuccessful())
    write(study/'quality.json',quality)
    (study/'quality.log').write_text(log.getvalue(),encoding='utf-8')
    if not result.wasSuccessful():
        print(log.getvalue()); raise ValueError('Regression tests failed')
    destination = ROOT/'docs/presentation/stage6/evidence/study-v4-guard2'
    destination.mkdir(parents=True,exist_ok=False)
    sources = dict(protocol=study/'protocol.json',dataset_protocol=study/'dataset/protocol.json',
        training=language/'study_v4/candidate/summary.json',acceptance=study/'acceptance/summary.json',
        physical=study/'acceptance/physical.json',negative_guards=study/'acceptance/negative_guards.json',
        quality=study/'quality.json')
    for split in ('previous-heldout','candidate-heldout','candidate-regression'):
        sources[split] = study/split/'report.json'
        sources[split+'-protocol'] = study/split/'protocol.json'
    for split in ('previous-validation','candidate-validation','previous-heldout','candidate-heldout','candidate-regression'):
        sources['retired-v4-'+split] = language/'study_v4'/split/'report.json'
    sources['retired-v4-protocol'] = language/'study_v4/protocol.json'
    sources['retired-v4-dataset-protocol'] = language/'study_v4/dataset/protocol.json'
    for name,path in sources.items():
        shutil.copyfile(str(path),str(destination/(name+'.json')))
    smoke = []
    for folder in sorted((study/'instructions').iterdir()):
        generation = read(folder/'generation.json')
        item = dict(path=str(folder),generation=generation)
        if (folder/'execution.json').exists():
            item['execution'] = read(folder/'execution.json')
            item['physics'] = read(folder/'physics/report.json')
        smoke.append(item)
    write(destination/'entrypoint-smoke.json',smoke)
    image = study/'acceptance/second-transport/final.jpg'
    if image.exists():
        import cv2
        pixels = cv2.imread(str(image))
        if pixels is None or pixels.std()<5:
            raise ValueError('Blank physical screenshot')
        shutil.copyfile(str(image),str(destination/'second-transport.jpg'))
    delivery = language/'delivery-study_v4_guard2'
    delivery.mkdir(exist_ok=False)
    shutil.copytree(str(study/'dataset'),str(delivery/'dataset'))
    for name in ('docs/STAGE6_REFINEMENT.md','docs/STAGE6_LANGUAGE.md',
                 'docs/presentation/stage6/REFINEMENT_RESULTS.md','docs/run_logs/2026-10-02-stage6-refinement.md'):
        target = delivery/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(str(ROOT/name),str(target))
    shutil.copytree(str(destination),str(delivery/'docs/presentation/stage6/evidence/study-v4-guard2'))
    shutil.copyfile(str(study/'protocol.json'),str(delivery/'study-protocol.json'))
    write(delivery/'model-index.json',dict(base=str(language/'model'),adapter=str(study/'candidate/adapter'),
        reports=str(study),stage3_included=False,accepted=acceptance['model_acceptance_passed'],
        automatic_execution_scope=acceptance['scope'],original_project=str(ROOT),
        documentation=str(delivery/'docs/STAGE6_REFINEMENT.md'),
        source_sha256=read(study/'dataset/protocol.json')['sha256']))
    write(destination/'files.json',dict(sha256={p.name:digest(p) for p in destination.iterdir() if p.is_file()},
        shared_delivery=str(delivery),weights_uploaded=False))
    print('PUBLISHED',destination,'SHARED',delivery)


if __name__=='__main__':
    main()
