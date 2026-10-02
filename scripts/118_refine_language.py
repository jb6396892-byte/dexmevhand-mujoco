#!/usr/bin/env python3
"""Frozen train/validation/new-test refinement, preserving the retired test."""
import argparse
import datetime
import importlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.language_planner.contracts import canonical, compact, digest, read, write
from fromrealhand.language_planner.sft import load_rows, verify_protocol
from fromrealhand.language_planner.refinement import evaluate_system, system_passes, verify_refinement

BASE = importlib.import_module('111_language_study')
CONFIG = ROOT/'configs/stage6-refinement-v4.json'
ORIGINAL = ROOT/'data/processed/stage6_language_v1/dataset'


def phrase_rows(split, phrases, wrappers):
    rows = []
    for goal, texts in phrases.items():
        for index, text in enumerate(texts):
            for scene in ('first','second'):
                for variant, wrapper in enumerate(wrappers):
                    rows.append(dict(id='v4-%s-%s-%02d-%s-%d' % (split,goal,index,scene,variant),
                        family='v4-%s-%s-%02d' % (split,goal,index),scene=scene,
                        instruction=wrapper.format(text),response=canonical(goal,scene),
                        provenance=dict(source='manually authored Chinese contrast',split=split)))
    return rows


def prepare(study):
    verify_protocol(ROOT,ORIGINAL)
    cfg = read(CONFIG)
    train = load_rows(ORIGINAL/'train.jsonl') + load_rows(study.parent/'study_v3/supplement.jsonl')
    train += phrase_rows('train',read(ROOT/'configs/stage6-refinement-train.json'),['{}','请执行任务：{}'])
    splits = dict(train=train, **{s:phrase_rows(s,read(ROOT/('configs/stage6-refinement-'+s+'.json')),['{}'])
                                for s in ('validation','heldout')})
    for i, split in enumerate(('train','validation','heldout')):
        inputs = {(r['instruction'],r['scene']) for r in splits[split]}
        if len(inputs) != len(splits[split]):
            raise ValueError('Duplicate input in '+split)
        for other in ('train','validation','heldout')[:i]:
            if inputs.intersection((r['instruction'],r['scene']) for r in splits[other]):
                raise ValueError('Split leakage')
    previous = {r['instruction'] for s in ('train','validation','heldout') for r in load_rows(ORIGINAL/(s+'.jsonl'))}
    if previous.intersection(r['instruction'] for r in splits['heldout']):
        raise ValueError('New test duplicates a previous sentence')
    study.mkdir(parents=True,exist_ok=False)
    dataset = study/'dataset'; dataset.mkdir()
    for split, rows in splits.items():
        (dataset/(split+'.jsonl')).write_text(''.join(compact(r)+'\n' for r in rows),encoding='utf-8')
    shutil.copyfile(str(ORIGINAL/'heldout.jsonl'),str(dataset/'regression.jsonl'))
    shutil.copyfile(str(ORIGINAL/'feasibility.json'),str(dataset/'feasibility.json'))
    paths = [CONFIG,Path(__file__).resolve(),ROOT/'scripts/111_language_study.py',
        ROOT/'scripts/119_verify_refined_language.py',ROOT/'scripts/120_run_refined_language.py',
        ROOT/'src/fromrealhand/language_planner/instruction_guard.py',
        ROOT/'src/fromrealhand/language_planner/refinement.py',
        ROOT/'configs/stage6-system-prompt.txt']
    paths += [ROOT/('configs/stage6-refinement-'+s+'.json') for s in ('train','validation','heldout')]
    write(dataset/'protocol.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        data_sha256={p.name:digest(p) for p in dataset.iterdir() if p.is_file()},
        counts={s:len(rows) for s,rows in splits.items()},
        scope='New manually authored phrases, two nominal scenes, no new physical generalization',
        retired_test='Original 56 rows are regression only; known failure phrases informed development',
        acceptance=cfg['acceptance'], heldout_for_selection=False))
    write(study/'protocol.json',dict(config=cfg,dataset_protocol_sha256=digest(dataset/'protocol.json'),
        model_source_sha256=digest(study.parent/'model/source.json'),
        old_adapter_sha256=digest(study.parent/'study_v3/formal/adapter/adapter_model.safetensors'),
        update='Train contrastive intent boundaries and add veto-only instruction contract'))
    print('PREPARED', {s:len(rows) for s,rows in splits.items()},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','baseline','train','heldout','regression','verify'])
    parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
    args = parser.parse_args()
    root = args.root.resolve()
    if Path('/media/smgbro/shared') not in root.parents:
        raise ValueError('Use shared storage for language artifacts')
    study = root/'language/study_v4'
    if args.command == 'prepare':
        prepare(study); return
    cfg = verify_refinement(ROOT,study)
    BASE.DATASET = study/'dataset'
    BASE.evaluate = evaluate_system
    if args.command == 'verify':
        print('Frozen refinement verified'); return
    if args.command == 'baseline':
        BASE.score(study,cfg,'validation',study.parent/'study_v3/formal/adapter','previous-validation')
    elif args.command == 'train':
        BASE.train(study,cfg,'candidate')
    elif args.command == 'heldout':
        summary = read(study/'candidate/summary.json')
        if not summary['completed'] or not system_passes(read(study/'candidate-validation/report.json')['metrics'],cfg['acceptance']):
            raise ValueError('Development gate failed; new test stays sealed')
        for adapter,name in ((study.parent/'study_v3/formal/adapter','previous-heldout'),
                             (study/'candidate/adapter','candidate-heldout')):
            BASE.score(study,cfg,'heldout',adapter,name)
    else:
        BASE.score(study,cfg,'regression',study/'candidate/adapter','candidate-regression')


if __name__ == '__main__':
    main()
