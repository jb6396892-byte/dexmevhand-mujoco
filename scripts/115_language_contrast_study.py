#!/usr/bin/env python3
"""Train-only contrast augmentation after a failed pilot; keep the test sealed."""
import argparse
import datetime
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.language_planner.contracts import canonical, compact, digest, read, write
from fromrealhand.language_planner.sft import load_rows, verify_protocol

STUDY = importlib.import_module('111_language_study')
PHRASES = ROOT/'configs/stage6-contrast-phrases.json'


def contrast_rows(phrases):
    rows = []
    for goal, texts in phrases.items():
        for index, phrase in enumerate(texts):
            for scene in ('first', 'second'):
                for variant, wrapper in enumerate(('{}', '请执行任务：{}')):
                    rows.append(dict(id='contrast-%s-%02d-%s-%d' % (goal, index, scene, variant),
                        family='contrast-%s-%02d' % (goal, index), instruction=wrapper.format(phrase),
                        scene=scene, response=canonical(goal, scene),
                        provenance=dict(language='train-only manually authored contrast',
                            motivation='development validation: final intent, stop and unsupported action')))
    return rows


def freeze(folder):
    verify_protocol(ROOT, STUDY.DATASET)
    cfg = read(STUDY.CONFIG)
    rows = contrast_rows(read(PHRASES))
    # Only an exact-overlap guard; no test predictions or scores are consulted.
    excluded = {r['instruction'] for split in ('validation', 'heldout')
                for r in load_rows(STUDY.DATASET/(split+'.jsonl'))}
    if any(r['instruction'] in excluded for r in rows):
        raise ValueError('Supplement overlaps a protected split; no training started')
    paths = [Path(__file__).resolve(), PHRASES, STUDY.CONFIG, Path(STUDY.__file__).resolve()]
    expected = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if not (folder/'protocol.json').exists():
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'supplement.jsonl').write_text(''.join(compact(r)+'\n' for r in rows), encoding='utf-8')
        write(folder/'protocol.json', dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            config=cfg, sha256=expected, supplement_sha256=digest(folder/'supplement.jsonl'),
            original_train_rows=len(load_rows(STUDY.DATASET/'train.jsonl')), additional_train_rows=len(rows),
            dataset_protocol_sha256=digest(STUDY.DATASET/'protocol.json'),
            rationale='Pilot v2 failed. Add contrast instructions only; same hyperparameters and thresholds.',
            validation_used_for_development=True, heldout_predictions_seen=False,
            heldout_content_use='Automatic exact-overlap guard only',
            previous_pilot_gate_sha256=digest(folder.parent/'study_v2/pilot-gate.json')))
    protocol = read(folder/'protocol.json')
    if protocol['sha256'] != expected or digest(folder/'supplement.jsonl') != protocol['supplement_sha256']:
        raise ValueError('Frozen contrast study changed')
    if protocol['dataset_protocol_sha256'] != digest(STUDY.DATASET/'protocol.json'):
        raise ValueError('Original dataset changed')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'baseline', 'pilot', 'formal', 'heldout'])
    parser.add_argument('--root', type=Path, default=Path('/media/smgbro/shared/lora'))
    args = parser.parse_args()
    root = args.root.resolve()
    if Path('/media/smgbro/shared') not in root.parents:
        raise ValueError('New language artifacts belong on shared storage')
    folder = root/'language/study_v3'
    cfg = freeze(folder)
    original_load = STUDY.load_rows
    def augmented_load(path):
        rows = original_load(path)
        if Path(path).resolve() == (STUDY.DATASET/'train.jsonl').resolve():
            rows += original_load(folder/'supplement.jsonl')
        return rows
    STUDY.load_rows = augmented_load
    if args.command == 'prepare':
        print(read(folder/'protocol.json'))
    elif args.command == 'baseline':
        STUDY.score(folder, cfg, 'validation')
    elif args.command in ('pilot', 'formal'):
        STUDY.train(folder, cfg, args.command)
    else:
        if not read(folder/'formal/summary.json')['completed']:
            raise ValueError('Formal training incomplete')
        metrics = read(folder/'formal-validation/report.json')['metrics']
        base = read(folder/'base-validation/report.json')['metrics']
        if not STUDY.pilot_passes(metrics, base, cfg['pilot_gate']):
            raise ValueError('Formal validation failed; keep heldout sealed')
        for adapter, name in ((None, 'base-heldout'), (folder/'formal/adapter', 'lora-heldout')):
            STUDY.score(folder, cfg, 'heldout', adapter, name)


if __name__ == '__main__':
    main()
