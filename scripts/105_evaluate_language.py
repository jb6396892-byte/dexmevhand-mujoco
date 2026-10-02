#!/usr/bin/env python3
"""Evaluate raw base/LoRA generations without silently falling back to rules."""
import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.language_planner.contracts import digest, read, write
from fromrealhand.language_planner.evaluation import evaluate
from fromrealhand.language_planner.sft import generate, load_model, load_rows, verify_model, verify_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--split', choices=['validation', 'heldout'], default='validation')
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/processed/stage6_language_v1/dataset')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_protocol(ROOT, args.dataset)
    if args.output.exists():
        raise ValueError('Refusing to overwrite evaluation; heldout is never a tuning set')
    source = verify_model(args.model)
    cfg = read(ROOT/'configs/stage6-lora.json')
    prompt = (ROOT/'configs/stage6-system-prompt.txt').read_text(encoding='utf-8')
    rows = load_rows(args.dataset/(args.split+'.jsonl'))
    receipt = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        producer='pretrained_model' if args.adapter is None else 'lora_model',
        model_id=source['model_id'], revision=source['revision'], split=args.split,
        model_source_sha256=digest(args.model/'source.json'),
        dataset_protocol_sha256=digest(args.dataset/'protocol.json'),
        adapter_sha256={} if args.adapter is None else {p.name: digest(p) for p in args.adapter.iterdir() if p.is_file()},
        inference_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(),
            ROOT/'src/fromrealhand/language_planner/sft.py', ROOT/'src/fromrealhand/language_planner/evaluation.py']})
    model, tokenizer = load_model(args.model, args.adapter)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'protocol.json', receipt)
    predictions = []
    for row in rows:
        raw = generate(model, tokenizer, row['instruction'], row['scene'], prompt, cfg['max_new_tokens'])
        predictions.append(dict(id=row['id'], raw=raw))
        write(args.output/'predictions.json', predictions)
        print(json.dumps(dict(id=row['id'], raw=raw), ensure_ascii=False), flush=True)
    report = evaluate(rows, predictions, read(ROOT/'configs/skill_plan.schema.json'),
        read(args.dataset/'feasibility.json'), cfg['feasibility_threshold'])
    write(args.output/'report.json', dict(report, producer=receipt['producer'],
        evaluation_protocol_sha256=digest(args.output/'protocol.json')))
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
