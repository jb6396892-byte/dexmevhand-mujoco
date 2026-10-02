#!/usr/bin/env python3
"""Shared-storage base comparison, gated LoRA pilot, formal SFT and untouched test."""
import argparse
import datetime
from functools import partial
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.language_planner.contracts import digest, read, write
from fromrealhand.language_planner.evaluation import evaluate
from fromrealhand.language_planner.sft import (
    collate, encode, generate, load_model, load_rows, verify_model, verify_protocol)

DATASET = ROOT/'data/processed/stage6_language_v1/dataset'
CONFIG = ROOT/'configs/stage6-study-v2.json'
PROMPT = (ROOT/'configs/stage6-system-prompt.txt').read_text(encoding='utf-8')


def pilot_passes(metrics, base, gate):
    return bool(metrics['total'] > 0 and metrics['semantic_accuracy'] >= gate['semantic_accuracy_min']
        and metrics['schema_legal']/metrics['total'] >= gate['schema_fraction_min']
        and metrics['unsupported_false_execution'] <= gate['unsupported_false_execution_max']
        and (not gate['not_worse_than_base'] or metrics['semantic_correct'] >= base['semantic_correct']))


def verify_study(folder):
    verify_protocol(ROOT, DATASET)
    path = folder/'protocol.json'
    if not path.exists():
        write(path, dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            config=read(CONFIG), dataset_protocol_sha256=digest(DATASET/'protocol.json'),
            sha256={str(p.relative_to(ROOT)): digest(p) for p in (CONFIG, Path(__file__).resolve())},
            heldout_used_for_selection=False))
    result = read(path)
    for name, expected in result['sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Language study changed; create a new version instead')
    return result['config']


def download(folder, cfg):
    from huggingface_hub import HfApi, snapshot_download
    model = folder.parent/'model'
    if (model/'source.json').exists():
        verify_model(model)
        print('Existing pinned model verified', flush=True)
        return
    request = folder/'download-request.json'
    if not request.exists():
        write(request, dict(model_id=cfg['model_id'], revision=HfApi().model_info(cfg['model_id']).sha))
    pinned = read(request)
    snapshot_download(pinned['model_id'], revision=pinned['revision'], local_dir=str(model),
        allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'LICENSE', 'README.md'])
    write(model/'source.json', dict(pinned, sha256={p.name: digest(p) for p in model.iterdir() if p.is_file()}))
    print('MODEL_READY', pinned, flush=True)


def score(folder, cfg, split, adapter=None, name=None):
    import torch
    model_path = folder.parent/'model'
    source = verify_model(model_path)
    destination = folder/(name or ('lora-'+split if adapter else 'base-'+split))
    if destination.exists():
        raise ValueError('Evaluation exists; no overwrite or test-set retuning')
    model, tokenizer = load_model(model_path, adapter)
    model.config.use_cache = True
    destination.mkdir(parents=True, exist_ok=False)
    receipt = dict(producer='lora_model' if adapter else 'pretrained_model', split=split,
        model_id=source['model_id'], revision=source['revision'],
        model_source_sha256=digest(model_path/'source.json'),
        dataset_protocol_sha256=digest(DATASET/'protocol.json'), study_protocol_sha256=digest(folder/'protocol.json'),
        adapter_sha256={} if adapter is None else {p.name: digest(p) for p in adapter.iterdir() if p.is_file()})
    write(destination/'protocol.json', receipt)
    rows = load_rows(DATASET/(split+'.jsonl'))
    predictions = []
    started = time.monotonic()
    for row in rows:
        raw = generate(model, tokenizer, row['instruction'], row['scene'], PROMPT, cfg['max_new_tokens'])
        predictions.append(dict(id=row['id'], raw=raw))
        write(destination/'predictions.json', predictions)
        print('GENERATE', destination.name, len(predictions), len(rows), row['id'], raw, flush=True)
    report = evaluate(rows, predictions, read(ROOT/'configs/skill_plan.schema.json'),
                      read(DATASET/'feasibility.json'), cfg['feasibility_threshold'])
    write(destination/'report.json', dict(report, producer=receipt['producer'], elapsed_s=time.monotonic()-started))
    del model
    torch.cuda.empty_cache()
    print('LANGUAGE', destination.name, json.dumps(report['metrics']), flush=True)
    return report


def train(folder, cfg, mode):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    if mode == 'formal' and not read(folder/'pilot-gate.json')['passed']:
        raise ValueError('Pilot failed; formal SFT must not start')
    destination = folder/mode
    if destination.exists():
        raise ValueError('Training directory exists; preserve previous attempt')
    source = verify_model(folder.parent/'model')
    set_seed(cfg['seed'])
    model, tokenizer = load_model(folder.parent/'model')
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg['lora_rank'],
        lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
        target_modules=cfg['target_modules'], bias='none'))
    parameters = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not parameters or any('lora_' not in n for n in parameters):
        raise ValueError('Unexpected trainable base weights')
    data = {split:[encode(r, tokenizer, PROMPT, cfg['max_length'])
                   for r in load_rows(DATASET/(split+'.jsonl'))] for split in ('train','validation')}
    destination.mkdir(parents=True, exist_ok=False)
    write(destination/'started.json', dict(mode=mode, epochs=cfg[mode+'_epochs'],
        trainable_parameters=sum(p.numel() for p in parameters.values()),
        max_sequence_tokens=max(len(r['input_ids']) for values in data.values() for r in values),
        source_revision=source['revision'], config=cfg, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    class Watch(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in (logs or {}).values()):
                raise FloatingPointError('Nonfinite training metric')
            write(destination/'progress.json', dict(step=state.global_step, epoch=state.epoch, logs=logs,
                gpu_peak_bytes=torch.cuda.max_memory_allocated(), updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    args = TrainingArguments(output_dir=str(destination/'checkpoints'),
        num_train_epochs=cfg[mode+'_epochs'], per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['gradient_accumulation'],
        learning_rate=cfg['learning_rate'], warmup_ratio=.05, lr_scheduler_type='cosine',
        max_grad_norm=1., weight_decay=0., bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, save_total_limit=2,
        report_to=[], logging_steps=10, logging_nan_inf_filter=False, seed=cfg['seed'], data_seed=cfg['seed'],
        dataloader_num_workers=0, optim='adamw_torch', prediction_loss_only=True, label_names=['labels'])
    trainer = Trainer(model=model, args=args, train_dataset=data['train'], eval_dataset=data['validation'],
        data_collator=partial(collate,pad_token_id=tokenizer.pad_token_id), callbacks=[Watch()])
    result = trainer.train()
    if not all(torch.isfinite(p).all().item() for p in parameters.values()):
        raise FloatingPointError('Nonfinite LoRA adapter')
    if not any(torch.count_nonzero(p).item() for n,p in parameters.items() if 'lora_B' in n):
        raise RuntimeError('Zero LoRA updates')
    adapter = destination/'adapter'
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    trainer.save_state()
    write(destination/'summary.json', dict(completed=True, steps=trainer.state.global_step,
        best_validation_loss=trainer.state.best_metric, best_checkpoint=trainer.state.best_model_checkpoint,
        metrics=result.metrics, gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        dataset_protocol_sha256=digest(DATASET/'protocol.json'),
        adapter_sha256={p.name:digest(p) for p in adapter.iterdir() if p.is_file()}, heldout_used=False))
    del trainer, model, parameters
    torch.cuda.empty_cache()
    report = score(folder,cfg,'validation',adapter,name=mode+'-validation')
    if mode == 'pilot':
        base = read(folder/'base-validation/report.json')['metrics']
        m = report['metrics']; gate = cfg['pilot_gate']
        passed = pilot_passes(m, base, gate)
        write(folder/'pilot-gate.json', dict(passed=passed, base=base, pilot=m,
            policy='Proceed only if preregistered validation gate passes; no heldout seen'))
        print('PILOT_GATE', passed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['download','baseline','pilot','formal','heldout'])
    parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
    args = parser.parse_args()
    root = args.root.resolve()
    if Path('/media/smgbro/shared') not in root.parents:
        raise ValueError('Language artifacts must stay on shared storage')
    folder = root/'language/study_v2'
    cfg = verify_study(folder)
    if args.command == 'download': download(folder,cfg)
    elif args.command == 'baseline': score(folder,cfg,'validation')
    elif args.command in ('pilot','formal'): train(folder,cfg,args.command)
    else:
        if not read(folder/'formal/summary.json')['completed']:
            raise ValueError('Formal training incomplete')
        for adapter,name in ((None,'base-heldout'),(folder/'formal/adapter','lora-heldout')):
            score(folder,cfg,'heldout',adapter,name)
