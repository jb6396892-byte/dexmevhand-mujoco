#!/usr/bin/env python3
"""Explicit download or LoRA/SFT; never modifies the dexmv environment."""
import argparse
import datetime
from functools import partial
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fromrealhand.language_planner.contracts import digest, read, write
from fromrealhand.language_planner.sft import collate, encode, load_model, load_rows, verify_model, verify_protocol


def require_shared(path):
    shared = Path(os.environ.get('STAGE6_SHARED', '/media/smgbro/shared')).resolve()
    path = Path(path).resolve()
    if not os.path.ismount(str(shared)) or shared not in path.parents:
        raise ValueError('Large model/runtime files must stay on the mounted shared disk')
    if os.statvfs(str(shared)).f_flag & os.ST_RDONLY:
        raise ValueError('Shared disk is read-only; no download or training started')
    if shutil.disk_usage(str(shared)).free < 3*1024**3:
        raise ValueError('Less than 3 GiB shared free space')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'download', 'train'])
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/processed/stage6_language_v1/dataset')
    parser.add_argument('--model', type=Path, default=Path('/media/smgbro/shared/fromrealhand-stage6/model'))
    parser.add_argument('--output', type=Path, default=Path('/media/smgbro/shared/fromrealhand-stage6/lora-v1'))
    args = parser.parse_args()
    protocol = verify_protocol(ROOT, args.dataset)
    cfg = read(ROOT/'configs/stage6-lora.json')
    if args.mode == 'check':
        print(dict(input_hashes_verified=True, counts=protocol['counts'], model_installed=args.model.exists(),
                   lora_training_started=False))
        return
    require_shared(args.model)
    if args.mode == 'download':
        if args.model.exists():
            raise ValueError('Model destination already exists')
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(cfg['model_id']).sha
        snapshot_download(cfg['model_id'], revision=revision, local_dir=str(args.model),
            allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'LICENSE', 'README.md'])
        write(args.model/'source.json', dict(model_id=cfg['model_id'], revision=revision,
            downloaded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            sha256={str(p.relative_to(args.model)): digest(p) for p in args.model.iterdir() if p.is_file()}))
        print('Model downloaded at pinned revision '+revision+'; training not started')
        return
    require_shared(args.output)
    if args.output.exists():
        raise ValueError('Refusing to overwrite previous training')
    monitor = read(ROOT/'data/processed/dual_video_v14c/supervision_200/status.json')
    if monitor.get('state') != 'completed':
        raise ValueError('Finish and review the existing DAPG run before starting LoRA')
    source = verify_model(args.model)
    if source['model_id'] != cfg['model_id']:
        raise ValueError('Unexpected base model')
    import torch
    import math
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    set_seed(cfg['seed'])
    model, tokenizer = load_model(args.model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg['lora_rank'],
        lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
        target_modules=cfg['target_modules'], bias='none'))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not trainable or any('lora_' not in name for name in trainable):
        raise ValueError('Only LoRA matrices may be trainable')
    prompt = (ROOT/'configs/stage6-system-prompt.txt').read_text(encoding='utf-8')
    # Only train and validation are consumed here; heldout remains unopened.
    datasets = {split: [encode(r, tokenizer, prompt, cfg['max_length'])
        for r in load_rows(args.dataset/(split+'.jsonl'))] for split in ('train', 'validation')}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'started.json', dict(config=cfg, base=source,
        dataset_protocol_sha256=digest(args.dataset/'protocol.json'),
        trainable_parameters=sum(p.numel() for p in trainable.values()),
        total_parameters=sum(p.numel() for p in model.parameters()), cuda=torch.version.cuda))
    training_args = TrainingArguments(output_dir=str(args.output/'checkpoints'),
        num_train_epochs=cfg['epochs'], per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'], gradient_accumulation_steps=cfg['gradient_accumulation'],
        learning_rate=cfg['learning_rate'], warmup_ratio=.05, lr_scheduler_type='cosine',
        max_grad_norm=1., weight_decay=0., bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, save_total_limit=2,
        report_to=[], logging_steps=10, seed=cfg['seed'], data_seed=cfg['seed'],
        dataloader_num_workers=0, optim='adamw_torch', prediction_loss_only=True)
    class FiniteTraining(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(value, (int, float)) and not math.isfinite(value)
                   for value in (logs or {}).values()):
                raise FloatingPointError('Nonfinite training metric; training halted')
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets['train'],
        eval_dataset=datasets['validation'], callbacks=[FiniteTraining()],
        data_collator=partial(collate, pad_token_id=tokenizer.pad_token_id))
    result = trainer.train()
    if not all(torch.isfinite(p).all().item() for p in trainable.values()):
        raise FloatingPointError('Nonfinite adapter; refusing completed receipt')
    if not any(torch.count_nonzero(p).item() for name, p in trainable.items() if 'lora_B' in name):
        raise RuntimeError('No nonzero LoRA update; refusing completed receipt')
    adapter = args.output/'adapter'
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    trainer.save_state()
    verify_protocol(ROOT, args.dataset)
    write(args.output/'summary.json', dict(completed=True, optimization_steps=trainer.state.global_step,
        best_checkpoint=trainer.state.best_model_checkpoint, best_validation_loss=trainer.state.best_metric,
        metrics=result.metrics, heldout_used=False, base_revision=source['revision'],
        dataset_protocol_sha256=digest(args.dataset/'protocol.json'),
        adapter_sha256={p.name: digest(p) for p in adapter.iterdir() if p.is_file()}))


if __name__ == '__main__':
    main()
