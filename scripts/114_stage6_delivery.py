#!/usr/bin/env python3
"""Environment receipt and language-only delivery with no stage3 artifacts."""
import argparse
import datetime
from importlib import metadata
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fromrealhand.language_planner.contracts import read, write, digest
from fromrealhand.language_planner.sft import encode, load_rows, verify_protocol


def environment(root):
    import torch
    from transformers import AutoTokenizer
    language = root/'language'
    packages = language/'runtime/packages'
    tokenizer = AutoTokenizer.from_pretrained(str(language/'model'),local_files_only=True,trust_remote_code=False)
    prompt = (ROOT/'configs/stage6-system-prompt.txt').read_text(encoding='utf-8')
    samples = [encode(r,tokenizer,prompt,1024) for split in ('train','validation')
               for r in load_rows(ROOT/('data/processed/stage6_language_v1/dataset/'+split+'.jsonl'))]
    x = torch.randn(64,64,device='cuda',dtype=torch.bfloat16)
    receipt = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        python=sys.version, executable=sys.executable,packages=str(packages),
        versions={d.metadata['Name']:d.version for d in metadata.distributions(path=[str(packages)])},
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        bf16=torch.cuda.is_bf16_supported(), cuda_matmul_finite=bool(torch.isfinite(x@x).all()),
        encoded_samples=len(samples),max_tokens=max(len(r['input_ids']) for r in samples),
        min_supervised_tokens=min(sum(v!=-100 for v in r['labels']) for r in samples),heldout_used=False)
    write(language/'environment.json',receipt)
    print(receipt)


def publish(root, study_name):
    language = root/'language'
    study = language/study_name
    verify_protocol(ROOT,ROOT/'data/processed/stage6_language_v1/dataset')
    formal_path = study/'formal/summary.json'
    formal = read(formal_path) if formal_path.exists() else None
    mode = 'formal' if formal else 'pilot'
    summary = read(study/mode/'summary.json')
    if not summary['completed']:
        raise ValueError('Language training incomplete')
    for name, expected in summary['adapter_sha256'].items():
        if digest(study/mode/'adapter'/name) != expected:
            raise ValueError('Adapter changed')
    acceptance_path = study/'physical-acceptance/summary.json'
    accepted = acceptance_path.exists() and read(acceptance_path)['model_acceptance_passed']
    destination = ROOT/'docs/presentation/stage6/evidence'/study_name.replace('_','-')
    destination.mkdir(parents=True,exist_ok=False)
    sources = dict(environment=language/'environment.json', protocol=study/'protocol.json',
        model_source=language/'model/source.json', pilot_gate=study/'pilot-gate.json',
        pilot=study/'pilot/summary.json')
    if formal:
        sources['formal'] = formal_path
    if acceptance_path.exists():
        sources['acceptance'] = acceptance_path
        sources['physical'] = study/'physical-acceptance/physical.json'
        sources['negative_guards'] = study/'physical-acceptance/negative_guards.json'
    for split in ('base-validation','pilot-validation','formal-validation','base-heldout','lora-heldout'):
        if (study/split/'report.json').exists():
            sources[split] = study/split/'report.json'
            sources[split+'-protocol'] = study/split/'protocol.json'
    for name,path in sources.items():
        shutil.copyfile(str(path),str(destination/(name+'.json')))
    delivery = language/('delivery-'+study_name)
    delivery.mkdir(exist_ok=False)
    for name in ('STAGE6_STUDY.md','STAGE6_LANGUAGE.md'):
        shutil.copyfile(str(ROOT/'docs'/name),str(delivery/name))
    result_doc = ROOT/'docs/presentation/stage6/STUDY_RESULTS.md'
    if result_doc.exists():
        shutil.copyfile(str(result_doc),str(delivery/'STUDY_RESULTS.md'))
    source_folder = delivery/'source'
    source_folder.mkdir()
    for path in [ROOT/'scripts/111_language_study.py', ROOT/'scripts/115_language_contrast_study.py',
                 ROOT/'scripts/stage6_study_python.sh', ROOT/'requirements-stage6.txt',
                 ROOT/'scripts/116_run_stage6_model.sh', ROOT/'configs/stage6-study-v2.json',
                 ROOT/'configs/stage6-contrast-phrases.json', ROOT/'configs/stage6-system-prompt.txt']:
        shutil.copyfile(str(path),str(source_folder/path.name))
    shutil.copytree(str(ROOT/'data/processed/stage6_language_v1/dataset'),str(delivery/'dataset'))
    if (study/'supplement.jsonl').exists():
        shutil.copyfile(str(study/'supplement.jsonl'),str(delivery/'dataset/supplement.jsonl'))
    shutil.copyfile(str(study/'protocol.json'),str(delivery/'study-protocol.json'))
    write(delivery/'model-index.json',dict(base=str(language/'model'),adapter=str(study/mode/'adapter'),
        pilot_adapter=str(study/'pilot/adapter'),reports=str(study),
        original_dataset_protocol=digest(ROOT/'data/processed/stage6_language_v1/dataset/protocol.json'),
        original_project=str(ROOT),stage3_included=False, formal_training_completed=bool(formal),
        accepted=bool(accepted), deployment_allowed=bool(accepted)))
    write(destination/'files.json',dict(files={p.name:digest(p) for p in destination.iterdir() if p.is_file()},
        shared_delivery=str(delivery),weights_uploaded_to_git=False))
    print('PUBLISHED',destination,'SHARED',delivery)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['environment','publish'])
    parser.add_argument('--root',type=Path,default=Path('/media/smgbro/shared/lora'))
    parser.add_argument('--study',choices=['study_v2','study_v3'],default='study_v3')
    args = parser.parse_args()
    if args.command == 'environment':
        environment(args.root.resolve())
    else:
        publish(args.root.resolve(),args.study)
