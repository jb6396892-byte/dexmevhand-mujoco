"""Desktop inference reuses the frozen model and veto-only acceptance contract."""
import contextlib
import datetime
from pathlib import Path
import sys

from .runtime import ROOT, emit
from ..language_planner.contracts import digest, read, write
from ..language_planner.guard_revision import instruction_contract, semantic_guard, verify_revision
from ..language_planner.refinement import deployment_matches


def checked_study(storage):
    study = Path(storage).resolve()/'language/study_v4_guard2'
    cfg = verify_revision(ROOT,study)
    receipt = read(study/'acceptance/summary.json')
    if not deployment_matches(receipt,study,study.parent/'model'):
        raise ValueError('Language/physics acceptance is missing or changed; execution locked')
    return study,cfg


def new_run(storage):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return Path(storage).resolve()/'language/desktop_runs'/stamp


def plan_instruction(storage, instruction, scene, output):
    study,cfg = checked_study(storage)
    output = Path(output).resolve()
    if output.parent != study.parent/'desktop_runs':
        raise ValueError('Desktop runs must stay in shared language/desktop_runs')
    output.mkdir(parents=True,exist_ok=False)
    emit('status',state='planning',message='正在检查指令',output=str(output))
    contract = instruction_contract(instruction)
    record = dict(instruction=instruction,scene=scene,model_called=False,
        execution_attempted=False,simulation_created=False,
        acceptance_sha256=digest(study/'acceptance/summary.json'))
    if not contract['allowed']:
        guard = dict(accepted=False,reason='instruction_not_admitted',instruction_contract=contract)
        write(output/'generation.json',dict(record,guard=guard,steps=0))
        emit('plan',accepted=False,guard=guard,model_called=False,output=str(output))
        return False
    from ..language_planner.sft import generate, load_model, verify_model
    emit('status',state='planning',message='加载 Qwen + LoRA')
    verify_model(study.parent/'model')
    with contextlib.redirect_stdout(sys.stderr):
        model,tokenizer = load_model(study.parent/'model',study/'candidate/adapter')
        raw = generate(model,tokenizer,instruction,scene,
            (ROOT/'configs/stage6-system-prompt.txt').read_text(encoding='utf-8'),cfg['max_new_tokens'])
    guard = semantic_guard(raw,instruction,scene,read(ROOT/'configs/skill_plan.schema.json'),
        read(study/'dataset/feasibility.json'),cfg['feasibility_threshold'])
    write(output/'generation.json',dict(record,raw=raw,guard=guard,model_called=True,producer='lora_model'))
    if guard['accepted']: write(output/'plan.json',guard['response']['plan'])
    emit('plan',accepted=guard['accepted'],guard=guard,raw=raw,model_called=True,output=str(output))
    return guard['accepted']


def validated_plan(storage, output):
    study,cfg = checked_study(storage)
    output = Path(output).resolve()
    if output.parent != study.parent/'desktop_runs': raise ValueError('Invalid desktop run directory')
    generated = read(output/'generation.json')
    if generated.get('producer')!='lora_model' or generated.get('acceptance_sha256')!=digest(study/'acceptance/summary.json'):
        raise ValueError('Plan does not match accepted model provenance')
    guard = semantic_guard(generated['raw'],generated['instruction'],generated['scene'],
        read(ROOT/'configs/skill_plan.schema.json'),read(study/'dataset/feasibility.json'),cfg['feasibility_threshold'])
    if not guard['accepted'] or read(output/'plan.json')!=guard['response']['plan']:
        raise ValueError('Plan changed or failed independent instruction guard')
    return guard['response']['plan']
