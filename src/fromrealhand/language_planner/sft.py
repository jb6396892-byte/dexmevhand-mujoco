"""Assistant-only SFT encoding. Heavy model dependencies are imported lazily."""
from pathlib import Path
import json

from .contracts import compact, digest, read


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_protocol(root, dataset):
    protocol = read(Path(dataset)/'protocol.json')
    for path, expected in protocol['sha256'].items():
        source = (Path(root)/path).resolve()
        if Path(root).resolve() not in source.parents or digest(source) != expected:
            raise ValueError('Frozen language input changed: '+path)
    return protocol


def verify_model(path):
    path = Path(path).resolve()
    source = read(path/'source.json')
    for name, expected in source['sha256'].items():
        artifact = (path/name).resolve()
        if path not in artifact.parents or digest(artifact) != expected:
            raise ValueError('Downloaded base model changed: '+name)
    return source


def messages(instruction, scene, system_prompt):
    if scene not in ('first', 'second'):
        raise ValueError('Unknown scene')
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 120:
        raise ValueError('Instruction must contain 1 to 120 characters')
    return [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': compact(dict(scene=scene, instruction=instruction))}]


def encode(row, tokenizer, system_prompt, max_length):
    prompt = messages(row['instruction'], row['scene'], system_prompt)
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(prompt + [{'role': 'assistant', 'content': compact(row['response'])}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('Chat template does not preserve the assistant boundary')
    if len(full) > max_length:
        raise ValueError('Sequence exceeds max_length; do not silently truncate the JSON target')
    return dict(input_ids=full, attention_mask=[1]*len(full), labels=[-100]*len(prefix)+full[len(prefix):])


def collate(features, pad_token_id):
    import torch
    if not features:
        raise ValueError('Empty training batch')
    width = max(len(row['input_ids']) for row in features)
    result = {}
    for key, pad in (('input_ids', pad_token_id), ('attention_mask', 0), ('labels', -100)):
        result[key] = torch.tensor([row[key]+[pad]*(width-len(row[key])) for row in features], dtype=torch.long)
    return result


def load_model(model_path, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('Stage6 requires the isolated CUDA runtime; no silent CPU fallback')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, torch_dtype=dtype, attn_implementation='sdpa').to('cuda')
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    return model, tokenizer


def generate(model, tokenizer, instruction, scene, system_prompt, max_new_tokens):
    import torch
    ids = tokenizer.apply_chat_template(messages(instruction, scene, system_prompt),
        tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
    model.eval()
    with torch.inference_mode():
        result = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True).strip()
