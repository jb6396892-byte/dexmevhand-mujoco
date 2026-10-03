#!/usr/bin/env python3
"""Download pinned public safetensors only, into the shared vision directory."""
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root=Path(os.environ['VISUAL_GRASP_ROOT'])
model_id='IDEA-Research/grounding-dino-tiny'
target=root/'models/grounding-dino-tiny'
previous=json.loads((target/'source.json').read_text()) if (target/'source.json').exists() else None
revision=previous['revision'] if previous else HfApi().model_info(model_id).sha
snapshot_download(model_id,revision=revision,local_dir=str(target),
    allow_patterns=['*.json','*.txt','*.safetensors','README.md','LICENSE*'])
record=dict(model_id=model_id,revision=revision,license='Apache-2.0',
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file() and p.name!='source.json'})
if previous and record!=previous: raise RuntimeError('Previously pinned detector files changed')
(target/'source.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
