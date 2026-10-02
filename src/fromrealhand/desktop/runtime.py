"""Explicit process environments and bounded newline-delimited JSON transport."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEGACY_PYTHON = '/home/smgbro/miniconda3/envs/dexmv/bin/python'
GPU_PACKAGE = ROOT.parent/'.local/mujoco-py-gpu'
GPU_EXTENSION = GPU_PACKAGE/'mujoco_py/generated/cymj_2.0.2.8_37_linuxgpuextensionbuilder_37.so'
MAX_PACKET = 2 * 1024 * 1024


def physics_environment(base=None):
    env = dict(os.environ if base is None else base)
    if not GPU_EXTENSION.is_file():
        raise RuntimeError('Missing isolated GPU mujoco-py extension: '+str(GPU_EXTENSION))
    if not env.get('DISPLAY'):
        raise RuntimeError('No X11 DISPLAY. Launch from the Ubuntu desktop.')
    for name in ('MUJOCO_PY_FORCE_CPU','PYTHONHOME','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH',
                 'QT_QPA_PLATFORM','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE'):
        env.pop(name,None)
    env.update(PYTHONPATH=':'.join(map(str,[GPU_PACKAGE,ROOT/'src',ROOT/'scripts',Path('/home/smgbro/dexmv-sim')])),
        LD_LIBRARY_PATH='/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu',
        LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so',
        __NV_PRIME_RENDER_OFFLOAD='1',__GLX_VENDOR_LIBRARY_NAME='nvidia',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1')
    return env


def language_environment(storage, base=None):
    env = dict(os.environ if base is None else base)
    runtime = Path(storage)/'language/runtime'
    for name in ('LD_PRELOAD','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','PYTHONHOME'):
        env.pop(name,None)
    env.update(PYTHONPATH=str(runtime/'packages')+':'+str(ROOT/'src'),
        LD_LIBRARY_PATH='/usr/lib/x86_64-linux-gnu',
        HF_HOME=str(runtime/'huggingface'),TMPDIR=str(runtime/'tmp'),
        PYTHONNOUSERSITE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    return env


def emit(kind, **fields):
    print(json.dumps(dict(type=kind,**fields),ensure_ascii=False,allow_nan=False,separators=(',',':')),flush=True)


class JsonLines:
    def __init__(self): self.buffer = b''

    def feed(self, chunk):
        self.buffer += bytes(chunk)
        messages = []
        while b'\n' in self.buffer:
            raw,self.buffer = self.buffer.split(b'\n',1)
            if len(raw)>MAX_PACKET: raise ValueError('Oversized worker message')
            if not raw.strip(): continue
            message = json.loads(raw.decode('utf-8'))
            if not isinstance(message,dict) or not isinstance(message.get('type'),str):
                raise ValueError('Invalid worker message')
            messages.append(message)
        if len(self.buffer)>MAX_PACKET: raise ValueError('Oversized incomplete worker message')
        return messages
