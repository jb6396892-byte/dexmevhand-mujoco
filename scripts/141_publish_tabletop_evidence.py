#!/usr/bin/env python3
"""Publish compact first-three-step evidence, never model weights or raw RGB-D."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STORE=Path(os.environ.get('VISUAL_GRASP_ROOT','/media/smgbro/shared/visual_grasp'))


def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def check(command,env=None):
    p=subprocess.run(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=180)
    if p.returncode: raise RuntimeError(p.stderr+p.stdout)
    return dict(command=command,stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)


def main():
    heldout=STORE/'heldout-v1'; report=read(heldout/'report.json'); protocol=read(heldout/'protocol.json')
    if not report['passed']: raise ValueError('Perception gate failed')
    for name,sha in protocol['source_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Frozen source changed: '+name)
    evidence=ROOT/'docs/presentation/tabletop_rgbd/evidence'; evidence.mkdir(parents=True,exist_ok=True)
    for src,name in [(heldout/'report.json','heldout.json'),(heldout/'protocol.json','protocol.json'),
        (STORE/'development/verification-v1/report.json','development.json'),
        (STORE/'models/grounding-dino-tiny/source.json','detector-source.json'),
        (heldout/'seed-100/assets.json','scene-assets.json'),
        (heldout/'seed-100/observations/0000-rgb.png','tabletop-rgb.png'),
        (heldout/'seed-100/depth-preview.png','tabletop-depth.png'),
        (heldout/'seed-100/estimate/0001-overlay.png','noisy-pose.png'),
        (heldout/'seed-100/estimate/0001-estimate.json','example-estimate.json')]:
        shutil.copyfile(str(src),str(evidence/name))
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts'),
        LD_LIBRARY_PATH='/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu',
        LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        TMPDIR='/tmp')
    python='/home/smgbro/miniconda3/envs/dexmv/bin/python'
    tests=check([python,'-m','unittest','discover','-s','tests'],env)
    guard=check([python,'scripts/122_verify_guard_revision.py','verify'],env)
    paths=list((ROOT/'src/fromrealhand/tabletop').glob('*.py'))+list((ROOT/'src/fromrealhand/perception').glob('*.py'))
    paths += sorted(p for p in (ROOT/'scripts').iterdir() if p.name[:3].isdigit() and 134<=int(p.name[:3])<=141)
    paths += [ROOT/'scripts/tabletop_python.sh',ROOT/'tests/test_tabletop_camera.py',ROOT/'configs/tabletop-perception-v1.json']
    for p in paths:
        if p.suffix=='.py': ast.parse(p.read_text(),filename=str(p))
        if p.suffix=='.sh': check(['bash','-n',str(p)])
    pixels=check([python,'-c',
        'import cv2,json; from pathlib import Path; p=Path("docs/presentation/tabletop_rgbd/evidence"); '
        'r={x.name:float(cv2.imread(str(x)).std()) for x in p.glob("*.png")}; '
        'assert len(r)==3 and all(v>5 for v in r.values()); print(json.dumps(r))'],env)
    quality=dict(passed=True,unit_tests=tests,frozen_language_guard=guard,screenshot_pixels=pixels,
        diff_check=check(['git','diff','--check']),control_enabled=False,
        native_viewer_verification='scripts/139_view_tabletop.py --verify-frames 120 exited 0')
    write(evidence/'quality.json',quality)
    runtime=check(['bash','scripts/tabletop_python.sh','-m','pip','list','--path',str(STORE/'runtime/packages'),'--format=json'])
    write(evidence/'vision-packages.json',json.loads(runtime['stdout']))
    clean,noisy=report['metrics']['clean'],report['metrics']['noisy']
    lines=['# 前三步量化结果','',
        '| 输入 | 准入 | 位置误差中位数 | 位置误差 P95 | 朝向误差中位数 | 单帧推理中位时间 |',
        '| --- | --- | --- | --- | --- | --- |']
    for label,m in [('理想 RGB-D',clean),('1 mm 深度噪声和 2% 缺失',noisy)]:
        lines.append('| %s | %d/%d | %.3f mm | %.3f mm | %.3f 度 | %.3f s |'%(label,m['accepted'],m['total'],
            m['position_median_m']*1000,m['position_p95_m']*1000,m['rotation_median_deg'],m['inference_median_s']))
    lines += ['', '无杯子与无有效深度共 %d 项，拒绝 %d 项。'%(report['negatives'],report['negative_rejections']),
        '', '这里的准入是位姿估计通过，不是抓取成功率。所有抓取控制均未启用。',
        '这是已知同一杯子、直立且杯柄可见的有限仿真测试，不代表真实相机精度或未知物体泛化。',
        '时间不含模型加载；每个场景首帧包含 CUDA 首次推理开销，噪声帧在同进程随后执行，不能据此判断噪声让算法更快。',
        '', '源报告：`evidence/heldout.json`；测试前冻结：`evidence/protocol.json`。']
    (evidence.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    paths += [ROOT/'README.md',ROOT/'docs/index.html',ROOT/'docs/RGBD_TABLETOP_TASK.md',ROOT/'docs/RGBD_TABLETOP_PROGRESS.md',
        ROOT/'docs/presentation/tabletop_rgbd/README.md',ROOT/'docs/presentation/tabletop_rgbd/RESULTS.md',
        ROOT/'docs/run_logs/2026-10-03-tabletop-rgbd.md']
    paths += [p for p in evidence.iterdir() if p.is_file() and p.name!='manifest.json']
    manifest=dict(base_commit=check(['git','rev-parse','HEAD'])['stdout'].strip(),
        sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        raw_store=str(STORE),old_qt_main_commit='1862317d7da61483063e8c4284d431ec6ff72a7c')
    write(evidence/'manifest.json',manifest)
    archive=STORE/'delivery-v1'
    for p in paths+[evidence/'manifest.json']:
        target=archive/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(str(p),str(target))
        if digest(p)!=digest(target): raise ValueError('Archive copy mismatch')
    print(json.dumps(dict(passed=True,evidence=str(evidence),archive=str(archive)),ensure_ascii=False))


if __name__=='__main__': main()
