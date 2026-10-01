#!/usr/bin/env python3
"""Build small public reports and figures from completed v12 measurements."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'data/processed/dual_video_v12'
OUT=ROOT/'docs/presentation/v12'


def read(path): return json.loads(path.read_text())


def main():
    (OUT/'evidence').mkdir(parents=True,exist_ok=True)
    (OUT/'assets').mkdir(exist_ok=True)
    plan=read(ROOT/'configs/v12-study.json')
    frozen=read(OUT/'evidence/frozen-policy.json')
    candidates=[r['method'] for r in frozen['candidates']];development={}
    for mode in candidates:
        data=read(RUN/mode/'development.json');development[mode]=data
        shutil.copy2(str(RUN/mode/'development.json'),str(OUT/'evidence'/(mode+'-development.json')))
        if (RUN/mode/'input.json').exists():
            shutil.copy2(str(RUN/mode/'input.json'),str(OUT/'evidence'/(mode+'-input.json')))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['task_count','strict_count'],['Task gate / 35','Original strict gate / 35']):
        vals=[development[m]['summary'][key] for m in candidates]
        ax.bar(candidates,vals,color=['#477A9E','#30856B','#B37948','#8768A8','#468B91'])
        ax.tick_params(axis='x',labelrotation=25,labelsize=7)
        ax.set_ylim(0,38);ax.set_title(title)
        for i,value in enumerate(vals): ax.text(i,value+.4,str(value),ha='center')
    fig.tight_layout();fig.savefig(str(OUT/'assets/development.png'),dpi=140);plt.close(fig)
    test=read(RUN/'heldout/summary.json')
    shutil.copy2(str(RUN/'heldout/summary.json'),str(OUT/'evidence/heldout.json'))
    shutil.copy2(str(RUN/'heldout/receipt.json'),str(OUT/'evidence/test-receipt.json'))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    methods=list(test['summary']);labels=['Old residual','Aligned phases','Reference + contact','Selected guard']
    for ax,key,title in zip(axes,['task_count','strict_count','mean_goal_m'],['Task gate / 16','Original strict gate / 16','Mean goal error (mm)']):
        vals=[test['summary'][m][key]*(1000 if key=='mean_goal_m' else 1) for m in methods]
        ax.bar(labels,vals,color=['#777777','#477A9E','#30856B','#B37948']);ax.set_title(title)
        ax.tick_params(axis='x',labelrotation=20,labelsize=8)
        ax.set_ylim(0,max(18,max(vals)*1.2))
        for i,value in enumerate(vals): ax.text(i,value+.2,'%.2f'%value if key=='mean_goal_m' else str(int(value)),ha='center')
    fig.tight_layout();fig.savefig(str(OUT/'assets/heldout.png'),dpi=140);plt.close(fig)
    for video in ('first','second'):
        folder=RUN/'renders'/video
        if folder.exists():
            for path in folder.glob('step_*.jpg'): shutil.copy2(str(path),str(OUT/'assets'/(video+'-'+path.name)))
            shutil.copy2(str(folder/'comparison.json'),str(OUT/'evidence'/(video+'-render.json')))
    lines=['# v12 实验结果','', '日期：2026-10-02。以下只适用于两段已知视频及本轮合成扰动，不代表实物泛化。','',
           '## 开发集：35 工况','', '| 方法 | 任务级 | 原严格级 | 等权平均目标误差 |', '|---|---|---|---|']
    names={'aligned_bc':'精确阶段采样 BC','contact_reference_bc':'参考条件＋接触 BC','old_residual':'旧残差 v10',
           'guard_015':'法向卸载 0.15','guard_030':'法向卸载 0.30','guard_060':'法向卸载 0.60','selected_guard':'开发选中的法向卸载'}
    for mode in candidates:
        r=development[mode]['summary']
        lines.append('| %s | %d/35 | %d/35 | %.2f mm |'%(names[mode],r['task_count'],r['strict_count'],r['mean_goal_m']*1000))
    lines+=['','原 v11 固定时间窗 BC 的开发结果为严格 24/35；按本轮任务门槛重新统计为 27/35。改变采样时序不等于必然提高成功率。',
            '', '![开发对照](assets/development.png)','', '## 冻结后的新留出集','',
            '候选在测试前选择为 **%s**。每视频 8 个组合扰动，四种方法共 64 次回放。'%names[test['selected_before_test']],
            '', '| 方法 | 任务级 | 原严格级 | 稳定抬杯 | 平均目标误差 |', '|---|---|---|---|---|']
    for mode in methods:
        r=test['summary'][mode]
        lines.append('| %s | %d/16 | %d/16 | %d/16 | %.2f mm |'%(names[mode],r['task_count'],r['strict_count'],r['lift_count'],r['mean_goal_m']*1000))
    lines+=['','![独立测试](assets/heldout.png)','',
            '任务级放宽了目标精度及抓法忠实度要求，但没有放宽 1 mm 穿透上限。改善必须在同一标准、同一组工况下比较；不能拿本轮 16 工况的比例直接对比 v11 的 24 工况。',
            '', '## 短训练接口']
    if (RUN/'dapg_smoke/summary.json').exists():
        smoke=read(RUN/'dapg_smoke/summary.json')
        shutil.copy2(str(RUN/'dapg_smoke/summary.json'),str(OUT/'evidence/dapg-smoke.json'))
        shutil.copy2(str(RUN/'dapg_smoke/iterations.json'),str(OUT/'evidence/dapg-iterations.json'))
        lines+=['','完成 %d 次 DAPG 接口短训，%d 次非零参数更新。每次两条完整轨迹；BC 和价值网络使用 GPU，MuJoCo 与旧 MJRL 策略自然梯度更新使用 CPU。'%(smoke['completed_iterations'],smoke['nonzero_updates']),
                '', '该短训策略仅作接口检查，未使用本轮已看过的测试集再次评估，也未替换冻结 BC。名义工况结果：']
        for row in smoke['nominal_post_training']:
            lines.append('- %s：任务级 %s，严格级 %s，目标误差 %.2f mm。'%(row['video'],row['task_pass'],row['strict_pass'],row['report']['final_distance_m']*1000))
    lines+=['','## 可复现与边界','',
            '- 完整轨迹、模型及原始数据留本机；公开代码、配置、冻结哈希、指标与仿真截图。',
            '- 所有新策略均使用实际物理闭环执行，不对运行中的杯子写入位姿。',
            '- 第二组是参考适配、参考误差和接触特征的组合试验，不能单独归因于某篇论文或某个特征。',
            '- 没有新的人类纠偏数据、完整 CR-DAgger 复现或大规模 MimicGen 数据生成。',
            '- 没有启动 2000 次长训练。', '', '[方法与代码说明](METHODS.md) · [运行命令](COMMANDS.md)']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(test['summary'],indent=2))


if __name__=='__main__': main()
