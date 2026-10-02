#!/usr/bin/env python3
"""Publish compact v13 tables and verify data admission, without new screenshots."""
import collections
import importlib
import io
import json
import pickle
import subprocess
import unittest
from pathlib import Path
import numpy as np
from fromrealhand.multivideo import trajectory_arrays

study = importlib.import_module('83_run_v13_routing')
ROOT, RUN, PLAN = study.ROOT, study.RUN, study.PLAN
OUT = ROOT/'docs/presentation/v13'


def compact(row):
    fields = ('final_distance_m', 'max_hand_scene_penetration_m', 'tail_slip_m',
              'tail_finger_contact_fraction', 'mean_tip_error_m', 'tail_mean_tip_error_m',
              'phase_contacts', 'scene_contact_peaks')
    return dict(video=row['video'], case=row['case'], task_pass=row['task_pass'],
                strict_pass=row['strict_pass'], lift_pass=row['lift_pass'],
                failures_strict=row['failures_strict'], phases=row['phases'],
                report={k:row['report'][k] for k in fields if k in row['report']})


def main():
    plan, old, parent, admission, demos, refs, base, data = study.inputs()
    comparison = json.loads((RUN/'comparison.json').read_text())
    corrections = json.loads((RUN/'corrections/admission.json').read_text())
    assert corrections['completed']
    modes = ['shared', 'video', 'phase']
    if corrections['accepted_cases']:
        modes += ['base_finetune', 'corrected']
    rows, source_hashes = {}, {}
    for mode in modes:
        path = RUN/mode/'development.json'
        result = json.loads(path.read_text())
        assert result['protocol_sha256'] == study.digest(PLAN)
        assert study.digest(result['policy']) == result['policy_sha256']
        assert len(result['reports']) == 35
        assert len({(r['video'], r['case']) for r in result['reports']}) == 35
        for r in result['reports']:
            for key, value in study.study.gates(r['report'], old).items():
                assert r[key] == value
        assert study.study.summarize(result['reports']) == result['summary']
        rows[mode] = result
        source_hashes[mode] = dict(policy=result['policy_sha256'], development=study.digest(path))
        public = {k:v for k,v in result.items() if k!='reports'}
        public['reports'] = [compact(r) for r in result['reports']]
        study.write_json(OUT/'evidence'/(mode+'.json'), public)
    assert rows['shared']['summary'] == json.loads((study.study.RUN/'aligned_bc/development.json').read_text())['summary']
    for mode in ('video', 'phase'):
        cp = pickle.loads(Path(rows[mode]['policy']).read_bytes())
        audit = json.loads((RUN/mode/'input.json').read_text())
        assert audit['input_sha256'] == study.digest(study.study.RUN/'aligned_bc/input.npz')
        assert audit['sampled_frames'] == len(data['features'])*plan['epochs']
        for entry in admission['reports']:
            video = parent['videos'][entry['video_id']]
            route = cp['routes'][video['name']]
            np.testing.assert_allclose(route.sum(1), 1.)
            assert np.isfinite(route).all() and (route>=0.).all()
            if mode == 'phase':
                expected = study.phase_routes(video['horizon'], np.load(entry['geometry']),
                    video['control']['time_scale'], plan['phase_blend_half_width_steps'])
                np.testing.assert_array_equal(route, expected)
        study.write_json(OUT/'evidence'/(mode+'-input.json'), audit)
    accepted = []
    for case in corrections['reports']:
        attempts = json.loads((RUN/'corrections'/(case['case']+'-attempts.json')).read_text())
        public_attempts = []
        for attempt in attempts:
            for key, value in study.study.gates(attempt['report'], old).items():
                assert attempt[key] == value
            public_attempts.append(dict(start_step=attempt['start_step'], gain=attempt['gain'],
                task_pass=attempt['task_pass'], strict_pass=attempt['strict_pass'], admitted=attempt['admitted'],
                goal_m=attempt['report']['final_distance_m'],
                scene_penetration_m=attempt['report']['max_hand_scene_penetration_m'],
                failures_strict=attempt['failures_strict']))
        study.write_json(OUT/'evidence'/(case['case']+'-attempts.json'), public_attempts)
        a = case['accepted']
        if a is None:
            continue
        assert a['task_pass'] and a['replay']['task_pass'] and a['half']['task_pass']
        for key, value in study.study.gates(a['half_report'], old).items():
            assert a['half'][key] == value
        assert a['replay_error'] < 1e-8 and a['label_action_error'] < 1e-12
        assert a['student_prefix_action_error'] < 1e-8 and a['student_prefix_observation_error'] < 1e-8
        assert study.digest(a['labels']) == a['labels_sha256']
        labels = np.load(a['labels'])
        rollout = pickle.loads((Path(a['labels']).parent/'rollout.pkl').read_bytes())['video_faithful']
        assert len(labels['steps']) == a['label_frames']
        assert np.isfinite(labels['features']).all() and np.isfinite(labels['labels']).all()
        assert labels['steps'].min() >= a['start_step']+plan['correction']['transition_steps']
        cp = pickle.loads(Path(corrections['student']['path']).read_bytes())
        assert (np.abs(labels['labels']) <= cp['residual_limits']['second']+1e-12).all()
        np.testing.assert_allclose(labels['labels']+cp['references']['second'][labels['steps']],
                                   rollout['actions'][labels['steps']], rtol=0., atol=1e-12)
        entry = next(e for e in admission['reports'] if e['video']=='second' and e['name']==case['case'])
        features, _, _ = trajectory_arrays(rollout, cp['references']['second'], parent['videos'][1], np.load(entry['geometry']))
        np.testing.assert_allclose(labels['features'], features[labels['steps']], rtol=0., atol=1e-10)
        accepted.append(case)
    assert len(accepted) == corrections['accepted_cases']
    if accepted:
        audits = [json.loads((RUN/m/'input.json').read_text()) for m in ('base_finetune','corrected')]
        assert audits[0]['samples_per_epoch'] == audits[1]['samples_per_epoch'] == len(data['features'])
        assert audits[0]['epochs'] == audits[1]['epochs'] == plan['correction']['finetune_epochs']
        for m, a in zip(('base_finetune','corrected'), audits):
            study.write_json(OUT/'evidence'/(m+'-input.json'), a)
    study.write_json(OUT/'evidence/comparison.json', comparison)
    study.write_json(OUT/'evidence/correction-admission.json', corrections)
    failures = {m:dict(collections.Counter(f for r in v['reports'] for f in r['failures_strict'])) for m,v in rows.items()}
    study.write_json(OUT/'evidence/failures.json', failures)
    preferred = min(modes, key=lambda m:(-rows[m]['summary']['task_fraction'],
                    -rows[m]['summary']['strict_fraction'], rows[m]['summary']['mean_goal_m']))
    study.write_json(OUT/'evidence/development-selection.json', dict(
        protocol_sha256=study.digest(PLAN), method=preferred, policy=rows[preferred]['policy'],
        policy_sha256=rows[preferred]['policy_sha256'], summary=rows[preferred]['summary'],
        selection='equal-video task fraction, strict fraction, then mean goal error',
        development_only=True, independent_holdout_evaluated=False, old_models_replaced=False))
    regressions = []
    if 'corrected' in rows:
        baseline = {(r['video'],r['case']):r for r in rows['base_finetune']['reports']}
        for r in rows['corrected']['reports']:
            before = baseline[(r['video'],r['case'])]
            if before['task_pass'] and not r['task_pass']:
                regressions.append(dict(video=r['video'], case=r['case'],
                    failures=r['failures_strict'],
                    baseline_peak_m=before['report']['max_hand_scene_penetration_m'],
                    corrected_peak_m=r['report']['max_hand_scene_penetration_m'],
                    peak_contact=r['report']['scene_contact_peaks'][0]))
    study.write_json(OUT/'evidence/correction-regressions.json', regressions)
    lines = ['# v13 结果：分工策略与物理验证纠偏', '',
        '日期：2026-10-02。全部为开发工况结果，不是独立留出或实物泛化证明；本轮不新增截图。', '',
        '## 共享与分工对照', '', '| 方法 | 任务级 | 严格级 | 第二视频任务/严格 | 等权目标误差 |',
        '|---|---|---|---|---|']
    names = dict(shared='共享 v12', video='分视频', phase='分阶段', base_finetune='仅原数据再训20轮', corrected='加入纠偏再训20轮')
    for mode in modes:
        s = rows[mode]['summary']; second = s['per_video']['second']
        lines.append('| %s | %d/35 | %d/35 | %d/17 / %d/17 | %.2f mm |'%(
            names[mode],s['task_count'],s['strict_count'],second['task'],second['strict'],s['mean_goal_m']*1000))
    lines += ['', '分工对照按原协议排序，选中 **%s** 用于纠偏。分工版本总参数量增加，且只有一个训练种子，不能据此证明路由结构的独立因果优势。'%names[comparison['selected']['mode']],
              '', '包含两组微调后，开发集首选候选为 **%s**；未替换旧模型，尚无独立留出验证。'%names[preferred],
              '', '## 第二视频分阶段偏离', '',
              '下表为相对各工况合格专家的杯子位置 RMSE，按第二视频 17 工况平均。', '',
              '| 方法 | 接近 | 闭合 | 抬杯 | 搬运 |', '|---|---|---|---|---|']
    for mode in modes:
        rr = [r for r in rows[mode]['reports'] if r['video']=='second']
        values = [np.mean([r['phases'][p]['cup_position_rmse_vs_expert_m'] for r in rr])*1000
                  for p in ('approach','closure','lift','transport_hold')]
        lines.append('| %s | %s |'%(names[mode], ' | '.join('%.2f mm'%v for v in values)))
    lines += ['', '## 轨迹误差与接触验收不一致', '',
        '纠偏模型在第二视频各阶段的平均杯子轨迹误差更小，但相对无纠偏微调新增 %d 个任务失败。以下记录具体接触峰值；不能仅凭更低的模仿误差认定控制改善。'%len(regressions), '',
        '| 工况 | 原/纠偏后峰值 | 最深接触对 | 时刻 |', '|---|---|---|---|']
    for r in regressions:
        lines.append('| %s | %.3f / %.3f mm | %s | %.3f s |'%(r['case'],r['baseline_peak_m']*1000,
            r['corrected_peak_m']*1000,' / '.join(r['peak_contact']['geoms']),r['peak_contact']['time_s']))
    lines += ['', '当前证据支持下一轮把接触安全余量直接纳入纠偏专家优化与训练验收；尚未证明哪个损失或控制器能解决此问题。',
        '', '## 纠偏数据准入', '',
        '选取 %d 个第二视频开发失败工况，%d 个通过完整任务物理门槛、确定性重放与半步长复核，共收录 %d 帧实际执行的专家标签。'%(
            len(corrections['reports']),corrections['accepted_cases'],corrections['label_frames']), '',
        '| 工况 | 接管步/反馈增益 | 标签帧 | 原步长/半步长严格级 | 峰值穿透 |', '|---|---|---|---|---|']
    for case in corrections['reports']:
        a = case['accepted']
        if a:
            lines.append('| %s | %d / %.2f | %d | %s / %s | %.3f mm |'%(case['case'],a['start_step'],a['gain'],
                a['label_frames'],a['strict_pass'],a['half']['strict_pass'],a['report']['max_hand_scene_penetration_m']*1000))
        else:
            lines.append('| %s | %d 次尝试后未准入 | 0 | 未通过 | 未收录 |'%(case['case'],case['attempts']))
    lines += ['', '专家接管轨迹成功不能算学生成功。纠偏只在闭合及抬杯初段提供标签，未混入过渡期未执行的建议动作；训练效果必须与等预算无纠偏微调对照比较。',
              '', '本轮准入轨迹全部使用反馈增益 0，即从真实学生前缀平滑切回该工况已验证的专家动作。它们不是新合成的人类力标签，也没有证明状态反馈修正器能可靠恢复大偏离。最接近上限的一条穿透约 0.999 mm，只满足当前最低门槛，安全余量不足。',
              '', '## 边界', '', '- 原 v9-v12 数据与模型保留；未调整物理参数或运行中的物体位姿。',
              '- 未启动长训练，未读取或重新调试旧留出集；本轮全部模型只在开发集比较。',
              '- 程序专家不是人类力标签，分工路由也不是语言任务规划器。',
              '- 未新增截图或重新检查桌面交互窗口。', '', '[方法与命令](README.md) · [结构化核验](evidence/verification.json)']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    print(stream.getvalue()[-1800:])
    assert result.wasSuccessful()
    subprocess.check_call(['git','diff','--check'], cwd=str(ROOT))
    study.write_json(OUT/'evidence/verification.json', dict(date='2026-10-02', tests=result.testsRun,
        skipped=len(result.skipped), success=True, development_rollouts=len(modes)*35,
        policy_and_data_hashes_checked=source_hashes, accepted_corrections=len(accepted),
        corrective_label_frames=corrections['label_frames'], routing_and_sampling_verified=True,
        label_execution_verified=True, independent_holdout=False, new_screenshots=0, long_training=False))
    print('VERIFIED',len(modes)*35,'development rollouts;',len(accepted),'admitted corrections',flush=True)


if __name__ == '__main__':
    main()
