> 本轮已在五小时额度记录达到 85% 时停止新搜索。末批 `slower_soft_control`：6 个初态都可四指抬杯，5/6 通过完整场景门槛，最坏穿透 1.046 mm 超过 1 mm，整体未准入；保存动作重放误差 0，旧资产哈希未变。
> 标称末态平均指尖误差 19.51 mm、拇指 30.67 mm，严格视频忠实度仍未通过。没有启动训练。
> 实验查看入口：`bash scripts/40_view_scene_candidate_gpu.sh`。窗口默认循环，Ctrl+C 退出。


# 完整场景约束与视频抓法续优化

## 更正旧验收范围

旧的 `finger_fidelity_v1` 仅通过手杯接触检查。本轮重新执行保存动作，检测到初态最大场景穿透 10 mm，涉及手桌接触，并存在手指间碰撞；新门槛判定失败。原始报告作为历史记录保留，不能再用其 `surface_physics_passed=true` 宣称完整场景合格。

证据位于 `data/processed/seq_dexycb_001/finger_fidelity_v1/scene_audit/`。源帧 3 的主相机看不到手，而变换后的部分人手指尖低于仿真桌面；真实桌沿位置和初始标签可信度还需要多视角/深度核验。不能据此任意平移杯子或移动桌面。

## 本轮实施

- 从旧候选续跑 45 个小角度修正候选，未解决完整场景问题。
- 重定向增加 `--scene-collisions`：手杯、手桌、启用的手指自碰撞共同优化。提高迭代上限后，71 帧全部收敛，几何平均指尖误差 5.79 mm，初态最大穿透 0.123 mm。
- 连续动作在每个物理子步统计手涉及的全部接触穿透。新增初态 ≤0.5 mm、全过程 ≤1 mm 的门槛，原手杯正间距、滑移、饱和等标准保持。
- 场景控制先做闭合量/增益搜索，再做 67 个逐指修正候选，按初态 0、1 的较差分数选取；随后将失败初态 2 加入控制搜索。初态 0、1、2 属于调参数据，不能称为独立测试集。
- 增加慢速轨迹试验及较小闭合量试验。所有结果独立保存，杯子初始化后自由运动。
- 训练准入加入几何帧收敛检查；缺失场景审计字段的报告默认不能通过新场景门槛。
- 30 项测试通过，包括拒绝旧穿桌初态的回归测试。

## 当前边界

完成初态修正后，多个候选可以四指连续抬杯，但视频忠实度仍不足。`gain_refine` 候选的末态均值约 19.95 mm、拇指约 31.24 mm，且种子 2 的场景穿透为 1.014 mm，不能放行。峰值发生在约 3.51 s 的接近阶段，早于主要闭合修正。

`slower_control` 将时间伸缩参数从 4 增至 5，标称末态均值约 19.36 mm，仍有初态场景穿透超过 1 mm。末批 `slower_soft_control` 进一步降低闭合量，准确结果见其 `admission.json`。所有候选均保留 `training_ready` 判定，不启动 DAPG。

旧版约 14.94 mm 的末态误差与新版不能单独按数字比较：旧版未满足新增完整场景约束。下一步应针对接近阶段的手指自碰撞、腕部执行器饱和和接触后拇指误差做分阶段优化，并检查真实桌沿和标签可见性。

## 结果路径与复现

主要目录：`data/processed/seq_dexycb_001/scene_fidelity_v2/`。

- `retarget/`：全部收敛的完整场景参考。
- `control/`：初始闭合量/反馈增益搜索。
- `refined_r1/`：67 个多初态逐指修正候选。
- `gain_refine/`、`robust_control/`：联合控制参数试验。
- `slower_control/`、`slower_soft_control/`：慢速及减小闭合量试验。

在配置好的 dexmv 环境中，用新输出目录续接：

```bash
python scripts/29_build_video_faithful.py --scene-collisions --iterations 180 --output NEW_RETARGET_DIR
python scripts/37_optimize_finger_reference.py \
  --candidate data/processed/seq_dexycb_001/scene_fidelity_v2/control/admission.json \
  --initial-correction data/processed/seq_dexycb_001/scene_fidelity_v2/refined_r1/best_parameters.json \
  --output NEW_CONTROL_DIR --search-seeds 0 1 2 --steps 0.015 0.0075 --quota-stop 85
```

额度使用本地最新 300 分钟窗口记录监测，85% 为停止新搜索的阈值。记录可能延迟；候选和最优参数逐次保存。控制搜索脚本 33 与逐指搜索脚本 37 都支持此机制。
