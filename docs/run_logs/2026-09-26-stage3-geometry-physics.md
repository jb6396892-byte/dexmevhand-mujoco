# 阶段 3：几何修复与自由杯子物理验收

日期：2026-09-26。结论：完成诊断与第一轮修复，物理验收未通过，未启动新 DAPG 训练。

## 验收边界

- 数据格式：新诊断示范包含 278 步，观测 39 维、动作 30 维以及 rewards/sim_data/model_data；字段检查完成，但动作饱和率 73.51%，默认质量检查失败。
- 策略能运行：此前的 20 次迭代模型能运行不等于成功抓取；本轮没有重新训练或宣称现有策略改善。
- 真实物理抓取：四组动作验证均未通过。物理段只初始化一次状态，之后全部通过 env.step(action)，没有逐帧修改杯子位姿。

## 基线诊断

输入 seq_dexycb_001 的源帧 20 至 73，共 54 帧。原数据、原示范、checkpoint 保留。

产物：`data/processed/seq_dexycb_001/geometry_audit_20260926/`。
包含逐帧 frames.csv、碰撞几何 contacts.json、summary.json，以及源图投影和仿真关键帧。

- 首次超过 1 mm 的穿透为源帧 26：C_thproximal 对 collision_mug_45，约 3.775 mm。
- 最严重为源帧 73：拇指近节对 collision_mug_46，约 29.313 mm；48/54 帧超过 1 mm。
- 源杯子包围盒约 116.966 × 93.075 × 81.384 mm；仿真视觉网格确实为其 0.8 倍。
- 将仿真网格除以 0.8 后，与源网格最近点平均差 2.31e-9 m，排除了不同杯子网格原点这一候选原因。
- 手关节投影与提供的 2D 标签差为 0 px。这只验证标签和投影计算自洽，不能当作独立人工标注的标定精度证明；关键帧仍需人工复核。
- 源帧率为 30 FPS，上游示范生成器默认按 25 FPS 处理，并仅平滑手轨迹。

最可能的组合原因：原重定向去掉了碰撞几何，主要拟合六个手部参考点而非抓持接触；杯子缩小但手部目标没有同样缩放；任务朝向不适合 Adroit 根关节范围，旧 ARTz 在所检帧全部卡在 0.5 rad。不是单独平移杯子即可解决。

## 独立修复

新增 scripts/22_repair_geometry.py，产物在 `data/processed/seq_dexycb_001/repair_v2/`。
第一轮 repair_v1 也保留，不覆盖旧 retargeting 和 demonstration。

1. 同时对手与杯子的目标应用 0.8 相似变换，并由初始腕部到杯子的方向确定任务绕 z 轴旋转，约 -189.799 度。
2. 根据杯子碰撞网格最低点，将整个任务向下移动 6.038 mm，使杯底初始距桌面 1 mm；没有单独挪杯子改变手杯关系。
3. 保留真实碰撞几何，以六个参考点、五个指尖、穿透惩罚和时间正则项优化 30 个手关节；以上一帧解热启动，最多 180 次优化迭代。
4. 54 帧均优化收敛，最大穿透 0.560 mm，无超过 1 mm 的采样帧；平均参考点误差仍有 11.43 mm。这是几何改善，不是已达到精确抓持。
5. 保存 retargeting.pkl、geometry.npz、frames.csv、summary.json 和六张关键帧图。

重要限制：插值到 100 Hz 后，最大运动学穿透仍达 5.913 mm。逐帧约束没有覆盖帧间路径，不能直接把这些姿态当成可执行示范。

## 动作恢复与物理结果

新增 src/fromrealhand/action_recovery.py 和 scripts/23_validate_repaired_actions.py。
新链路在 forward 之后恢复期望 qacc，再调用逆动力学；考虑执行器完整仿射偏置与增益，归一化一次后交给 env.step，未重复缩放。
例如根执行器 gain=500、位置 bias=-200，零力平衡控制量是 0.4*q，而不是 q。
这些改动仅用于独立新链路；旧 demo_builder 及原示范没有自动替换。

物理环境与训练保持 object_scale=0.8、friction=(1,0.5,0.01)、solref='-6000 -300'。
最终权威结果在 `repair_v2/dynamics_forces/summary.json`，含四段 MP4 和逐控制步 CSV。
环境有 3 mm margin，因此接触以物理子步法向力 > 0.001 N 判断，不能简单使用 dist<=0。
之前 dynamics/ 和 dynamics_substeps/ 的零距离接触计数仅保留为排查过程，不用于最终结论。

| 动作方式 | 步数 | 有力接触步数 | 最高杯中心 z (m) | 持续接触抬杯 (s) | 最终目标距离 (m) | 饱和率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 逆动力学开环 | 278 | 8 | 0.06534 | 0.03 | 5.7661 | 73.51% |
| 反馈跟踪 | 278 | 12 | 0.05016 | 0 | 0.1874 | 26.70% |
| 反馈增益加倍 | 278 | 10 | 0.05585 | 0 | 0.3950 | 50.47% |
| 反馈慢速两倍时长 | 455 | 4 | 0.07378 | 0.01 | 0.2134 | 26.12% |

初始杯中心 z=0.034612 m。开环物理最大穿透 2.039 mm，其他三组未出现负接触距离；有 margin 接触仍可能推动杯子。
本轮门槛：有力接触且杯中心较初始升高 15 mm，连续至少 0.5 s，同时最大手杯穿透不超过 5 mm、没有运行异常。
四组全部失败。最高点反映短暂运动，不能说明抓持。门槛也只是抬杯门槛，不代表完整 relocate 成功。
诊断 pkl 名称为 relocate-mug-repair-diagnostic.pkl，明确 training_eligible=false，没有生成通过门槛的 physical_demo.pkl。

## 复现命令

在项目根目录执行，输出目录请使用新名称以保留已有结果。

```bash
cd /home/smgbro/mujoconew/GITHUB
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
conda activate dexmv
python scripts/21_diagnose_geometry.py --output data/processed/seq_dexycb_001/geometry_audit_20260926
python scripts/22_repair_geometry.py --iterations 180 --output data/processed/seq_dexycb_001/repair_v2
python scripts/23_validate_repaired_actions.py data/processed/seq_dexycb_001/repair_v2/geometry.npz --output data/processed/seq_dexycb_001/repair_v2/dynamics_forces
python scripts/06_validate_demo.py data/processed/seq_dexycb_001/repair_v2/dynamics_forces/relocate-mug-repair-diagnostic.pkl
PYTHONPATH=src python -m unittest discover -s tests -v
```

实际使用 /home/smgbro/miniconda3/bin/conda run --no-capture-output -n dexmv 执行等价命令。
物理验证返回 1 是验收失败的预期行为，不是 Python 崩溃；格式脚本因饱和率超过默认 40% 返回 1。
11 项单元/集成测试通过，存在旧 chumpy/NumPy 弃用警告。四段视频可解码，前三段各 70 帧、慢速 114 帧。

## 修改文件与下一步

本轮新增：scripts/21_diagnose_geometry.py、scripts/22_repair_geometry.py、scripts/23_validate_repaired_actions.py、src/fromrealhand/action_recovery.py、tests/test_action_recovery.py、本文。
未清理或回退现有 dirty worktree；未下载数据、升级环境、修改已有 checkpoint；未提交或推送 Git。

下一轮按以下次序修复，暂不扩大训练：

1. 对 100 Hz 连续路径而非仅 30 Hz 源帧施加碰撞、速度、加速度与执行器可行域约束，消除插值穿透和高饱和动作。
2. 从静止初态分离验证“接近、闭合、保持”，记录各指接触力和杯底离桌高度；当前轨迹没有维持力闭合，不能只调高 PD 增益。
3. 用自由杯子的反馈轨迹优化或接触感知控制生成动作；检查腕根角度边界、指尖误差、初始速度与接触冲量。不要再以强制杯子轨迹的逆动力学输出直接训练。
4. 单条持续抬杯通过后，独立重放保存动作并检查多次复位；再生成正式示范、运行 20 次 DAPG 和多种种子评估。
