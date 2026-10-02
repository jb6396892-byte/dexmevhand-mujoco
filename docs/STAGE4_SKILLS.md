# 阶段 4：四技能数据与执行接口

本页的阶段编号对应 [ROADMAP](ROADMAP.md)。当前完成的是技能框架及两条名义示范的回放验证，不是四个独立技能策略训练，也不是高层语言规划器。

## 本轮实现

- `configs/stage4-skills.json`：输入、输出、事件阈值、动作维度和超时预算。
- `src/fromrealhand/skills.py`：自动分段、片段校验、技能契约和连续参考动作接口。
- `scripts/91_build_skill_dataset.py`：重新物理执行已准入示范，导出技能片段。
- `scripts/92_validate_skill_replay.py`：独立初始化回放与不重置的连续拼接检查。
- `scripts/93_build_stage4_evidence.py`：汇总小型证据 JSON 与一张阶段曲线图。

不改动 v14c 冻结文件、原示范、已有 checkpoint 或正在训练的阶段采样逻辑。新的物理事件标签只用于阶段 4。

## 分段定义

控制频率 100 Hz；以下接触事件需连续满足 5 步，排除瞬时碰撞。接触力来自仿真求解器，不是视频中的真实接触力标签。

| 技能 | 启动条件 | 结束事件 |
|---|---|---|
| reach | 有限、未超安全上限的初态 | 至少一个手指对杯子法向力大于 0.01 N |
| grasp | 已发生有效接触 | 至少三指承力，必须含拇指 |
| lift | 满足上述对握支撑条件 | 保持支撑，杯底离桌至少 50 mm |
| transport | 已达到抬升条件 | 杯底至少 15 mm、距目标不超过 30 mm、保持支撑 100 步，且片段末尾仍满足 |

这只是可解释的技能标注与运行契约。三指对握并不意味着视频的四指抓法已完全复现，三指法向力也不是力闭合的数学证明。`reach` 包含原序列的初始等待；到达事件是在确认持续接触后才切换。

每步同时检查有限性、手与场景穿透不超过 1 mm、关节超限不超过 0.02 rad。技能启动失败、事件未达成、超时、安全失败分别记录，安全失败优先于成功。整条源示范另外复核原有 surface 物理门槛，不用片段契约替代全部抓法验收。

## 当前边界与验证

区间统一为控制动作的 `[start, stop)`，对应状态从 `start` 到 `stop`。源帧是参考动作对应的原视频时钟，不是将仿真控制步直接当作视频帧。

| 视频 | reach | grasp | lift | transport |
|---|---|---|---|---|
| first/video_seed_0 | [0,512) | [512,538) | [538,619) | [619,1417) |
| second/nominal | [0,550) | [550,582) | [582,694) | [694,1450) |

自动边界全部标记为 `review_status: pending`，尚未获得用户人工确认。当前仅使用开发数据中的两条名义轨迹，不读取独立留出数据。后续已按 [五个子阶段](STAGE4_PLAN.md) 推进 4.1，完成自动检查及助手看图复核；发现第二视频搬运入口短窗稳定性警告，详见 [第一子阶段报告](presentation/stage4/BOUNDARY_REVIEW.md)。

实测：8/8 个片段独立回放、2/2 条连续拼接通过。源示范观测复现误差为 0；独立片段最大 qpos/qvel 绝对误差为 `1.715e-13`，验收上限 `1e-7`。第一、第二视频全程最大手与场景穿透约 0.577 / 0.901 mm，最终目标误差约 0.853 / 3.858 mm。两条源示范原物理与忠实度检查均通过。

## 数据格式

本地产物：`data/processed/stage4_skills_v1/`。manifest 保留输入哈希、几何哈希、控制参数、视频 ID、片段边界和逐片段文件哈希。

每个技能 `.pkl` 包含：

```text
observations[T, obs_dim]      动作执行前的观测
actions[T, 30]               原有归一化手部动作，不二次缩放
rewards[T], sim_data[T]      与原 demonstration 兼容的字段
model_data, physics_model   场景与接触参数
initial_snapshot            时间、qpos/qvel、控制、求解器 warm start、环境计数器
terminal_snapshot           完整结束状态
terminal_observation        最后一个动作之后的观测
initial_metrics, metrics[T] 接触、杯底、目标距离和子步安全峰值
expected_post_states[T, 73] 参考执行后的 qpos/qvel，用于确定性验证
metadata                    视频、动作区间、原帧范围、人工审核状态
```

恢复求解器 warm start 很重要，仅恢复 qpos/qvel 不足以作为精确接触重放的完整约定。文件是可信本地 pickle，不要加载来历不明的 pickle。

独立片段允许在回放开始恢复一次初态，这与环境 reset 的含义一致。连续拼接只在整条轨迹开头恢复一次；技能切换时不恢复任何状态。执行期间只调用 `env.step(action)`，杯子自由运动，不施加物体辅助力。回放类明确叫 `SegmentedReference`，不冒充学习策略。

## 复现

```bash
cd /home/smgbro/mujoconew/GITHUB
conda activate dexmv
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# 原目录已有本轮证据；复跑使用新的目录，脚本拒绝覆盖。
python scripts/91_build_skill_dataset.py --output data/processed/stage4_skills_repeat
python scripts/92_validate_skill_replay.py --dataset data/processed/stage4_skills_repeat
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 下一步

1. 人工核对两个视频的事件边界，重点看第二视频的对握形成及后续无名指加入时刻。
2. 在已验收的其余开发轨迹上扩展分段，保留失败记录；不能把毫米扰动副本算作新的真实视频来源。
3. 为每个技能补充观测适配器、目标条件、入口状态扰动与恢复数据，训练独立或条件化技能策略。这属于阶段 5，不把整条参考动作切片直接称为学会了可复用技能。
4. 等本轮 200 次 DAPG 完成后先看训练后名义评估，再冻结候选、预注册新的独立评估。训练采样通过率不是留出成功率。

[答辩材料与证据](presentation/stage4/README.md) · [200 次训练监督](TRAINING_200.md)
