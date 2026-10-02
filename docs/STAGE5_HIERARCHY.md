# 阶段 5：层次化技能执行系统（C4 + D1）

按用户 2026-10-02 最新指定，本阶段是**技能注册、规则规划与可靠执行框架**，覆盖 C4 + D1。此前将“阶段 5”称为低层技能训练的安排不再作为本次范围；C3 技能策略学习另列后续，未启动新训练。正在进行的 200 次 DAPG 仍是整体抓杯任务。

## 子阶段与验收

| 顺序 | 工作 | 产物 | 当前结果 |
|---|---|---|---|
| 5.1 | 注册技能与校验计划 | `configs/skill_registry.yaml`、`hierarchy/registry.py` | 完成；四技能、两个名义场景、严格 Schema，拒绝重复字段及未知参数 |
| 5.2 | 接入前置、成功、安全条件 | 复用冻结的 `SkillContract` / `GuardedSkill` | 完成；不改原 1 mm 穿透门槛，连续执行不重置 |
| 5.3 | 超时、重试、停止、剩余计划重校验 | `hierarchy/executor.py` | 完成；16 项仿真用例满足预期，包含中途停止，未证明滑落后重抓 |
| 5.4 | 中文规则规划 | `hierarchy/planner.py`、`scripts/16_run_instruction.py` | 完成；15 条别名正确映射，6 条不支持指令被拒绝 |
| 5.5 | 端到端与资料 | `scripts/15_run_skill_plan.py`、验证脚本、两张图 | 完成；正常固定计划 4/4，全量 142 项单元测试通过 |

## 系统结构

```text
中文指令
  -> 规则规划器（精确别名，不是 LLM）
  -> JSON Schema + 技能注册表
  -> 顺序执行器（前置条件、预算、失败路由）
  -> 已验证专家动作适配器
  -> env.step(normalized_action)
  -> MuJoCo 真实接触/杯底/目标距离反馈
  -> 成功、停止、有限重试或剩余计划重校验
```

当前后端明确叫 `verified_reference`：调用阶段 4 通过物理验证的专家动作，不加载长训中的 checkpoint，不是四个新学成的技能网络。高层决定调用哪些技能，底层动作数值仍来自冻结参考；反馈用于判断安全与完成，不会自动生成新的抓取动作。

注册表只接受 `mug`、`first/second` 两个名义场景和已知目标。目标位置采用该场景预设值，暂不接受任意坐标、倒水或放手命令。后端仍依赖原参考时钟，只允许 `reach` 开始的有序前缀；不能随机拼接独立技能片段。

## 条件与停止语义

| 技能 | 前置条件 | 成功条件 |
|---|---|---|
| reach | 状态有限、无超限 | 有效手杯接触连续 5 步 |
| grasp | 已接触杯子 | 至少三指承力且含拇指，连续 10 步 |
| lift | 已形成上述支撑 | 保持支撑，杯底至少 50 mm，连续 10 步 |
| transport | 满足抬杯条件 | 杯底至少 15 mm、目标距离不超过 30 mm，支撑保持 100 步且末帧仍满足 |

接触阈值 0.01 N，穿透上限 1 mm，关节超限上限 0.02 rad。抬杯/搬运持续失去支撑 20 步则停止。成功必须同时满足实际事件和参考片段完整执行，不能为了提前切换而跳过原动作时钟。

- **物理失败**：穿透、非有限状态、关节超限、入口不满足或持续丢失支撑，不盲目重试，停止后续仿真步。
- **暂时取不到动作**：最多重试 2 次，保持当前游标；不重放已执行动作，不重置场景，不清空技能步数预算。
- **重新规划请求**：最多 1 次，只能在技能边界重新检查当前安全、入口条件、已完成技能和剩余前缀。不能后退时钟；没有经过验证的恢复路径就停止。当前不是为物理滑落生成新路径。
- **超时**：技能步数、总步数 3200、墙钟 120 秒三重预算。墙钟检查是协作式的，不能强制中断卡死的底层 C 调用。
- **安全停止**：停止继续调用 `env.step`；只适用于本机仿真，不等于实体机器人急停。检测穿透后停止不代表此前没有发生过穿透。
- **窗口结束**：最后画面保持供检查，不继续积分物理；关闭窗口退出。保持画面不能作为持续抓稳的新证据。

每次执行记录 `plan.json`、`report.json` 和 `trace.npz`。报告包含技能事件、原时钟、终止原因、重试/重规划次数、物理指标和来源。`Ctrl+C` 或关闭原生窗口会停止该次执行，不影响后台训练服务。

“停止/取消”规则只取消本次新计划，不会给另一个运行进程发送信号；正在执行的原生窗口用关闭窗口或该终端的 `Ctrl+C` 停止。

## 使用命令

```bash
cd /home/smgbro/mujoconew/GITHUB
conda activate dexmv

# 只看计划，不需要创建 MuJoCo 环境。
python scripts/16_run_instruction.py "抓起杯子" --dry-run
python scripts/15_run_skill_plan.py --dry-run

# 从终端打开原生 MuJoCo 窗口，不是 MP4。
bash scripts/101_view_skill_plan_gpu.sh "抓起杯子" --scene first
bash scripts/101_view_skill_plan_gpu.sh "把杯子搬到目标位置" --scene second
```

无窗口执行或离屏截图：

```bash
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python scripts/15_run_skill_plan.py --plan configs/skill_plan_pickup.json
python scripts/16_run_instruction.py "抓起杯子" --scene second
python scripts/90_supervise_training.py status
PYTHONPATH=src python -m unittest discover -s tests -q
```

完整仿真验收使用 `scripts/100_validate_hierarchy.py`，默认产物已经存在，重跑必须指定新的 `--output` 和 `--publish` 目录，不能覆盖旧结果。截图需 `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`。本轮自动验收使用离屏 GPU 渲染，没有操作用户桌面的原生窗口。

## 下一步

1. 等 200 次整体任务训练结束，先检查末次策略的两视频名义评估，不把训练采样统计当独立泛化。
2. C3 低层学习：接入技能条件残差 BC 小试，替换专家后端前重新验证观测、时钟、技能终止和连续衔接。
3. 优先补充可达接触恢复数据。目前只有规则布尔前置判断，没有学习到的技能成功概率，也没有可靠的滑落重抓控制器。
4. 新控制器通过后，再做真正的状态驱动恢复规划与新的独立评估；高层 LLM 训练仍是后续工作。

[完整结果与答辩提纲](presentation/stage5/README.md) · [本轮日志](run_logs/2026-10-02-stage5-hierarchy.md)
