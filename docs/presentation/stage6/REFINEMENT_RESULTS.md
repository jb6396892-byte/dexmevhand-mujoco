# 阶段 6 答辩材料：从模型建议到受控技能执行

## 一句话成果

在真实视频衍生的两套已验证专家参考技能上，完成中文指令规划、对比 LoRA、语义一致性门控、物理验收和受控执行。
最新 100 条合成中文测试中，原始语义正确率由旧 LoRA 的 80% 提升到 94%，门控后越界误接受为 0。
不是端到端视觉语言动作模型，也不是无限开放语言安全系统。

## 推荐四页 PPT

### 1. 问题：JSON 合法不等于任务正确

旧失败：“将杯子放到我手上”被解释为 `lift`。
`reach -> grasp -> lift` 本身合法，也能在标准仿真中运行，但没有完成交接任务，因此仍然是错误计划。

改进前只有 Schema、前置技能和历史物理余量检查，无法发现原指令与合法计划之间的语义错配。

### 2. 方法：对比学习 + 独立否决 + 物理闭环

```text
视频手/物体轨迹 -> 重定向与物理优化 -> 可执行专家技能参考
                                                  ^
中文 -> Qwen + LoRA -> JSON -> 原指令一致性 -> 物理余量 -> 技能执行器
                                         拒绝越界       env.step
```

训练数据由 800 行扩充到 1160 行，增加仅接近/抬杯、抓稳/交接、搬运/放置等对比。
LoRA 4 epoch、580 更新，按验证 loss 选择 checkpoint；门控只能拒绝，不能替模型编造正确答案。
词汇契约初版误拒绝过多，保留失败后另起 guard2，只修复语法等价识别，权重完全不变。

核心逻辑，来自 `src/fromrealhand/language_planner/guard_revision.py`：

```python
if not contract['allowed']:
    reason = 'instruction_not_admitted'
elif gate['response']['plan']['goal'] != contract['goal']:
    reason = 'instruction_plan_mismatch'
```

### 3. 结果：区分模型收益与门控收益

| 最终新 100 条语言测试 | 旧 LoRA | 新 LoRA |
|---|---:|---:|
| 原始语义正确 | 80 | 94 |
| 原始 Schema 合法 | 98 | 100 |
| 仅原门控时越界误接受 | 18 | 6 |
| 添加同一 guard2 后越界误接受 | 0 | 0 |
| 合法任务正确接受 | 50/50 | 50/50 |

这说明微调改善原始输出，门控负责阻止仍存在的错误。不能说微调本身保证零越界。
240 条旧输出回归：236 条原始语义正确，140 条合法任务全部放行、100 条不支持任务全部拒绝。
回归不当作独立测试。新测试只有 50 个表达乘两个已知场景，不代表新物理场景泛化。

### 4. 可执行性与边界

实际模型输出去重为 10 个计划：8 个动作、2 个停止，物理检查全部通过。
第一/第二视频搬运最大穿透为 0.577/0.901 mm，保持原 1 mm 门槛；末端目标距离 0.853/3.858 mm。
执行中无状态写入，杯子由接触动力学运动，不是每帧设置物体位置。
底层是已验证专家参考，不是第 200 次 DAPG 网络或独立学成的四个技能网络。

![实际语言模型计划在第二视频标准场景完成搬运](evidence/study-v4-guard2/second-transport.jpg)

截图中绿色区域是目标标记；画面来自第二视频 `transport` 完成时。这里只新增这一张物理截图。

仍不支持：倒水、放置、交接给人、任意目标坐标、未知物体与新场景。
语言模型仍可能误解输入，因此保留原始结果、保守拒绝、超时和实时物理停止，不声称通用安全证明。

## 演示命令

```bash
cd /home/smgbro/mujoconew/GITHUB
# 只生成和检查计划。
bash scripts/116_run_stage6_model.sh "抓起杯子" --scene second
# 原生 MuJoCo 窗口，关闭窗口退出。
bash scripts/116_run_stage6_model.sh "把杯子搬到目标位置" --scene second --execute --render
# 预期拒绝；不创建仿真、不施加动作。
bash scripts/116_run_stage6_model.sh "将杯子放到我手上" --execute
```

冻结产物和共享盘必须存在。`--execute` 还要求验收收据与模型、数据、预测及门控代码哈希匹配。

## 证据索引

- [最终验收收据](evidence/study-v4-guard2/acceptance.json)
- [新模型新测试逐条结果](evidence/study-v4-guard2/candidate-heldout.json)
- [旧模型相同新测试对照](evidence/study-v4-guard2/previous-heldout.json)
- [240 条回归](evidence/study-v4-guard2/candidate-regression.json)
- [物理执行](evidence/study-v4-guard2/physical.json)
- [正式入口实测](evidence/study-v4-guard2/entrypoint-smoke.json)
- [175 项回归测试](evidence/study-v4-guard2/quality.json)
- [具体训练和冻结方法](../../STAGE6_REFINEMENT.md)
- [本轮命令与文件记录](../../run_logs/2026-10-02-stage6-refinement.md)

方法参考：[Qwen 官方模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、
[PEFT LoRA](https://github.com/huggingface/peft/tree/v0.14.0)、[SayCan](https://say-can.github.io/)。
复用既有开源方法并增加工程边界，不把本项目描述为已验证的新通用机器人基础模型。
