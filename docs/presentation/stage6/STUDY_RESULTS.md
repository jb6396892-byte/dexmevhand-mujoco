# 阶段 6 实际训练与独立测试

> 本页保留 v3 的训练和拒收记录。最新 `study_v4_guard2` 受控验收已通过，见 [最新答辩材料](REFINEMENT_RESULTS.md)；不得把本页历史 1 条越界误接受当作新版本结果。

2026-10-02。**隔离环境、正式 LoRA 和独立对照已经完成；语言安全门槛未过，模型暂不准自动执行。**
下面全部为实际模型生成，不是规则回退或标准答案模拟。失败结果和首次失败小试保留。

## 存储与训练

- 语言相关大文件仅在 `/media/smgbro/shared/lora/language/`，阶段 3 不放这里。
- 基础模型：`model/`，Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`。
- 正式 adapter：`study_v3/formal/adapter/`；主权重 SHA256 `7b9c33d183768ada07186ece43bb0189ea5f16addd6a260732c770820ceb3386`。
- 旧 v2 小试：`study_v2/`，未覆盖；正式报告、原始输出、物理验收在 `study_v3/`。
- 中文资料和模型索引副本：`delivery-study_v3/`，索引中的 `accepted=false`、`deployment_allowed=false`。
- 共享盘依赖目录实际占约 12 GB，基础模型约 959 MB；exFAT 的分配单元使许多小文件占用放大。
- 原 dexmv 保持 Python 3.7 / torch 1.13.1；现代语言包独立安装。Linux 仅放约 65 MB 引导环境。

正式训练为 800 条样本、8 epoch、800 个更新，GPU 用时约 524 秒；不是额外 800 次 DAPG。
rank=16、alpha=32、q/k/v/o 投影，仅更新 2,162,688 个 LoRA 参数；batch=2、梯度累积=4、LR=2e-4。
最长序列 338 token，仅 assistant JSON 计损失。峰值分配显存 2,521,321,984 字节；GPU 总占用与此不同。
按预先约定的验证 loss 选第 500 步（第 5 轮）权重，随后才打开留出集；没有按留出集选模型。

## 从失败小试到正式训练

| 验证集，42 行 | 语义正确 | JSON 合法 | 不支持指令误接受 |
|---|---:|---:|---:|
| 基础模型 | 9/42 | 11/42 | 0 |
| 原 504 条训练的小试 v2 | 34/42 | 42/42 | 1 |
| 增强至 800 条的小试 v3 | 42/42 | 42/42 | 0 |
| 正式 8 epoch | 42/42 | 42/42 | 0 |

首次错误集中在复合意图、停止和越界动作。补充 74 组人工对比表达，经两场景与两种包装增加 296 行。
超参数、系统提示、门槛保持不变；800 行输入没有精确重复，验证和留出句子没有复制进训练。
验证集用于开发，因此其 100% 不能当作独立泛化证明。

## 独立留出结果

测试是预先冻结的 28 个基础中文表达 × 两标准场景，共 56 行。
这不是 56 个独立物理场景，也不是新受试者视频。

| 同一留出集 | 基础模型 | 正式 LoRA |
|---|---:|---:|
| 原始 JSON 合法 | 7/56 (12.5%) | 55/56 (98.2%) |
| 完整语义正确 | 4/56 (7.1%) | 52/56 (92.9%) |
| 接近 / 抓住 / 抬起 / 搬运 / 停止 | 全部 0/8 | 6/8、8/8、8/8、8/8、8/8 |
| 正确拒绝越界任务 | 4/16 | 14/16 |
| 越界任务被门控误接受 | 0 | 1 |
| 总部署验收 | 不通过 | 不通过 |

基础模型很多输出格式错误，0 误接受不是语义更安全的证明。LoRA 明显提高了有限测试的语义和格式指标，
但不能据此宣称任意中文指令可靠。测试后没有改权重、提示或门槛。

四条失败：

1. “手先到杯子边上去”：两场景都错解为抬杯，增加了用户没要求的技能。
2. “将杯子放到我手上”：第一场景错解为抬杯，被门控误接受；**未实际执行这个错误计划**。
3. 同一句在第二场景生成未注册的 `handle` 目标，被 Schema 拦截。

部署要求为语义 >=90% 且不支持指令误接受为 0；前者达到，后者未达到。
实际运行 `scripts/116_run_stage6_model.sh ... --execute` 已验证会在创建仿真前拒绝。
不要手动改验收 JSON 的布尔字段绕过限制。

## 物理接口不是语义正确性的替代

语义正确且经门控接受的模型计划去重后，两场景各 4 个动作目标和停止，共 10/10 通过。
其中 8 个实际动作计划只初始化一次，然后通过 `env.step` 执行；2 个停止计划不创建仿真、不施加动作。
非法技能前置、未知非标准场景、过高余量要求三类门控检查也通过。

`framework_physics_passed=true`，但 `model_acceptance_passed=false`。
`107_verify_language_pipeline.py` 返回 1 是预期的总体拒收，不是仿真崩溃。
低层是已验证视频专家参考，**不是 200 次 DAPG 网络**；这 10 项也不是独立新物理场景泛化。

## 可检查的资料

- [首次失败小试](evidence/study-v2/pilot-validation.json)
- [正式训练收据](evidence/study-v3/formal.json)
- [基础模型留出原始结果](evidence/study-v3/base-heldout.json)
- [LoRA 留出原始结果](evidence/study-v3/lora-heldout.json)
- [语言/物理总验收](evidence/study-v3/acceptance.json)
- [去重物理计划报告](evidence/study-v3/physical.json)
- [隔离环境版本](evidence/study-v3/environment.json)
- [实训命令与原生窗口入口](../../STAGE6_STUDY.md)
- [163 项回归测试与冻结文件核验](../post200/evidence/v1/verification.json)

## 答辩建议 4 页

1. 架构：视频 -> 物理参考技能；中文 -> Qwen + LoRA -> JSON -> 可行性门控 -> 执行器。
2. 方法：assistant-only SFT、参数高效更新、前置技能约束；展示 v2 失败与对比指令修正，不隐藏负结果。
3. 结果：同一留出集 4/56 对 52/56；再单列 1 个越界误接受和 10 个物理接口检查。
4. 边界：有限合成中文、两个标准场景、非学习型可行性分数、执行后端仍为专家参考；模型没有通过部署验收。

关键代码：

```python
# src/fromrealhand/language_planner/sft.py
labels = [-100] * len(prefix) + full[len(prefix):]
# 只监督 assistant 的完整 JSON，system/user/padding 不参与损失。

# scripts/111_language_study.py，115 复用同一训练实现
if mode == 'formal' and not read(folder/'pilot-gate.json')['passed']:
    raise ValueError('Pilot failed; formal SFT must not start')
```

本项目组合已有方法，不宣称发明 LoRA 或 SayCan。
采用 [Qwen 官方模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、
[PEFT 官方 LoRA 实现](https://github.com/huggingface/peft/tree/v0.14.0)，
借鉴 [SayCan](https://say-can.github.io/) 的语言建议与低层能力分层，但当前分数只是历史接触余量，不是学习成功概率。

## 下一步

在新的开发数据中补足“只接近/后续动作”和“放置/交给人/当前不支持任务”的边界，重新预注册新留出集。
也可比较更强基础模型；模型选择只用开发数据。不要在已经看过的 56 行上反复修正再宣称独立通过。
阶段 3 的新增接触穿透是另一问题，不能由语言模型训练结果代替修复。
