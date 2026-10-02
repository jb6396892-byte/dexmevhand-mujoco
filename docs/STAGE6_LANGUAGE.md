# 阶段 6：语言模型高层规划器（D2）

2026-10-02 最新更新：**阶段 6 已通过有限指令范围、两标准场景的受控验收。**
新增对比 LoRA 和独立语义门控，最终新 100 条测试原始正确 94/100，系统错误计划放行 0，合法任务放行 50/50。
详见 [最新结果与命令](STAGE6_REFINEMENT.md)。旧 v3 的 52/56 和 1 条越界失败仍保留为历史结果。
C3 独立低层技能网络仍未完成，不因语言模型训练结束而自动完成。

存储更正：最初沙箱内看到 shared 为只读，宿主机实际为 `exfat rw`，不是硬盘故障；未修盘或重挂载。
新增语言环境、模型与训练结果已写入 `/media/smgbro/shared/lora/language/`。阶段 3 仍在项目原目录。

## 现在有什么

| 子步骤 | 状态 | 产物 |
|---|---|---|
| 规则与日志生成标签 | 已完成 | v4 共 1160 条训练；最终 guard2 新测试 100 条，旧输出回归 240 条 |
| 技能计划 JSON Schema | 已验证 | `configs/skill_plan.schema.json`，技能和前置顺序均受约束 |
| 小模型 LoRA/SFT | v4 对比训练 4 epoch 完成 | `lora/language/study_v4_guard2/candidate/adapter/`，与 v4 权重一致 |
| 输出安全边界 | 已验证 | 未知技能、顺序错误、额外字段、重复字段、场景篡改均拒绝 |
| 低层可行性过滤 | 已验证，非学习概率 | 标准场景物理余量分数与实时技能契约 |
| 指令与计划一致性 | 有限词汇契约，只能否决 | 交接、放置、未知后缀或目标错配拒绝 |
| 接口物理验收 | 8 动作计划 + 2 停止计划通过 | 最终新测试的实际模型计划去重后执行 |
| 模型与系统语言验收 | 原始正确 94%，系统受控通过 | 模型 6 条错误由门控阻止，合法任务放行 50/50 |

全量 175 项测试通过。真实 Qwen tokenizer、PEFT/CUDA 及训练均已实测，
LoRA 更新参数 2,162,688 个，训练显存分配峰值约 2.52 GB。旧 `dexmv` 未升级。

## 架构与边界

```text
中文指令 + 外部指定 scene
    -> Qwen2.5-0.5B-Instruct + LoRA（默认只生成，显式执行受门控约束）
    -> 原始 JSON，不自动修复、不暗中回退规则
    -> JSON Schema + 完整前置技能顺序
    -> 原指令契约与生成目标一致性，未知表达保守拒绝
    -> 标准场景可行性余量过滤
    -> 阶段 5 执行器：实时前置条件、安全、超时、有限重试
    -> 已验证专家参考动作 -> env.step(action) -> 自由杯子动力学
```

语言模型不输出手指关节角，不写入物体位姿。当前执行后端不是 200 次训练所得 DAPG 网络，也不是四个独立技能网络。要替换后端，须另外完成 C3 及连续技能衔接验收。

模型响应示例：

```json
{"decision":"execute","plan":{"schema_version":1,"scene":"first","object":"mug","goal":"lift","skills":["reach","grasp","lift"]}}
```

未知任务响应为 `{"decision":"reject"}`。明确停止用 `goal=stop, skills=[]`，拒绝与停止不混淆。抓起只到 lift，不能擅自增加 transport；倒水、放手、未知物体、任意目标坐标尚不支持。

**结构合法不代表理解正确。** 例如把“拿起”错解为“搬运”，JSON 可以合法且物理可执行，但语言评估仍判错。否定指令被错误接受单列为严重失败。Schema 无法保证对无限种自然语言都理解正确。

## 原始数据来源与划分

本地目录：`data/processed/stage6_language_v1/dataset/`。

- `train.jsonl`：84 个基础句子，乘两个场景与三种礼貌包装，共 504 行。
- `validation.jsonl`：21 个基础句子、42 行；用于开发和选择 checkpoint，不算独立测试。
- `heldout.jsonl`：28 个基础句子、56 行；v3 当时的冻结测试，本轮已知结果后仅用于回归。
- `protocol.json`：数据、代码、配置与来源 SHA256。
- `execution_audit.json`：阶段 5 成功与失败日志索引，故障注入不伪装成新的语言样本。
- `feasibility.json`：阶段 4 两标准场景技能入口实测结果。

中文表达是人工编写的合成改写，不是从真实用户采集的指令。计划标签由注册规则规划器生成，并与两条完整搬运日志中的成功技能核对；不是从视频直接得到语言标签。训练副本不得当作 504 个独立真实任务。划分按基础句子分组，不是严格的句式结构 OOD 测试；仍只有两个原视频来源。

已生成目录禁止覆盖；修改冻结代码或样本后应创建新版本目录和协议，不能悄悄更新旧哈希。

## 可行性分数

对标准场景中的每个技能，使用已有物理回放记录：

```text
skill_score = max(0, 1 - max_scene_penetration / 0.001)
plan_score = min(skill_score for all required skills)
accept = nominal_scene AND plan_score >= 0.05
```

前提是该技能本身通过原物理验收；没有证据或新工况则拒绝。`0.05` 对应至少 0.05 mm 的历史穿透余量。第一视频完整搬运分数约 0.423，第二视频约 0.099；把要求提高为 0.2 时第二视频计划会被过滤。运行时依然检查原 1 mm 穿透门槛与其他物理条件，不能靠历史分数绕过检查。

这是归一化的**物理余量分数**，不是经过概率校准的成功率，不是已经学好的 affordance/value 网络，也不能推断毫米级新初态可执行。当前只有每场景每技能一个标准入口证据。

## 当前操作入口

环境已安装，不需重复安装。最新训练与门控流程见 [修正记录](STAGE6_REFINEMENT.md)，
不要再用历史 `103` 默认路径新建另一套环境。所有新增模型和日志均在共享盘 lora；Linux 仅留 65 MB 引导环境。

```bash
cd /home/smgbro/mujoconew/GITHUB
# 只生成计划，不执行机械动作。
bash scripts/116_run_stage6_model.sh "把杯子抓起来" --scene second
# 在原生 MuJoCo 窗口中显式执行，关闭窗口退出。
bash scripts/116_run_stage6_model.sh "把杯子搬到目标位置" --scene second --execute --render
```

当前 `study_v4_guard2` 的匹配收据为 `model_acceptance_passed=true`，可以显式受控执行。
旧 v3/v4 的拒收结论不变。交接给人、倒水、放置、任意目标坐标仍不支持；缺少共享盘或冻结产物不匹配时拒绝启动。
100 条语言测试不是 100 个独立物理场景；底层仍只有两标准场景的专家参考。

## 方法依据

- [Qwen2.5-0.5B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)：选择可做有限中文指令实验的小模型，不声称它是当前最优机器人规划模型。
- [LoRA 原作者源码](https://github.com/microsoft/LoRA)、[Hugging Face PEFT](https://github.com/huggingface/peft)：采用现成参数高效微调实现，不自行实现模型训练引擎。
- [SayCan 官方项目与论文](https://say-can.github.io/)：借鉴语言建议必须受低层能力约束的思想；此处仅为标准场景历史余量门控，不是其学习价值函数或完整复现。

最新结果见 [修正答辩材料](presentation/stage6/REFINEMENT_RESULTS.md)，旧 v3 见 [历史训练材料](presentation/stage6/STUDY_RESULTS.md)。
原框架结果与一张截图见 [历史材料](presentation/stage6/README.md)，原过程见 [框架运行记录](run_logs/2026-10-02-stage6-framework.md)。
