# 阶段 6：语言模型高层规划器（D2）

2026-10-02：按用户最新编号推进 D2。**本轮只完成框架、数据与接口验收，不安装新环境、不下载模型、不训练 LoRA。** 用户在执行中明确选择“先只搭框架”。C3 独立低层技能网络仍未完成，不因阶段编号变化自动完成。

存储更正：最初沙箱内看到 shared 为只读，沙箱外复核实际为 `exfat rw`，不是硬盘故障；未做修盘、重挂载或任何共享盘写入。后续需在正常终端或获准的权限环境操作，不要据此格式化、修复磁盘。

## 现在有什么

| 子步骤 | 状态 | 产物 |
|---|---|---|
| 规则与日志生成标签 | 已完成 | 504 条训练、42 条验证、56 条留出样本 |
| 技能计划 JSON Schema | 已验证 | `configs/skill_plan.schema.json`，技能和前置顺序均受约束 |
| 小模型 LoRA/SFT | 代码就绪，未实跑 | 独立环境脚本、冻结配置、仅 assistant 标签计损失 |
| 输出安全边界 | 已验证 | 未知技能、顺序错误、额外字段、重复字段、场景篡改均拒绝 |
| 低层可行性过滤 | 已验证，非学习概率 | 标准场景物理余量分数与实时技能契约 |
| 接口物理验收 | 8 动作计划 + 2 停止计划通过 | 使用标准答案测试，不是语言模型预测 |
| 模型语言与物理验收 | 未执行 | 模型权重、实测语言正确率和泛化结果均尚无 |

全量 157 项测试通过，其中新增 15 项。训练标签编码使用假 tokenizer 验证掩码和 padding；真实 Qwen tokenizer、PEFT/CUDA API 及显存占用尚待独立环境安装后检查，不能说训练环境已经跑通。

## 架构与边界

```text
中文指令 + 外部指定 scene
    -> Qwen2.5-0.5B-Instruct + LoRA（待训练）
    -> 原始 JSON，不自动修复、不暗中回退规则
    -> JSON Schema + 完整前置技能顺序
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

## 数据来源与划分

本地目录：`data/processed/stage6_language_v1/dataset/`。

- `train.jsonl`：84 个基础句子，乘两个场景与三种礼貌包装，共 504 行。
- `validation.jsonl`：21 个基础句子、42 行；只用于选择 checkpoint。
- `heldout.jsonl`：28 个基础句子、56 行；冻结候选后一次性评估，不用于选模型或调阈值。
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

## 后续执行顺序

下面是恢复工作后的命令，**本轮没有运行安装、下载或训练命令**。从项目目录执行。所有大包、模型、临时下载和 LoRA 存到 shared；Linux 仅放很小的 Python 引导环境，避免 exFAT 不支持 venv 符号链接的问题。不要安装到 `dexmv`。

### 1. 检查和安装隔离环境

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/103_setup_language_env.sh --check-only
# 重新确认开始后才执行：
bash scripts/103_setup_language_env.sh --install
```

安装目的地 `/media/smgbro/shared/fromrealhand-stage6/`；Linux 引导目录 `data/runtime/stage6-venv/`。安装失败时保留现场，不重复覆盖，也不要删除现有 dexmv。默认预留 shared 10 GiB，使用固定依赖版本；实际下载、依赖兼容性和 exFAT 包运行仍待验证。

### 2. 核验数据并下载固定版本模型

```bash
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/104_train_language_lora.py check
bash scripts/stage6_python.sh scripts/104_train_language_lora.py download
```

下载时解析并记录 Hugging Face revision 和文件哈希；只使用 safetensors，`trust_remote_code=False`。下载与训练分开，不会下载后自动开始。

### 3. LoRA 小试与验证集检查

```bash
bash scripts/stage6_python.sh scripts/104_train_language_lora.py train
bash scripts/stage6_python.sh scripts/105_evaluate_language.py \
  --model /media/smgbro/shared/fromrealhand-stage6/model \
  --adapter /media/smgbro/shared/fromrealhand-stage6/lora-v1/adapter \
  --split validation --output data/processed/stage6_language_v1/lora-validation
```

预设 rank=16、alpha=32、q/k/v/o 投影、8 epochs、batch=2、梯度累积=4、学习率 2e-4。冻结基础模型，只更新 LoRA；mask 掉 system/user 和 padding，禁止截掉过长 JSON 标签。按验证 loss 选择最佳 epoch。没有成功完成的训练收据时，不应进入部署。

### 4. 冻结后比较基础模型与 LoRA

```bash
bash scripts/stage6_python.sh scripts/105_evaluate_language.py \
  --model /media/smgbro/shared/fromrealhand-stage6/model \
  --split heldout --output data/processed/stage6_language_v1/base-heldout
bash scripts/stage6_python.sh scripts/105_evaluate_language.py \
  --model /media/smgbro/shared/fromrealhand-stage6/model \
  --adapter /media/smgbro/shared/fromrealhand-stage6/lora-v1/adapter \
  --split heldout --output data/processed/stage6_language_v1/lora-heldout
```

分别记录原始 JSON 合法率、完整语义正确率、各目标正确率、同义表达一致性、拒绝/误执行数。过滤后的合法率不能替代模型原始输出质量。已看过的留出集不可继续调参后再称新留出。

### 5. 实际模型计划接 MuJoCo

加载原有 MuJoCo 环境变量后，用 legacy Python：

```bash
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/107_verify_language_pipeline.py \
  --model-evaluation data/processed/stage6_language_v1/lora-heldout \
  --output data/processed/stage6_language_v1/lora-acceptance --capture
```

同一场景同一计划只实际执行一次，不把 56 个同义句重复计为 56 个独立物理成功。最终需两场景四种动作目标与停止共 10 个计划全通过，并满足留出语义正确率至少 90%、不支持指令误执行数为 0。

合格后才能用 `106_run_language_plan.py --execute --acceptance-report .../summary.json` 执行学习规划；不带 `--execute` 默认只生成/检查，`--render` 可打开原生 MuJoCo 窗口。当前框架验收报告的 `model_acceptance_passed=false`，不能冒充此部署收据。

## 方法依据

- [Qwen2.5-0.5B-Instruct 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)：选择可做有限中文指令实验的小模型，不声称它是当前最优机器人规划模型。
- [LoRA 原作者源码](https://github.com/microsoft/LoRA)、[Hugging Face PEFT](https://github.com/huggingface/peft)：采用现成参数高效微调实现，不自行实现模型训练引擎。
- [SayCan 官方项目与论文](https://say-can.github.io/)：借鉴语言建议必须受低层能力约束的思想；此处仅为标准场景历史余量门控，不是其学习价值函数或完整复现。

结果与一张截图见 [答辩材料](presentation/stage6/README.md)，本轮过程见 [运行记录](run_logs/2026-10-02-stage6-framework.md)。
