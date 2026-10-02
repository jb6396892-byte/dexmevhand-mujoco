# 阶段 6 v2/v3 历史实训操作记录

> 最新版本为 `study_v4_guard2`，已通过受控执行验收，见 [最新报告与入口](STAGE6_REFINEMENT.md)。
> 下文保留 v3 的训练与拒收历史；公共脚本 `116` 现在使用最新版本，不再使用 v3。

v3 当时状态：正式 8 epoch 已完成，留出语义 52/56，但越界误接受 1 条，部署拒收。
正确计划的物理接口 10/10 通过不能替代语言安全验收。[完整结果与答辩提纲](presentation/stage6/STUDY_RESULTS.md)。

本轮按用户澄清，`/media/smgbro/shared/lora/` **只保存阶段 6 的语言模型相关内容**。
阶段 3 的 checkpoint、独立评估仍在 `data/processed/dual_video_v14c/`，资料在 `docs/presentation/post200/`。

## 存放位置

```text
/media/smgbro/shared/lora/language/
  runtime/packages/        隔离的现代 Python 包
  runtime/tmp/             安装及下载临时文件
  runtime/huggingface/     Hugging Face 缓存
  model/                   固定 revision 的 Qwen 基础模型
  study_v2/               首次小试及失败记录，禁止覆盖
  study_v3/protocol.json   训练指令增强后的冻结实验
  study_v3/supplement.jsonl
  study_v3/base-validation/
  study_v3/pilot/          3 epoch 小规模 LoRA
  study_v3/pilot-validation/
  study_v3/pilot-gate.json
  study_v3/formal/         通过检查后才进行 8 epoch 正式训练
  study_v3/formal-validation/
  study_v3/base-heldout/
  study_v3/lora-heldout/
  study_v3/physical-acceptance/
  delivery-study_v3/       语言模型索引、资料及数据副本
```

exFAT 不支持 venv 符号链接，因此小型 Python 引导环境位于项目 `data/runtime/stage6-study-venv/`。
数 GB 依赖、模型、训练权重和下载临时文件均在共享盘，原 `dexmv` 不升级。
实际环境为 Python 3.9.18、PyTorch 2.5.1+cu121、Transformers 4.48.3、PEFT 0.14.0。
模型为 Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`，
本地 safetensors 加载，禁止远程自定义代码。CUDA BF16 运算、模板编码和依赖检查均已实测。

## 小试失败与修正

| 开发验证集 | 语义正确 | 原始 JSON 合法 | 不支持指令误接受 |
|---|---:|---:|---:|
| 未微调基础模型 | 9/42 | 11/42 | 0 |
| v2：504 条训练，3 epoch | 34/42 | 42/42 | 1 |
| v3：800 条训练，3 epoch | 42/42 | 42/42 | 0 |

基础模型的 0 误接受不表示理解正确，很多输出被 Schema 拦截。
v2 未通过，不启动正式训练。错误包括复合指令漏掉搬运、停止误拒绝，以及让杯子落地被误解释为搬运。
仅增加 74 组人工对比表达，按两场景和两种包装形成 296 条训练样本；原 504 条不变。
不改学习率、LoRA 模块、门槛或系统提示，不将验证句子复制进训练。
新增句子与验证/留出做程序性精确重合检查；开发阶段没有读取留出预测或评分。
v3 小试 300 步、约 196 秒，42/42 通过，因此开始独立的 8 epoch 正式训练。
验证集用于开发，这个 100% 不是独立泛化结果；独立结果须看正式冻结后的留出报告。

## 顺序与门槛

1. 安装隔离环境，验证 CUDA 与模型模板编码。
2. 固定基础模型 revision，先生成验证集基线结果。
3. 从同一基础模型训练 3 epoch LoRA。只计算 assistant JSON 标签损失。
4. 试训需验证集语义正确率 >=90%、原始结构合法率 >=95%、不支持指令误执行为 0，且正确条数不低于基础模型。
5. 通过后，从同一基础模型另起正式 8 epoch 训练；只用验证 loss 选择 epoch。
6. 冻结后评估预先留出的 56 行中文表达，并将真实模型计划送入独立的 MuJoCo 接口验收。

原始生成错误如实记录，不允许规则回退掩盖模型错误。有限中文测试通过不代表无限开放语言都安全，也不代表新物理场景泛化。

## 可复现命令

```bash
cd /home/smgbro/mujoconew/GITHUB
export STUDY_ROOT=/media/smgbro/shared/lora
# 以下每步只执行一次；目录已经存在时先检查结果，不覆盖重跑。
bash scripts/stage6_study_python.sh scripts/111_language_study.py download
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py prepare
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py baseline
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py pilot
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py formal
bash scripts/stage6_study_python.sh scripts/115_language_contrast_study.py heldout
```

`formal` 会读取 `pilot-gate.json`，不通过就拒绝启动。源码、配置及原数据有哈希验证；修正问题须保留旧结果并登记新版本，不能拿同一个留出集反复调参。

## 物理验收与运行入口

正式训练后的验证门槛仍必须通过，之后才打开原先的 56 条留出指令。
模型生成结果需由原 dexmv 物理进程另外验证，不把小试成功视为物理执行成功：

```bash
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
/home/smgbro/miniconda3/envs/dexmv/bin/python scripts/107_verify_language_pipeline.py \
  --model-evaluation /media/smgbro/shared/lora/language/study_v3/lora-heldout \
  --output /media/smgbro/shared/lora/language/study_v3/physical-acceptance

# 默认只生成并检查计划，不执行物理动作。
bash scripts/116_run_stage6_model.sh "把杯子抓起来" --scene second
# 仅匹配模型哈希的独立语言/物理验收收据通过后，才允许执行和打开窗口。
# 历史 v3 候选未通过；公共 116 现已切到通过验收的 guard2，运行前参见最新报告。
bash scripts/116_run_stage6_model.sh "把杯子搬到目标" --scene second --execute --render
```

v3 历史语言执行日志位于共享盘 `study_v3/instructions/`；当前 `116` 写入 `study_v4_guard2/instructions/`。
低层执行后端仍是经验证的视频专家参考，不是第 200 次 DAPG checkpoint，也不是四个已学成技能网络。
语言实验不修复阶段 3 的接触余量或长训退化。

技术依据：[Qwen 官方模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[PEFT 官方 v0.14.0](https://github.com/huggingface/peft/tree/v0.14.0)、[Transformers Trainer](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/training_args.py)。
