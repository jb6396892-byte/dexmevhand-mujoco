# 阶段 6 实训操作记录

本轮按用户澄清，`/media/smgbro/shared/lora/` **只保存阶段 6 的语言模型相关内容**。
阶段 3 的 checkpoint、独立评估仍在 `data/processed/dual_video_v14c/`，资料在 `docs/presentation/post200/`。

## 存放位置

```text
/media/smgbro/shared/lora/language/
  runtime/packages/        隔离的现代 Python 包
  runtime/tmp/             安装及下载临时文件
  runtime/huggingface/     Hugging Face 缓存
  model/                   固定 revision 的 Qwen 基础模型
  study_v2/protocol.json   预注册语言实验
  study_v2/base-validation/
  study_v2/pilot/          3 epoch 小规模 LoRA
  study_v2/pilot-validation/
  study_v2/pilot-gate.json
  study_v2/formal/         通过检查后才进行 8 epoch 正式训练
  study_v2/formal-validation/
  study_v2/base-heldout/
  study_v2/lora-heldout/
```

exFAT 不支持 venv 符号链接，因此小型 Python 引导环境位于项目 `data/runtime/stage6-study-venv/`。
数 GB 依赖、模型、训练权重和下载临时文件均在共享盘，原 `dexmv` 不升级。

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
bash scripts/stage6_study_python.sh scripts/111_language_study.py baseline
bash scripts/stage6_study_python.sh scripts/111_language_study.py pilot
bash scripts/stage6_study_python.sh scripts/111_language_study.py formal
bash scripts/stage6_study_python.sh scripts/111_language_study.py heldout
```

`formal` 会读取 `pilot-gate.json`，不通过就拒绝启动。源码、配置及原数据有哈希验证；修正问题须保留旧结果并登记新版本，不能拿同一个留出集反复调参。

技术依据：[Qwen 官方模型](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)、[PEFT 官方 v0.14.0](https://github.com/huggingface/peft/tree/v0.14.0)、[Transformers Trainer](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/training_args.py)。
