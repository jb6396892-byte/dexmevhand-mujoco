# 从真实人手视频训练 MuJoCo 灵巧手

这个项目用于把真实的人手抓杯视频转换成 DexMV 可用的 demonstration，
再通过行为克隆和 DAPG 强化学习训练 MuJoCo Adroit 灵巧手策略。

项目后续会扩展为层次化模仿学习系统：高层控制器理解“抓杯子”“倒水”
等自然语言任务并生成技能计划，低层控制器负责执行精确的关节动作。

## 当前状态

已经完成：

- `relocate-mug` 完整训练和策略可视化。
- MuJoCo、DexMV、dexmv-learn 和 DAPG 环境检查。
- 视频抽帧、目录准备、手部 retarget、轨迹可视化。
- demonstration 生成、格式检查、DAPG 训练和策略回放入口。
- 层次化模仿学习的架构与实施路线设计。
- DexYCB Subject 01 已在共享盘完成解压和校验，并筛出两条右手 `025_mug` 轨迹。
- 第一条真实轨迹已完成坐标转换、重投影、retarget、MuJoCo 离屏回放和 demonstration 验证。
- 已用 MANO 官方模型恢复手部旋转，完成标签对比和 20 次 DAPG 短训练。

下一步：

- 扩充真实抓杯轨迹，分析当前策略回放没有接触杯子的原因。
- 将轨迹拆成 `reach`、`grasp`、`lift` 和 `transport` 技能。

详细文档：

- [网页执行看板](docs/index.html)
- [层次化模仿学习架构](docs/ARCHITECTURE.md)
- [分阶段实施路线](docs/ROADMAP.md)
- [实际操作执行计划](docs/EXECUTION_PLAN.md)
- [数据与标注格式](docs/DATA_FORMAT.md)
- [实验记录模板](docs/WORK_LOG_TEMPLATE.md)
- [MANO 与 20 次短训练记录](docs/run_logs/2026-09-24-mano-smoke20.md)

网页版看板可以直接打开 `docs/index.html`。需要发布到 GitHub Pages 时，在仓库
`Settings -> Pages` 中选择从 `main` 分支的 `/docs` 目录部署。

## 整体流程

```text
真实 RGB/RGB-D 视频
  -> 视频抽帧
  -> 手部 3D 姿态估计
  -> 杯子 6D 位姿估计
  -> 相机坐标转换到世界坐标
  -> 人手到 Adroit 灵巧手 retarget
  -> 生成 DexMV demonstration pickle
  -> 行为克隆 + DAPG/TRPO 训练
  -> 策略可视化与评估
```

DexMV 不负责从原始视频估计手部和物体位姿。本项目目前消费外部姿态估计
结果，并负责后续的坐标转换、retarget、demonstration 生成和策略训练。

## 层次化扩展

```text
自然语言指令 + 场景状态
  -> 高层任务规划器
  -> 经过校验的技能计划
  -> 技能执行器和可行性检查
  -> MuJoCo 低层策略
  -> 状态反馈和重新规划
```

高层模型只输出结构化技能，不直接输出 30 维关节动作。低层首先包含：

```text
reach(mug)
grasp(mug)
lift(mug, height)
transport(mug, target_pose)
```

倒水阶段再增加：

```text
tilt(mug, angle)
upright(mug)
place(mug, target_pose)
release(mug)
```

## 复用的本机项目

本项目不复制 DexMV 的大型资源，而是复用：

- `/home/smgbro/dexmv-sim`
- `/home/smgbro/dexmv-learn`
- conda 环境 `dexmv`

当前项目目录：

```text
/home/smgbro/mujoconew/GITHUB
```

## 单条真实轨迹目录

```text
data/real_data/relocate_mug/seq_000/
  rgb/
  depth/
  calib/
    camera_matrix.npy
    dist_coeffs.npy
    camera_to_world.npy
  hand_pose/
    results_global_000.npy
    joints_000.npy
  object_pose/
    000.npy
  annotations/
    skill_segments.json
  meta.json
  retargeting.pkl
```

生成 demonstration 最少需要：

- `hand_pose/results_global_*.npy`
- `hand_pose/joints_*.npy`
- `object_pose/*.npy`
- 如果物体仍在相机坐标中，还需要 `calib/camera_to_world.npy`

物体位姿可以是直接保存的 `4x4` 矩阵，也可以是包含杯子位姿的 NumPy 字典。

## 环境检查

默认本机路径保存在 `.env.example`。需要覆盖时可以创建本地 `.env`，
该文件不会提交到 Git。

关键 MuJoCo 环境变量：

```bash
LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

运行检查：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/00_check_env.sh
```

## DexYCB 与 MANO 资源

大型数据和受许可约束的 MANO 模型统一保存在 `shared` 固态硬盘：

```text
/media/smgbro/shared/DexYCB/
  archives/      # DexYCB tar.gz 和用户从 MANO 官网下载的 mano_v1_2.zip
  dataset/       # calibration、models 和后续 subject 数据
  mano/models/   # MANO_RIGHT.pkl、MANO_LEFT.pkl
  checksums/     # 下载文件 SHA-256
```

公共 DexYCB 资源下载完成后运行：

```bash
bash scripts/12_prepare_external_assets.sh --extract
```

MANO 模型不能匿名下载。登录 `https://mano.is.tue.mpg.de/`、接受研究许可并下载
`mano_v1_2.zip`，将原始压缩包放入上述 `archives/`，然后再次运行同一命令。
脚本会校验压缩包、提取左右手模型并放到 `.env.example` 配置的 `MANO_ROOT`。

验证 MANO 左右手模型可以完成前向计算：

```bash
set -a && source .env && set +a
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/13_validate_mano.py
```

## 基本使用方法

创建一条轨迹的目录模板：

```bash
python scripts/02_prepare_pose_dirs.py --seq seq_000
```

从视频抽帧：

```bash
python scripts/01_extract_frames.py \
  --video data/raw_videos/seq_000.mp4 \
  --output data/real_data/relocate_mug/seq_000/rgb
```

外部程序生成手部和物体位姿后，运行 retarget：

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/03_retarget_one.py \
  --hand-dir data/real_data/relocate_mug/seq_000/hand_pose \
  --output data/real_data/relocate_mug/seq_000/retargeting.pkl
```

扫描 DexYCB 中的右手 `025_mug` 抓取序列：

```bash
python scripts/09_scan_dexycb.py \
  --root data/external/dexycb \
  --object 025_mug \
  --hand-side right \
  --output data/processed/dexycb_mug_sequences.json
```

转换选定序列和相机视角：

```bash
python scripts/10_convert_dexycb.py \
  --root data/external/dexycb \
  --sequence SUBJECT/SEQUENCE \
  --camera CAMERA_SERIAL \
  --output data/real_data/relocate_mug/seq_dexycb_001
```

生成手部重投影、物体坐标轴和轨迹报告：

```bash
MPLCONFIGDIR=/tmp/fromrealhand-matplotlib \
python scripts/11_visualize_source_pose.py \
  --sequence-dir data/real_data/relocate_mug/seq_dexycb_001 \
  --output data/processed/seq_dexycb_001
```

可视化 retarget 后的手和杯子：

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/04_visualize_retargeting.py \
  --retargeting data/real_data/relocate_mug/seq_000/retargeting.pkl \
  --object-dir data/real_data/relocate_mug/seq_000/object_pose \
  --camera-to-world data/real_data/relocate_mug/seq_000/calib/camera_to_world.npy
```

生成 DexMV demonstration：

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/05_generate_demo.py \
  --sequence-dir data/real_data/relocate_mug/seq_000 \
  --output data/demonstrations/relocate-mug-real.pkl \
  --trajectory-id seq_000
```

检查 demonstration：

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/06_validate_demo.py \
  data/demonstrations/relocate-mug-real.pkl
```

训练和可视化：

```bash
bash scripts/07_train_dapg.sh
bash scripts/08_visualize_policy.sh /path/to/best_policy.pickle
```

MANO 版本的短训练使用已验收的 `hand_pose_mano/` 和
`data/demonstrations/relocate-mug-mano-real.pkl`。本机无可用 CUDA，运行：

```bash
TRAIN_ENTRY="$PWD/scripts/15_train_dapg_cpu.py" bash scripts/07_train_dapg.sh \
  "$PWD/configs/dapg-mug-mano-smoke.yaml"
```

单轨迹短训练的策略回放没有接触杯子；结果见
[MANO 与短训练记录](docs/run_logs/2026-09-24-mano-smoke20.md)。

## 仍然缺少的内容

- 更多受试者和视角的真实抓杯轨迹，用于避免单轨迹过拟合。
- 技能切分、技能成功条件和统一技能执行器。
- 高层自然语言到技能计划的模型。
- 倒水任务的真实视频、MuJoCo 环境和低层技能。

## 注意事项

- 第一阶段只训练状态输入策略，暂不训练端到端图像策略。
- 第一批实验保持 `object_scale=0.8`，与已经跑通的环境一致。
- 不要在调用 `YCBRelocate.step()` 前再次缩放 action，环境内部已经处理。
- 坐标转换和投影验证必须在 retarget 之前完成。
- 原始视频、数据集、标定结果、策略和 checkpoint 不上传 GitHub。
