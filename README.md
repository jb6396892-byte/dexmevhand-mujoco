> 最新进展（v9，2026-10-01）：第二视频通过当前场景单条示范最低准入。联合优化动作与自由杯子动力学后，无名指末秒承力接触 100%，四指门槛、全程物理重放及半步长均通过；目标误差 5.25 / 7.86 mm，最大穿透 0.954 / 0.796 mm。已导出独立示范，未启动新训练或泛化测试。
> 详见 [v9 动力学优化与训练前验收](docs/run_logs/2026-10-01-dynamic-contact-v9.md)。窗口：`bash scripts/65_view_dynamic_contact_gpu.sh`。这是优化专家动作，不是新训练策略；无名指在抬杯后的过渡阶段加入，尚不等于逐帧忠实还原全部接触时序。
>
> 历史 v8：修正了扩大碰撞 margin 导致的静态距离误判，新增原参数下的接触连续规划、动作余量约束和动态跟踪检查。第二视频本轮 21 次当前场景回放，12 次物理门槛通过，0 次四指抓法通过；当时无名指持续接触仍未解决。旧稳定三指版本保留，未启动训练或新的泛化测试。
> 详见 [v8 连续接触与动作可执行性](docs/run_logs/2026-10-01-contact-continuation-v8.md)。几何可接触不等于动作可执行，失败候选未加入训练。当前无法读取有效实时额度，不能报告剩余百分比。
>
> 历史 v7：冻结的 v6 残差已完成十个留出合成场景，物理、数值抓法及 20 mm 到位门槛通过 10/10，固定动作基线 6/10；平均目标误差 5.51 mm 对 10.61 mm。优势主要在目标变化，旋转场景未优于基线。
> 第二视频完成无名指局部搜索、接触目标修正及全手可达性诊断，持续四指接触仍未通过。保留原稳定版本，没有启动新训练。
> 详见 [v7 新场景测试与无名指诊断](docs/run_logs/2026-09-29-heldout-and-ring-v7.md)。
>
> 历史 v6：增加抓稳后的有界目标反馈，第一视频专家终点误差约 0.85 mm，第二视频由 64 mm 降至约 1.6 mm；独立初态与半步长检查通过。
> 第一视频 16 条精确示范已通过验收并重新训练残差；新场景对照尚待完成。第二视频仍受四指接触忠实度限制，未加入训练。
> 详见 [v6 搬运闭环记录](docs/run_logs/2026-09-28-transport-feedback-v6.md)。窗口：`bash scripts/57_view_transport_gpu.sh first` 或 `second`。
>
> 历史 v5：第二视频已修复滑落，冻结候选通过 6 个初态和半步长物理检查；严格忠实度仍未准入训练。
> 16 条示范训练的残差在 10 个新合成场景中完整通过 10/10，平均目标误差 17.64 mm（固定动作 18.66 mm），最差 24.72 mm（固定动作 36.32 mm）。优势有限，仍只有一个独立训练视频。
> 详见 [v5 对握修复与配对验证](docs/run_logs/2026-09-28-opposition-and-residual-v5.md)。第二视频物理窗口：`bash scripts/54_view_second_video_gpu.sh`，这是专家控制动作重放，不是新训练策略。
>
> 历史 v4：已定位旧策略首次偏离，完成 300 轮 BC 与专家纠偏对照，新增固定参考动作＋学习残差版本。
> 扩展残差策略在 10 个留出变化场景中物理通过 10/10、物理及忠实度同时通过 9/10；尚未整体优于固定动作基线。没有启动 2000 次训练。
> 详见 [v4 策略学习与数据扩展记录](docs/run_logs/2026-09-28-policy-learning-v4.md)。窗口：`bash scripts/49_view_residual_policy_gpu.sh`，这是保存的策略动作物理重放，不是在线网络推理。
>
> 历史 v3：完整视频示范通过物理、视频忠实度、独立种子与半步长检查，6 条示范已导出，20 次 DAPG 短训已完成。
> 神经网络策略独立回放中 2/11 可抬杯保持，0/11 通过完整搬运验收。详见 [v3 优化与短训记录](docs/run_logs/2026-09-27-scene-fidelity-v3.md)。示范窗口：`bash scripts/43_view_video_faithful_v3_gpu.sh`。
> 下列 v1/v2 记录保留作历史对照，不代表最新准入状态。

> 近表面接触修正：见 [优化方法与验收标准](docs/SURFACE_CONTACT_OPTIMIZATION.md)。使用 `bash scripts/36_view_surface_grasp_gpu.sh` 查看。
> 新候选通过 6 个窄初态及半步长物理检查；旧版存在提前接触，严格视频忠实度仍未通过，未启动训练。

> 视频抓法实验版本：见 [版本说明](docs/VIDEO_FAITHFUL_VERSION.md) 和 [验证记录](docs/run_logs/2026-09-27-video-faithful.md)。
> 旧版保留在 `main` / `physical-grasp-verified-v1`；本分支物理抓杯通过，严格视频忠实度尚未通过，未启动新训练。


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
- 旧短训策略虽可运行，但物理回放未抬起杯子；已记录失败原因。
- 基于首条视频生成了可自由运动、可独立重放的物理抓杯示范；11 个窄初态验证通过。
- v3 完整视频示范通过现有准入门槛；6 条示范的训练采样重放误差为零，20 次 DAPG 短训完成。
- v3 原始神经网络策略 2/11 可抬杯保持，0/11 通过完整验收；增加 BC 到 300 轮仍未解决闭环失败。
- v5 使用 16 条合格示范（仍来自一段独立真实视频）训练残差，并完成固定动作配对测试。第二段视频稳定物理抓杯已通过，严格忠实度仍不通过，未混入训练。

下一步：

- 第二视频 v9 已通过当前场景四指物理示范最低准入并停止在训练前。先向用户汇报，再决定示范扩展、学习方法及新场景验证；不自动启动长训练或泛化。
- 扩充多条独立真实抓杯轨迹，区分视频来源与毫米级初态扰动副本。
- 将轨迹拆成 `reach`、`grasp`、`lift` 和 `transport` 技能。

详细文档：

- [网页执行看板](docs/index.html)
- [层次化模仿学习架构](docs/ARCHITECTURE.md)
- [从视频到可执行策略的实施设计](docs/VIDEO_IMITATION_PLAN.md)
- [分阶段实施路线](docs/ROADMAP.md)
- [实际操作执行计划](docs/EXECUTION_PLAN.md)
- [数据与标注格式](docs/DATA_FORMAT.md)
- [实验记录模板](docs/WORK_LOG_TEMPLATE.md)
- [MANO 与 20 次短训练记录](docs/run_logs/2026-09-24-mano-smoke20.md)
- [几何与物理失败诊断](docs/run_logs/2026-09-26-stage3-geometry-physics.md)
- [可执行示范及训练准入](docs/run_logs/2026-09-26-physical-grasp-training-ready.md)

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
  -> 连续轨迹和自由杯子物理验收
  -> 生成真实物理动作的 DexMV demonstration
  -> 行为克隆 + DAPG 训练
  -> 独立初态和未见过序列的策略物理评估
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

## 物理示范与短训练

在已准备共享盘数据的本机终端中，直接查看**专家保存动作**的 MuJoCo 物理回放：

```bash
bash scripts/28_view_verified_grasp_gpu.sh
```

这不是训练策略，也不是逐帧强制设置杯子位姿。进行新示范的课程/GPU 预检和 20 次 DAPG 短训：

```bash
bash scripts/27_train_verified_smoke.sh
bash scripts/27_train_verified_smoke.sh --train
```

短训完成后仍需对新策略单独运行物理抓取评估。仓库不包含 DexYCB 数据、MANO 模型、示范 pkl 或 checkpoint；新机器克隆后需按许可自行配置这些资源。

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

可视化 retarget 后的手和杯子。窗口播放可加 `--loop` 持续循环（Ctrl+C 结束）、`--fps 10` 放慢；单次播放默认在末帧停留 5 秒。若窗口报 `GLEW initialization error`，使用离屏 MP4：

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/04_visualize_retargeting.py \
  --retargeting data/real_data/relocate_mug/seq_dexycb_001/retargeting_mano_aligned.pkl \
  --object-dir data/real_data/relocate_mug/seq_dexycb_001/object_pose \
  --camera-to-world data/real_data/relocate_mug/seq_dexycb_001/calib/camera_to_mujoco.npy \
  --skip-frame 20 \
  --output-video data/processed/seq_dexycb_001/aligned_retargeting.mp4
```

这只是按记录位姿播放，不是动作驱动的动力学抓取验证。

本机 MuJoCo 交互窗口使用项目外侧的隔离 GPU 扩展 `../.local/mujoco-py-gpu`；原 `dexmv` 环境的 CPU 扩展未修改。以下窗口只回放记录的手杯位姿，**不是训练模型**：

```bash
bash scripts/19_visualize_aligned_gpu.sh --loop --fps 10
```

按 Ctrl+C 结束；不加 `--loop` 时只播放一次并在末帧停留 5 秒。启动脚本已设置 NVIDIA PRIME、GLEW/GL 预加载和旧版 MuJoCo 所需库路径。


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

要看已经训练的 GPU 短训策略在真实 MuJoCo 动力学里执行动作，运行：

```bash
bash scripts/20_visualize_trained_policy_gpu.sh
```

该脚本使用 `best_policy.pickle` 的确定性动作，正常调用 `env.step(action)`；不强制移动手或杯。也可以把其他策略文件路径作为第一个参数。当前 20 次迭代的模型仍未学会抬杯，见[物理回放记录](docs/run_logs/2026-09-25-policy-physics.md)。


MANO 版本的短训练使用已验收的 `hand_pose_mano/` 和
`data/demonstrations/relocate-mug-mano-real.pkl`。此前 GPU 驱动不可用时，使用 CPU 入口运行：

```bash
TRAIN_ENTRY="$PWD/scripts/15_train_dapg_cpu.py" bash scripts/07_train_dapg.sh \
  "$PWD/configs/dapg-mug-mano-smoke.yaml"
```

早期单轨迹短训练在离屏检查环境中的策略回放没有接触杯子；结果见
[MANO 与短训练记录](docs/run_logs/2026-09-24-mano-smoke20.md)。

后续手杯坐标、观测时序和动作动力学验收见
[对齐版 MANO 示范记录](docs/run_logs/2026-09-24-aligned-demo.md)。对齐版示范尚未通过动力学抓取验收，不用于完整 DAPG 训练。

### GPU 环境

本机 RTX 4060 的 `dexmv` 环境已有 `torch 1.13.1+cu117`；不需要单独安装
CUDA Toolkit，也不要直接升级这个旧环境中的 PyTorch/NumPy。若 `nvidia-smi`
无法连接驱动，先检查 `uname -r` 和 `modinfo nvidia`。2026-09-24 的检查显示
运行内核 `6.17.0-35-generic` 没有匹配的 NVIDIA 模块，已安装的 590 模块仅适用于
`6.17.0-14-generic`。系统预装了 `7.0.0-30-generic`，但也没有对应模块。

使用管理员权限安装 Ubuntu 仓库提供的匹配驱动和 HWE 内核模块，之后重启：

```bash
sudo ubuntu-drivers install
sudo reboot
```

重启后在本项目目录验证（第一条应显示 NVIDIA GPU，第二条应打印 CUDA 训练成功）：

```bash
nvidia-smi
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/16_check_gpu.py
```

2026-09-24 验证结果：启动内核为 `7.0.0-34-generic`，NVIDIA 驱动为
`595.91.07`，RTX 4060 上的 PyTorch 前向、反向传播和优化器更新均通过。
本机有 Ubuntu 22.04/24.04 双系统；若安装驱动后重启仍进入旧内核，检查
`/boot/efi/EFI/ubuntu/grub.cfg` 是否指向当前 Ubuntu 24.04 根分区。
本次通过重新安装当前系统的 UEFI GRUB 修复了引导指向问题。

GPU 验证通过后，使用默认训练入口（不设置 `TRAIN_ENTRY`）；默认入口会将
DAPG 的神经网络 baseline 放在 GPU 上：

```bash
bash scripts/07_train_dapg.sh "$PWD/configs/dapg-mug-mano-smoke.yaml"
```

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
