> 最新 v3 完整视频示范已通过现有物理与视频忠实度门槛，并完成 20 次短训；模型仍未通过完整搬运验收。见 [v3 实验记录](run_logs/2026-09-27-scene-fidelity-v3.md)，窗口入口为 `scripts/43_view_video_faithful_v3_gpu.sh`。下文是旧版历史记录。
> 后续审计发现本页旧候选存在约 2.40–2.83 mm 的承力间距。下文“物理通过”仅指旧门槛，不等于近表面接触合格。
> 请优先查看 [近表面接触优化与新验收](SURFACE_CONTACT_OPTIMIZATION.md)，使用入口 36；本页保留作为历史记录。

# 视频抓法复现版本

本版本保留完整视频中的五指轨迹，目标是复现源视频的杯身包握。当前已实现可执行的四指接触抓杯，但严格的视频忠实度验收仍未通过；不要将本版本的诊断 rollout 直接作为已准入训练示范。

## 两个版本

| 项目 | 已验证基线 | 视频抓法实验版本 |
| --- | --- | --- |
| Git 分支 | `main` | `video-faithful-v1` |
| 保留标签 | `physical-grasp-verified-v1` | 独立分支提交 |
| 轨迹来源 | 视频关键姿态加分阶段平滑控制 | 源帧 3–73 的完整五指轨迹 |
| 状态 | 原有窄初态物理示范准入通过 | 物理抓取通过，严格视频忠实度未通过 |
| 实时窗口 | `scripts/28_view_verified_grasp_gpu.sh` | `scripts/31_view_video_faithful_gpu.sh` |

基线代码位于 `5dea040b134a73507481ee71095d621707babaa7`，旧示范、重定向、准入报告均有 SHA-256 校验，未覆盖。原有 20 次短训和新示范训练入口保持原状。

## 源视频实际抓法

序列为 `20200709-subject-01/20200709_150949`，右手抓取 `025_mug`。检查了主相机 `840412060917` 以及 `836212060125`、`839512060362`、`841412060263`、`932122060861` 的同步图像。

视频显示拇指与其余手指相对，沿杯壁包握杯身；它不是仅由拇指和食指完成的二指捏取。主视角遮挡了部分手指，因此结合其他视角和 MANO 关节标签判断。源数据没有真实接触力标签，不能声称复现了人的逐指受力大小。

## 新版本做了什么

1. 保留全部 71 个有效源帧的五指对应关系，对每根手指的关节方向和指尖位置进行优化。
2. 手和杯共同使用原有任务坐标变换及 `object_scale=0.8`，在杯子坐标系中比较指尖误差。
3. 加入原 Adroit 固定腱的耦合限制。第一轮优化忽略了这些限制，导致几何姿态与真实执行姿态不同；新版本使用模型中的腱约束，保持原物理模型。
4. 给手指关节留出 0.04 rad 控制余量。使用平滑时间映射将视频轨迹放慢，约 11.84 秒完成接近、包握、抬杯和末态保持；动作时序保持源视频顺序，但速度不是原速。
5. 基于实际杯子位姿加入手腕与指尖反馈，并记录每根手指的接触力、接触比例和相对杯子的指尖误差。
6. 从保存初态开始，只执行保存的归一化动作，独立检查重放一致性；杯子随后完全由 MuJoCo 动力学运动。

## 当前结果和边界

最终候选在 `data/processed/seq_dexycb_001/video_faithful_v1/refined/`。其中 `best/diagnostic_rollout.pkl` 可供实时物理回放，`admission.json` 记录准入状态。

标称种子 0 的结果：

- 连续接触抬杯 6.55 秒，最终杯底高度约 80.9 mm。
- 末尾 1 秒中，拇指、食指、中指、无名指接触比例均为 100%；小指没有持续受力。
- 四指平均法向接触力依次约 3.01、3.86、0.28、1.28 N。
- 动作饱和率约 0.0366%，记录到的最大手杯负接触距离为 0。
- 保存动作独立重放的最大观测误差为 0。
- 整段平均指尖误差约 22.2 mm，末尾平均约 19.0 mm；拇指末尾误差约 26.8 mm。

种子 0–3 共四个窄初态均通过物理抓取门槛，四指包握持续存在。这只是杯子 XY 毫米级扰动验证，不代表广泛泛化。位置误差均以缩放后的仿真坐标计算。

严格忠实度门槛为：整段平均指尖误差 <20 mm、末尾平均 <15 mm、每根手指末尾平均 <25 mm，且末尾至少四指持续接触并包含拇指、食指。当前候选未满足全部条件，所以 `training_ready=false`。有界轨迹修正未得到更好的可通过候选，最终仍选择修正前的稳定方案。

## 直接打开仿真窗口

在 Ubuntu 桌面终端运行：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/31_view_video_faithful_gpu.sh
```

窗口循环执行新候选的保存动作，按 `Ctrl+C` 结束。只看一次可加 `--episodes 1`。对比旧版本仍使用：

```bash
bash scripts/28_view_verified_grasp_gpu.sh
```

两个入口播放的都是专家控制器产生的物理动作，不是本轮新训练的神经网络策略。本版本没有启动 DAPG。

## 复现命令

以下 Python 命令应在已配置 MuJoCo 路径的 `dexmv` 环境中执行。所有输出目录必须是新目录，脚本拒绝覆盖已有结果。输入大数据继续使用共享盘现有资源。

```bash
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
conda activate dexmv

python scripts/29_build_video_faithful.py --output data/processed/seq_dexycb_001/video_faithful_reproduction/retarget
python scripts/30_validate_video_faithful.py \
  --geometry data/processed/seq_dexycb_001/video_faithful_reproduction/retarget/geometry.npz \
  --output data/processed/seq_dexycb_001/video_faithful_reproduction/dynamics \
  --scales 4 --closures 0.15 --gains 160
```

后续应针对手指在杯壁上的接触位置与滑动修正剩余误差，并保留独立种子验收。达到上述忠实度门槛之后，再导出训练示范并开展短训练。
