# 命令与执行记录

工作目录：`/home/smgbro/mujoconew/GITHUB`。

## 无渲染检查与训练

以下环境变量作用于当前终端；没有升级依赖。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
PY=/home/smgbro/miniconda3/envs/dexmv/bin/python

$PY scripts/66_develop_second_video.py
$PY scripts/67_prepare_multivideo.py
$PY scripts/68_compare_multivideo.py
# 先提交 frozen_policies.json 的小型证据副本，再执行下一条。
$PY scripts/69_evaluate_multivideo.py
$PY scripts/70_build_v10_evidence.py
PYTHONPATH=src $PY -m unittest discover -s tests -v
```

以上实验输出默认使用新目录，已存在时拒绝覆盖。不要原样重复最终留出评估后再挑选结果。若复现实验，应使用新的独立输出路径，并声明复现而非新的独立测试。

## 截图

```bash
$PY scripts/31_view_video_faithful.py \
  --geometry data/processed/seq_dexycb_002/retarget_v1/geometry.npz \
  --rollout data/processed/seq_dexycb_002/dynamic_contact_v9_verified/nominal/diagnostic_rollout.pkl \
  --simulation-only --output data/processed/dual_video_v10/v9_cpu_screenshots
```

CPU 离屏渲染成功，重放观测误差 0。尝试 GPU 隐藏 GLFW 窗口时出现 `GLX: Failed to find a suitable GLXFBConfig`；该试验参数没有保留到正式代码。交互仿真仍沿用 `bash scripts/65_view_dynamic_contact_gpu.sh`。

## 中断记录

首轮 BC 在第 50 epoch 的首条开发回放结束后，统计代码使用了大写 `TH_force_n`，而原模拟器使用小写 `th_force_n`，因此统计输出中断。改用源模块的 `FINGERS` 常量后修复。中断目录保留为 `data/processed/dual_video_v10/learning_interrupted_metrics_key`；正式训练使用相同协议、相同种子重跑，没有根据中断表现更改网络或训练参数。

## 数据边界

GitHub 只同步代码、协议、小型 JSON 指标、方法说明和模拟截图。原视频、MANO、原示范大文件与 checkpoint 保留本地；未重新下载共享盘数据。
