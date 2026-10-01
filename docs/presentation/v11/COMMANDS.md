# 运行命令与证据路径

工作目录：`/home/smgbro/mujoconew/GITHUB`。复用原 `dexmv` 环境，不升级遗留 MuJoCo 依赖。

```bash
cd /home/smgbro/mujoconew/GITHUB
conda activate dexmv
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

PYTHONPATH=src python -m unittest discover -s tests
python scripts/71_develop_safe_experts.py --resume
python scripts/72_train_corrective_residual.py
python scripts/76_select_residual_gain.py
python scripts/74_build_v11_evidence.py
# 检查并提交 evidence/frozen-policy.json 与实现后，才允许下面的新留出评估。
python scripts/73_evaluate_corrective.py --freeze data/processed/dual_video_v11/learning/safety_frozen_policy.json
python scripts/74_build_v11_evidence.py
```

这些不是需要再次全部执行的安装步骤。已有输出时大部分实验脚本会拒绝覆盖；必须使用新实验目录和新的测试协议，不能删除本轮证据后反复挑测试结果。`71 --resume` 仅跳过已完成且协议一致的工况。

## 结果位置

| 内容 | 本机路径，相对仓库根目录 |
|---|---|
| 预注册 | `configs/v11-study.json` |
| 专家、失败候选、半步长报告 | `data/processed/dual_video_v11/experts/` |
| 根部单独优化诊断存档 | `data/processed/dual_video_v11/diagnosis/` |
| BC、纠偏标签、开发策略回放 | `data/processed/dual_video_v11/learning/` |
| 冻结后留出全部回放 | `data/processed/dual_video_v11/heldout/` |
| 截图、曲线、精简指标、方法 | `docs/presentation/v11/` |

完整轨迹、checkpoint 和带授权限制的原始数据不上传 GitHub；GitHub 保存代码、协议、哈希与小型证据。

## 打开原生仿真窗口

选中的策略完成冻结后，可以直接在 Ubuntu 桌面终端运行：

```bash
python3 scripts/75_view_corrective_gpu.py second --episodes 1
python3 scripts/75_view_corrective_gpu.py first --episodes 1
python3 scripts/75_view_corrective_gpu.py second --expert --episodes 1
```

从 `learning/frozen_policy.json` 确认选中的策略目录，再从对应开发报告中找到 `diagnostic_rollout.pkl` 和该工况的 `geometry.npz`：

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
bash scripts/31_view_video_faithful_gpu.sh \
  --rollout /完整路径/diagnostic_rollout.pkl \
  --geometry /该工况完整路径/geometry.npz --episodes 1
```

窗口播放的是已保存的闭环策略动作，经物理引擎重放；不是实时重新推理策略，也不是逐帧设置杯子位姿。重新推理的定量证据由脚本 72/73 生成。本轮离屏截图验证不代表已经重新测试了这条交互窗口命令。

## 环境检查记录

一次测试命令遗漏动态库环境变量，触发了 `mujoco_py` 重编译；已终止该测试进程并使用上述固定库路径重新执行。早期 76 项测试中 1 项因 MANO 未挂载而跳过；随后共享盘恢复且补充两项测试，78 项全部通过。没有据此升级 Python、NumPy 或 MuJoCo。
