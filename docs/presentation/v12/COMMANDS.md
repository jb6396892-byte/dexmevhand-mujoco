# v12 操作与查看

## 环境

```bash
cd /home/smgbro/mujoconew/GITHUB
conda activate dexmv
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

## 已执行实验的复现入口

已有结果时脚本拒绝覆盖。不要删除已看过的测试结果反复选模型，新的实验应使用新协议与输出目录。

```bash
python scripts/77_run_v12_study.py train --mode aligned_bc
python scripts/77_run_v12_study.py develop --mode aligned_bc
python scripts/77_run_v12_study.py train --mode contact_reference_bc
python scripts/77_run_v12_study.py develop --mode contact_reference_bc
python scripts/81_compare_contact_guard.py
python scripts/77_run_v12_study.py freeze
# 提交 docs/presentation/v12/evidence/frozen-policy.json 后才运行：
python scripts/77_run_v12_study.py heldout
python scripts/78_train_residual_dapg_smoke.py --train
python scripts/79_build_v12_evidence.py
PYTHONPATH=src python -m unittest discover -s tests
```

开发评估支持在协议和 checkpoint 哈希相同的情况下继续未完成的工况；新留出测试不用于调参。`78` 只做 20 次接口短训，不是 2000 次完整训练。

## 原生 MuJoCo 画面

在 Ubuntu 桌面终端中运行，不需要打开 MP4：

```bash
cd /home/smgbro/mujoconew/GITHUB
python3 scripts/80_view_v12_gpu.py second --episodes 1
python3 scripts/80_view_v12_gpu.py first --episodes 1
# 检查 DAPG 短训结果，而非冻结 BC：
python3 scripts/80_view_v12_gpu.py second --dapg --episodes 1
```

这会打开 MuJoCo 原生窗口。显示的是保存的闭环策略动作经物理引擎重放，不是在线重新推理，也不是逐帧写杯子位姿。绿色透明杯子是无碰撞目标标记。无桌面会话时只能离屏渲染。

本机完整结果：`data/processed/dual_video_v12/`。GitHub 小型证据：`docs/presentation/v12/evidence/`。
