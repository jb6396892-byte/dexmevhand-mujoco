# v14 操作与复现

## 实验环境

```bash
cd /home/smgbro/mujoconew/GITHUB
conda activate dexmv
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

本机为 RTX 4060 Laptop 8 GB。旧版 MuJoCo 和策略自然梯度在 CPU 执行，价值网络使用 CUDA；这不是 GPU 并行物理仿真。没有升级旧版 Python / NumPy / MuJoCo。

## 原生仿真窗口

在 Ubuntu 桌面终端运行，不需要 MP4 播放器：

```bash
cd /home/smgbro/mujoconew/GITHUB
python scripts/89_view_v14_gpu.py second
python scripts/89_view_v14_gpu.py second --case cup_y_minus
python scripts/89_view_v14_gpu.py first
```

短训完成后：

```bash
python scripts/89_view_v14_gpu.py second --dapg
```

窗口播放的是闭环控制器曾经生成的实际动作，通过 `env.step(action)` 再执行。只在开头设置初态，杯子随后自由运动；这不是在线网络推理，也不是逐帧写杯子位姿的动画。脚本沿用已经配置的 NVIDIA PRIME / GPU 渲染入口；本轮离屏回放验证不替代桌面窗口测试。

## 按顺序重现实验

```bash
python scripts/86_v14_contact_readiness.py develop --config configs/v14c-study.json
python scripts/86_v14_contact_readiness.py freeze --config configs/v14c-study.json
```

冻结凭据必须先提交到 Git，再运行新留出测试；策略、配置、执行代码的 SHA256 均受检查。

```bash
python scripts/87_train_v14_dapg.py --config configs/v14c-study.json
python scripts/86_v14_contact_readiness.py heldout --config configs/v14c-study.json
python scripts/88_build_v14_evidence.py --config configs/v14c-study.json
```

已有结果目录不会被覆盖。以上是历史实验命令，不应对同一目录反复重跑；新增实验需要独立版本与测试种子，并在看结果之前写入协议。原留出集一旦用于调参，就不能继续称为独立测试。

## 长训练入口

```bash
python scripts/87_train_v14_dapg.py --config configs/v14c-study.json --long --iterations 200
```

此命令这里只作说明，本轮没有执行。入口要求 `data/processed/dual_video_v14c/readiness.json` 全部通过且冻结哈希一致，目前已满足。先用 200 次验证学习趋势，再决定是否使用 `--iterations 2000`。长训练从经过留出测试的冻结 BC 加预测修正器开始，而不是默认采用未经留出测试的短训末次策略。当前迭代均值约 65.8 秒，200 次约 3.7 小时，2000 次约 36.6 小时，仅为同负载下的粗估。

## 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
git diff --check
```
