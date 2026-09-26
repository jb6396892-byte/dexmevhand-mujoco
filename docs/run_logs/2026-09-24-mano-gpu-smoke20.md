# GPU DAPG 短训练记录（2026-09-24）

- 任务：`relocate-mug-0.8`，状态输入，MANO 真实演示 1 条（292 步）。
- 配置：`configs/dapg-mug-mano-smoke.yaml`，BC 初始化 + DAPG；20 次迭代，每轮 5 条采样、2 条评估，2 个 CPU 进程。
- 运行名：`mano_gpu_smoke20`；默认上游入口 `dexmv-sim/examples/train.py`，价值函数使用 GPU。MuJoCo 采样及策略主要计算仍在 CPU。
- 系统：RTX 4060 Laptop GPU、NVIDIA 595.91.07、PyTorch 1.13.1+cu117；CUDA 前向、反向传播和参数更新验证通过。
- 20 次迭代均完成，输出 `best_policy.pickle`；策略 3356 个参数均为有限值。
- 第 0 次采样 / 评估回报：-19.37 / 9.27；第 19 次：31.61 / -16.64。采样最高回报为 31.61，但评估波动明显。
- 独立确定性回放 400 步：总奖励 -4.47，手杯接触 0 步；杯子初始 / 最高高度均约 0.0403 m，未抬起。保存 8 张离屏图像。

训练输出：`training_log/dapg_relocate-mug-0.8_relocate-mug-mano-real_0.1_100_mano_gpu_smoke20_seed200/`。
独立回放：`data/processed/seq_dexycb_001/policy_gpu_smoke20/`。

结论：GPU 训练链路可运行，但当前单条演示没有产生有效抓取策略；不能用训练回报上升替代接触和抬升成功率。先检查演示与环境重置时的手杯初始位姿及坐标范围，再扩充多条有效轨迹和多次独立评估，不直接执行 2000 次完整训练。
