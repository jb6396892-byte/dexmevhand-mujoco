# GPU 短训策略的真实动力学回放（2026-09-25）

先前 `scripts/19_visualize_aligned_gpu.sh` 每帧设置手和杯子的记录位姿，属于运动学可视化，不能用于判断抓取是否成功。训练后的策略权重位于 `training_log/dapg_relocate-mug-0.8_relocate-mug-mano-real_0.1_100_mano_gpu_smoke20_seed200/iterations/best_policy.pickle`，由 20 次 DAPG 短训得到。

现在用 `scripts/20_visualize_trained_policy_gpu.sh` 打开 MuJoCo 窗口，内部 `scripts/20_replay_trained_policy.py` 以训练采样参数创建 `YCBRelocate`：`object_scale=0.8`、`randomness_scale=0.25`、`solref=-6000 -300`、摩擦系数 `(1, 0.5, 0.01)`。从正常 `reset()` 开始，逐步把策略确定性动作送入 `env.step(action)`；不写入手或杯的 `qpos`。

种子 200、400 步的结果：总奖励 `70.48`，手杯接触 `382` 步；杯子初始高度 `0.04065 m`、最高 `0.04400 m`，没有接触中抬升 `15 mm` 的连续帧，**抓取未成功**。详细数值在 `data/processed/seq_dexycb_001/policy_gpu_smoke20/live_physics_seed200.json`。

旧的 `scripts/16_evaluate_policy_offscreen.py` 使用环境默认随机性和接触参数，曾得到 0 步接触；它与本次训练一致参数的结果不能直接比较。两种检查一致说明该 20 次短训模型不会抬杯。下一步先修示范的手杯几何和可执行动作，再训练并评估多种复位初态。
