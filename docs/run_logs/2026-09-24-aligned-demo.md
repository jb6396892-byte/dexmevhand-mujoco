# 对齐版 MANO 示范及动力学验收（2026-09-24）

原始 `relocate-mug-mano-real.pkl` 和训练结果不变。新结果单独保存在 `data/demonstrations/relocate-mug-mano-aligned-real.pkl`。

## 处理

1. 以视频第 20 帧为示范起点，依据该帧杯子位姿生成 `calib/camera_to_mujoco.npy`。对手、杯子和由末帧杯子位姿生成的目标使用同一个 4x4 变换，将杯子初始中心放到 `(-0.0625, 0, 0.04065) m`；不改原始外参。这与 DAPG 采样环境的杯子初始 x/z 一致。
2. 用 `--limit-global-pose` 重新重定向 MANO 手部，防止手根关节大幅超出 MuJoCo 范围。
3. 用 `--aligned-task-frame` 生成 213 步示范，跳过上游 hindsight 平移；在采集观测与逆动力学动作前执行 `sim.forward()`，并夹紧插值造成的微小手根超限。
4. 用 `scripts/18_verify_dynamic_demo.py` 在 DAPG 采样环境 `relocate-mug-0.8` 中比较重置、示范强制位姿和从示范初态正常执行动作的结果。脚本在动态抓取未重现时返回非零退出码。

## 验收

| 检查 | 结果 |
| --- | --- |
| 杯子初始中心 | `(-0.0625, 0, 0.04065) m`，与 DAPG 采样环境 x/z 一致；y 受复位随机性影响 |
| 保存观测 vs 同帧 MuJoCo 观测 | 最大绝对误差 `0` |
| 六个手根关节超出致动器控制范围比例 | 全部 `0` |
| 强制位姿回放 | 194 帧接触，杯子最高 `0.2215 m`；此检查直接设置状态，不证明动作可执行 |
| 保存动作的正常动力学回放 | 3 帧接触，杯子最高 `0.0781 m`，但接触时没有连续抬升 `15 mm`；**未通过** |

剩余偏差：训练环境以种子 200 复位时，杯子为 `(-0.0625, 0.0134, 0.04065) m`，掌心为 `(-0.0069, -0.2000, 0.1504) m`，目标为 `(-0.1284, -0.0022, 0.1576) m`；示范首帧的杯子、掌心、目标分别为 `(-0.0625, 0, 0.04065)`、`(-0.1181, 0.0689, 0.0782)`、`(-0.0785, 0.0271, 0.2200) m`。共同平移保持了视频中的手杯相对位姿，但不能同时匹配训练复位的掌心和随机目标。示范动作由上游逆动力学估算，并非轨迹闭环控制；正常动力学回放不能稳定抓取。下一步需要校准手杯几何、定义与示范兼容的复位分布，并拟合/验证可执行动作，**不能把这份示范投入完整 DAPG 训练**。

重复验证（在项目根目录，并按现有 DexMV 环境设置 `LD_LIBRARY_PATH`、`LD_PRELOAD` 和 `PYTHONPATH`）：

```bash
SEQ=data/real_data/relocate_mug/seq_dexycb_001
CONDA=/home/smgbro/miniconda3/bin/conda
$CONDA run -n dexmv python scripts/17_align_task_frame.py --sequence-dir "$SEQ" --reference-index 20 --output "$SEQ/calib/camera_to_mujoco.npy"
$CONDA run -n dexmv python scripts/03_retarget_one.py --hand-dir "$SEQ/hand_pose_mano" --output "$SEQ/retargeting_mano_aligned.pkl" --camera-to-world "$SEQ/calib/camera_to_mujoco.npy" --invalid-policy nearest --limit-global-pose
$CONDA run -n dexmv python scripts/05_generate_demo.py --sequence-dir "$SEQ" --retargeting "$SEQ/retargeting_mano_aligned.pkl" --camera-to-world "$SEQ/calib/camera_to_mujoco.npy" --skip-frame 20 --aligned-task-frame --output data/demonstrations/relocate-mug-mano-aligned-real.pkl --trajectory-id seq_dexycb_001_aligned
$CONDA run -n dexmv python scripts/18_verify_dynamic_demo.py data/demonstrations/relocate-mug-mano-aligned-real.pkl --output data/processed/seq_dexycb_001/aligned_demo_validation.json
```

详细数值见 `data/processed/seq_dexycb_001/aligned_demo_validation.json`；原始数据和生成的 pkl 均不上传 GitHub。
