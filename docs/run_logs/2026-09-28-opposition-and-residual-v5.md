# v5：第二视频对握保持与残差的配对验证

## 本轮结论

第二视频已有稳定的物理抓杯候选，固定参数通过独立复位、动作重放和半步长检查，但仍不满足严格视频忠实度准入，未加入训练。残差网络在新合成场景上出现小幅优势，不能解释为跨真实视频泛化。

旧模型、示范和检查点未覆盖。没有启动 2000 次 DAPG。所有物理成功证据均来自初始化后 `env.step(action)`，没有逐帧写入杯子状态，没有改变杯子质量、摩擦或碰撞网格。

## 1. 更正旧失败描述

上轮将第二视频概括为“未抬杯”不够准确：`control_zero_v2` 的最佳候选能短暂抬杯，最高杯底约 124 mm、连续保持 2.05 s，但在源视频约第 57 到 59 帧丢失接触并滑落。旧候选的末段平均指尖误差 186.4 mm，最终目标误差 207.2 mm。

新增 `51_audit_opposition.py` 重新执行保存的动作，每步记录手指接触力、杯底高度和接触法向。拇指与另一手指承力法向点积低于 -0.3 记为对向接触；这是诊断指标，不是严格力封闭证明。

开发时修复了几何名称大小写以及模型加载后缓存视图的问题；`opposition_audit_before` 和 `opposition_audit_before_v2` 的零力结果作废。有效旧结果为 `opposition_audit_before_v3`，并与原始逐指力报告交叉校验。旧候选抬杯采样中 92.7% 有对向接触，但末段为零，因此主要问题是保持过程中失去对握，而不是从未形成对握。

## 2. 优化方法与冻结验证

以旧候选 `close=0.3`、Cartesian gain 75、approach gain 75、closure lead 0.15、time scale 5 为起点。用坐标下降搜索手指／手腕的关节参考偏置，步幅依次为 0.12、0.06、0.03 rad，偏置限幅 ±0.3 rad。评分优先惩罚物理失败，再考虑逐指误差和接触缺失。

搜索结果放在 `data/processed/seq_dexycb_002/opposition_v3/`；另有无名指专项搜索 `ring_contact_v4/`。不把搜索后期候选自动替换已验证版本。

主搜索共 67 个候选，最后最优候选的末段平均指尖误差 13.06 mm。无名指专项搜索共 13 个候选，最优为 12.57 mm，但仍未形成持续无名指接触，不能据此解除忠实度限制。

推荐物理查看版本是提前冻结并独立验证的 `opposition_verified_v3/`。参数快照 `frozen_parameters.json` 与来源哈希已保存。

| 指标 | 冻结候选 seed 0 |
| --- | --- |
| 连续抬杯保持 | 7.85 s |
| 末段最低杯底高度 | 153.50 mm |
| 末段平均指尖误差 | 13.96 mm |
| 最终目标距离 | 64.25 mm |
| 手杯最大穿透 | 0.861 mm |
| 手与全部场景最大穿透 | 0.911 mm |
| 末段滑移 | 0.0067 mm |
| 保存动作观测重放误差 | 0 |
| 抬杯采样的对向承力比例 | 99.87% |
| 末段对向承力比例 | 100% |

种子 0、10、11、12、13、14 的物理检查全部通过，保持时间 7.84 到 7.86 s。0 是搜索初态，10 到 14 未参与本次候选搜索；这些仍只是 ±2 mm 初始扰动。物理步长减半后，保存动作和闭环专家都通过物理检查，目标误差约 64.22 mm。

拇指、食指、中指末段持续接触，无名指和小指没有。现有严格忠实度要求至少四指持续接触，因此 `training_ready=false`。没有为了让结果通过而放宽门槛。另需注意：原物理任务的目标门槛为 100 mm，64 mm 并不是高精度搬运成功。

第二视频缓存的人手标签显示无名指／小指末段离杯面代理距离较大，但这不是实际接触力证据；在恢复原视频复核前，不擅自把三指准入改为“忠实复现”。

## 3. 残差是否有实际优势

将上轮 12 条数据与朝向变化数据合并，去除重复原始轨迹，共 16 条。合并先核验准入及 SHA-256，并检查重复键的全部状态／模型数据是否一致。仍只有第一段真实视频，第二视频没有混入。

训练沿用 82 维状态／速度／旋转／时相、64×64 网络、GPU BC，100 和 300 epochs 均在开发种子上 3/3 完整通过。配对测试使用预定的 300 epochs 模型，不根据测试结果选模型。

新测试场景为：X/Y 各 -4、-6 mm 初始位置，X/Y 各 ±18 mm 目标偏移，以及杯子 ±7° 旋转。它们未加入训练；比较同一初态下固定参考动作和固定参考动作＋神经残差，均不在线调用专家。

| 指标（10 个合成场景） | 固定参考动作 | 学习残差 |
| --- | --- | --- |
| 物理通过 | 10/10 | 10/10 |
| 物理＋忠实度通过 | 10/10 | 10/10 |
| 完整通过且目标误差 ≤20 mm | 7/10 | 8/10 |
| 平均目标误差 | 18.66 mm | 17.64 mm |
| 最差目标误差 | 36.32 mm | 24.72 mm |

逐场景距离比较，残差 7 胜 3 负。分组平均误差：

- 位置变化：14.46 → 15.20 mm，略差。
- 目标变化：23.02 → 20.29 mm，改善。
- 朝向变化：18.36 → 17.23 mm，改善。

结论只能是：在这一小组确定性合成测试中，残差降低了平均及最差终点误差，且未损失完整通过率。平均改善约 1 mm，不能宣称显著、全面或跨视频优势；只用了一个网络训练种子。此测试集未来用于调参后，必须另划留出集。

产物：`data/processed/seq_dexycb_001/learning_v5/merged/`、`bc_residual/`、`test/evaluation.json` 和 `test/paired_summary.json`。数据和模型留本机，不上传 GitHub。

## 4. 查看与复现

Ubuntu 桌面终端查看第二视频的新物理候选：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/54_view_second_video_gpu.sh
```

窗口已实际打开并跑完 1450 步，重放误差为 0。这是专家控制动作的物理回放，不是第二视频的训练策略。

本机 GPU 窗口 GLX 正常，但 GPU 离屏 EGL 初始化失败。CPU 离屏渲染成功，图片保存在 `opposition_verified_v3/visual_review_cpu/`；使用 `--simulation-only` 明确不读取原 RGB，不能把这些图片当作新的源视频对齐证据。

运行 Python 实验时使用已有 `dexmv` 环境与以下变量：

```bash
conda activate dexmv
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python -m unittest discover -s tests -v
```

关键命令（输出必须换成新目录，避免覆盖）：

```bash
python scripts/37_optimize_finger_reference.py \
  --candidate data/processed/seq_dexycb_002/control_zero_v2/admission.json \
  --output data/processed/seq_dexycb_002/opposition_reproduction \
  --max-trials 67 --steps .12 .06 .03
python scripts/53_verify_opposition_candidate.py \
  --geometry data/processed/seq_dexycb_002/retarget_v1/geometry.npz \
  --parameters data/processed/seq_dexycb_002/opposition_verified_v3/frozen_parameters.json \
  --output data/processed/seq_dexycb_002/verification_reproduction
python scripts/48_expand_grasp_distribution.py --case-set test-v5 \
  --policy data/processed/seq_dexycb_001/learning_v5/bc_residual/policy_bc_300.pickle \
  --output data/processed/seq_dexycb_001/learning_v5/test_reproduction
```

本轮共享 SSD 未连接。只读检查本机 `dd` 分区后确认不是该数据盘，已卸载恢复原状；没有修改挂载配置或迁移数据。用户明确同意跳过共享盘／原视频复核，MANO 资产测试因此跳过，不重新下载。

测试：43 项中 42 项通过、1 项 MANO 资产测试按缺少挂载跳过；`git diff --check` 通过。最终接触审计增加逐步力提取一致性校验，并重新通过 1450 步物理重放。

GitHub 可直接查看的小型证据文件：[配对指标](evidence/v5-paired-summary.json)、[冻结候选完整验收](evidence/v5-second-video-admission.json)、[对握接触审计](evidence/v5-opposition-audit.json)。仿真轨迹与模型检查点仍保留本机。

## 5. 仍然阻碍扩大训练的项目

1. 第二视频末段虽稳定，但目标误差约 64 mm。下一步需要接触保持条件下的目标闭环补偿，并保持相同物理门槛。
2. 第二视频四指忠实度门槛未通过。恢复共享盘后先复核该视频应要求的接触手指，不能靠降低门槛掩盖失败。
3. 残差优势仍小且局部，需要多网络训练种子、更多初态和独立真实视频验证；原来的 39 维 DAPG 入口不能直接使用此 82 维残差 checkpoint。
4. 本轮没有通过第二视频的训练准入，不导出它为成功训练示范，不启动完整强化学习。
