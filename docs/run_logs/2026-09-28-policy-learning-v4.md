# v4：策略偏离诊断、BC 对照与数据扩展

## 结论

本轮没有启动 2000 次 DAPG，也没有覆盖 v3 示范或模型。旧的独立神经网络策略仍未学会可靠抓取；新增的“固定参考动作＋学习残差”能执行当前视频动作，但不能等同于通用抓取策略。

所有成功证据均来自初始化后通过 `env.step(action)` 控制手、杯子自由运动的回放。未逐帧写入杯子位姿，没有二次缩放动作，没有放松现有物理及忠实度门槛。

## 1. 首次偏离

`scripts/45_diagnose_policy_divergence.py` 对齐 v3 示范和旧 20 次 DAPG 策略的种子 0 到 5，初始状态完全一致。误差须持续 10 个控制步（100 ms）才记为偏离。

| 指标 | 阈值 | 首次偏离 |
| --- | --- | --- |
| 归一化动作 RMSE | 0.05 | 第 14 到 20 步，0.14 到 0.20 s |
| 手根部位置误差 | 10 mm | 第 20 到 22 步 |
| 手指关节 RMSE | 0.15 rad | 第 52 到 55 步 |
| 杯子位置误差 | 5 mm | 第 104 到 108 步 |

动作和手根部偏离发生在源视频第 3 帧的初始保持阶段，不是搬运末段才出错。抓不到的轨迹与能抬起却距目标约 10 到 11 cm 的轨迹均有早期偏离，不能只改最终目标。

CSV 分别记录接近、闭合、抬杯、搬运的动作、关节和杯子误差。阶段按参考帧 `<25.5/<40.5/<55/其余` 划分，是控制器阶段约定，并非独立人工视频标注。

产物：`data/processed/seq_dexycb_001/learning_v4/diagnosis/`，含 `diagnosis.json`、六个逐步 CSV、`matched_trajectories.png`。

## 2. 学习方法对照

GPU 监督训练，CPU MuJoCo 闭环评估；MLP 为 64×64，Adam 学习率 0.001，batch 128。此处是固定条件的新 BC 对照，不与旧上游 batch 32 初始化视为完全相同实验。

开发种子为 0、2、5；“完整通过”要求物理与忠实度同时通过。

| 方法 | 训练量 | 离线动作 RMSE | 开发集完整通过 |
| --- | --- | --- | --- |
| 原始 39 维观测 BC | 5 轮 | 0.03290 | 0/3 |
| 原始 39 维观测 BC | 300 轮 | 0.00433 | 0/3 |
| 观测＋速度＋杯子旋转＋时相，82 维 | 300 轮 | 0.00226 | 0/3 |
| 上述策略加专家纠偏 | 4 轮，每轮 100 epochs | 最终聚合 RMSE 0.01893 | 0/3 |
| 固定参考动作＋82 维网络残差，6 条示范 | 100 轮 | 0.000524 | 3/3 |
| 固定参考动作＋82 维网络残差，12 条示范 | 300 轮 | 0.000445 | 3/3 |

原始 BC 的 25 轮模型有 1/3 可抬杯，但不满足完整验收。单纯降低离线损失不足以保证闭环稳定。

专家纠偏在学生实际访问的状态上标注专家动作；采样时专家混合比例依次为 0.8、0.5、0.2、0。评估时关闭专家。四轮均未解决失败，不能宣称现有 DAgger 配方有效；失败纠偏轨迹也没有当作成功示范准入。

残差动作：`a(t) = a_reference(t) + network(observation, velocity, rotation, phase)`。参考动作固定在 checkpoint 中，不在线调用专家，也不写入杯子状态。依然依赖这条视频的时间表和初态范围，并非从零独立输出完整动作的策略。

## 3. 闭环结果与基线

6 条示范训练的残差模型 `bc_residual/policy_bc_100.pickle` 在未训练复位种子 20 到 29 上，物理与忠实度 10/10 通过。这仍是同一视频的小扰动，不代表新抓法泛化。

12 条示范训练的 `bc_residual_expanded/policy_bc_300.pickle` 在 10 个留出场景中：

- 位置变化：X/Y 各 -3 mm、-7 mm；目标变化：X/Y 各 ±15 mm；杯子独立旋转 ±3°。
- 物理通过 10/10；物理及忠实度同时通过 9/10。
- -3° 一例持续抬杯 8.38 s，但末段平均指尖误差 18.12 mm，超过 15 mm 忠实度门槛，不是抓杯物理失败。
- 平均最终目标误差：残差 20.24 mm，固定参考动作 18.16 mm；若增加 20 mm 精度指标，两者都只有 7/10。
- 固定参考动作本身在这些场景完整通过 10/10。因此现阶段不能把成功归因于网络，也不能宣称网络整体优于固定动作。
- 修订输入检查后追加种子 30：物理及忠实度通过，目标误差 18.59 mm。

物理门槛包含接触、持续抬杯、穿透、滑移、关节及有限值等；沿用的目标距离门槛为 100 mm。20 mm 是额外报告的更严格指标，没有事后悄悄改动准入标准。

留出结果一旦用于后续调参，应改称开发集，并另划新测试集。

## 4. 数据扩展及拒绝项

`scripts/48_expand_grasp_distribution.py` 对手、杯和人手参考施加一致刚体变换；目标偏移在搬运段平滑引入。每个候选检查手部 FK 一致性、关节限位、专家物理执行、保存动作重放及忠实度，未通过者不进入示范。

- 主训练集：原有 6 条＋6 条合格合成轨迹，共 12 条。新增为 X/Y 负方向 5 mm 位移、目标 X/Y ±10 mm。
- 正方向位移和整体场景 ±2° 旋转触发关节限位，被拒绝，没有裁剪关节掩盖问题。
- 另验证杯子独立旋转 ±5°、±10° 的 4 条合成轨迹，保存在 `orientation/`，未混入上述 12 条训练集。
- 独立真实视频来源数依然是 1，不能把合成数量写成视频数量。
- 新增第二段真实序列 `20200709_151032`，路径 `data/real_data/relocate_mug/seq_dexycb_002/`，共 74 帧、73 有效帧。RGB/深度仍链接共享盘，无重复下载。
- MANO 检查通过；73 帧重定向均收敛，静态平均指尖拟合误差 6.26 mm。
- 第二序列的初始控制移植和 12 组零手指修正／闭合时序搜索仍未抬杯，`training_ready=false`。静态拟合好不代表动力学可执行，未将其加入训练。

下一步应单独修第二段视频的拇指对握、接触建立与闭合参考，不继续照搬第一段的手指补偿。初始约 31 mm 指尖误差、小指未持续接触、缺少真实接触力标签的边界仍存在。

## 5. 文件与运行

新增：`45_diagnose_policy_divergence.py`、`46_compare_behavior_cloning.py`、`47_dagger_video_policy.py`、`48_expand_grasp_distribution.py`、`49_view_residual_policy_gpu.sh`、`src/fromrealhand/policy_learning.py` 和对应测试。

修改：29 支持独立序列及共同坐标变换；30 支持显式专家标注回调；31 支持选择源视频；41 支持闭合量搜索；44 区分普通策略与残差 checkpoint，修复 NumPy 布尔求和的 JSON 序列化错误。

主要输出均在 `data/processed/seq_dexycb_001/learning_v4/`。模型、示范、原始数据不上传 GitHub，仅同步代码与记录。输出目录必须是新目录，避免覆盖实验。

```bash
cd /home/smgbro/mujoconew/GITHUB
conda activate dexmv
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 以下输出目录名用于新复现实验，不覆盖本轮 learning_v4 目录。
python scripts/45_diagnose_policy_divergence.py --output data/processed/diagnosis_reproduction
python scripts/46_compare_behavior_cloning.py --features phase --residual \
  --epochs 25 100 --output data/processed/residual_reproduction
python scripts/48_expand_grasp_distribution.py --case-set train \
  --output data/processed/expansion_reproduction
python -m unittest discover -s tests -v

# Ubuntu 桌面终端打开 MuJoCo；这是保存的残差策略动作物理重放。
bash scripts/49_view_residual_policy_gpu.sh
```

验证：40 项 unittest 全部通过。窗口测试 `bash scripts/49_view_residual_policy_gpu.sh --episodes 1 --speed 8` 正常退出，1417 步观测重放误差为 0，无 `Missing GL version`。旧版窗口脚本 43 保留不变。

本轮额度按本地会话的 300 分钟窗口 usage 定期读取，训练／搜索默认 85% 停止新实验。收尾时保留余量用于测试、记录和同步，不承诺下一会话仍能自动读取相同额度来源。

## 6. 后续训练准入

1. 修复第二段独立视频，完成投影／接触审查以及纯动作物理重放；不通过不进数据集。
2. 在合格朝向数据上继续残差对照，解决 -3° 忠实度回退；以新留出集验证，报告与固定动作基线的配对差异。
3. 要求更严格的搬运精度与未见视频成功率，再决定是否用残差策略做强化学习。当前 checkpoint 的 82 维观测及固定参考动作不能直接交给原生 39 维 DAPG 入口；需要一致的环境包装与示范特征转换。
4. 包装与短训验收通过后才考虑 2000 次完整训练。本轮尚未完成这些条件。
