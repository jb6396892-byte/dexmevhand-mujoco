# v9：第二视频达到当前场景示范最低准入要求

## 结论与停止边界

**第二视频无名指持续承力问题已在当前场景修复，单条示范通过物理、数值抓法、保存动作重放和半步长验收。** 已停止优化，未启动新的 BC、DAPG、长轮次训练或新场景泛化测试。

这里的“可训练”是当前场景单条示范准入，不代表已训练出新策略，也不代表跨初态、跨视频或实机泛化通过。原三指稳定版本和现有 checkpoint 全部保留。

## 优化方法

新增 `scripts/64_optimize_dynamic_contact.py`，直接通过 MuJoCo 动力学优化动作，而不是对固定杯子做静态 IK：

1. 使用 v8 已有完整动作作为初值。先从原视频初态正常执行到控制步 850，缓存这个实际到达的状态；每个搜索候选从同一状态继续执行。
2. 对 30 维执行器动作加入有界修正，在步 850 到 1150 之间用五次平滑函数逐渐启用。平移修正按 1 cm、其他自由度按 0.12 rad 的等效控制尺度参数化，并通过既有执行器增益换算成归一化动作。
3. 每个候选均由 `env.step(action)` 推进。杯子具有原有质量、摩擦和自由关节；不逐帧写入杯子 qpos，不施加外部托举力，不修改碰撞网格或 margin。
4. 代价同时考虑无名指到杯子的间隙、四指承力、原视频指尖相对几何、杯子目标位置、运动速度、全手穿透、动作饱和及修正幅度。远距离网格间隙只用于引导；最终接触必须由原生碰撞和接触力确认。
5. 先用有限差分最小二乘改善动作。遇到接触状态跳变后，改用 SciPy 的有界 Powell 搜索，联合调整手整体姿态、手腕和其他手指。
6. 出现持续四指接触候选后手动停止 Powell，冻结已经保存的第 240 号候选，再从最初状态完整执行 1450 步。这里没有宣称优化器已收敛或找到全局最优解。

关键区别：最终 Powell 候选的无名指四个修正参数，与进入 Powell 时完全相同。改进来自手整体、手腕及食指/中指动作的配合，使自由杯子进入无名指可承力的位置，而不是继续增大无名指闭合力度。这是当前候选的参数对照结果，不是所有抓法的通用规律。

搜索记录保存了 132 次粗差分、101 次细差分和至少 260 次 Powell 评估。它们都是同一场景的开发候选，不是独立测试场景；Powell 在一轮评估中被主动中断，所以不把落盘的 260 次冒充精确总评估次数。

## 验收结果

完整证据：[v9 动力学验收摘要](evidence/v9-dynamics-summary.json)。标准步长为 2 ms，半步长为 1 ms，控制周期均为 10 ms。

| 指标 | 门槛 | 标准步长 | 半步长 |
| --- | --- | ---: | ---: |
| 末秒无名指承力接触占比 | >=80% | 100% | 100% |
| 末秒拇指/食指/中指占比 | 各 >=80% | 100% / 100% / 100% | 100% / 98% / 100% |
| 无名指平均接触力 | 实际承力，不以几何相交替代 | 2.03 N | 2.87 N |
| 最终目标距离 | <=20 mm | 5.25 mm | 7.86 mm |
| 最大手场景穿透 | <=1 mm | 0.954 mm | 0.796 mm |
| 初始穿透 | <=0.5 mm | 0.109 mm | 0.109 mm |
| 承力接触最大间隙 | <=0.5 mm | 0.200 mm | 0.200 mm |
| 末秒杯子相对手掌滑移 | <=5 mm | 0.019 mm | 0.054 mm |
| 动作饱和比例 | <1% | 0.423% | 0.423% |
| 关节越界 | <=0.02 rad | 0 | 0 |
| 末段平均指尖误差 | <15 mm | 12.39 mm | 12.65 mm |
| 最大单指末段平均误差 | <25 mm | 15.18 mm | 14.55 mm |

标准步长保存动作重放的最大观测误差为 **0**。连续抬杯保持约 7.86 s，末秒杯底离桌约 182 mm（半步长约 180 mm）。全部三次验收均通过现有物理及数值抓法门槛。

旧 `transport_v5` 三指版本的目标误差约 1.60 mm，比本版本更小。因此本次新增成果是持续四指承力和训练准入，不是宣称到位精度也超过旧版本。

## 仍然存在的边界

- 无名指首次承力在控制步 1080 后，即约 10.81 s；此时杯心已经抬到约 224 mm。当前是先三指抓起、后四指保持的物理可执行过渡，不是已经证明与原视频逐帧接触时序完全相同。
- 小指末秒没有持续接触；现行门槛是至少四指，不是五指。
- 原视频没有真实接触力标签，表中力值是 MuJoCo 预测值，不代表对实物手指力的准确恢复。
- 只有第二视频当前场景的一条新示范。尚未做新复位种子、新目标、新杯朝向等泛化检查。
- 单条轨迹相对自身动作参考的残差标签全为零，不能用它独自证明残差学习有效。后续需要经用户确认的数据扩展及策略闭环对照，而不是直接宣称适合 2000 次完整训练。

## 已冻结产物

- 优化动作与参数：`data/processed/seq_dexycb_002/dynamic_contact_v9_powell/best_actions.npy`、`best.json`。
- 正式准入：`data/processed/seq_dexycb_002/dynamic_contact_v9_verified/admission.json`。
- 全程物理回放：该目录下的 `nominal/diagnostic_rollout.pkl`。
- 重放与半步长证据：`saved_replay/`、`half_timestep/`。
- 独立命名示范：`data/demonstrations/relocate-mug-second-dynamics-v9.pkl`。
- 示范 SHA-256：`0d36fc6b59dad849aaef6cb4b8994381f93a6001406723d7291c412957e3b081`。

示范包含 1 条轨迹、1450 步，观测 39 维、动作 30 维，必需字段完整。与已有学习入口相同的 phase 特征预处理得到 `(1450, 82)`，全部状态、特征和动作有限。仅检查输入，没有创建或更新策略网络。

三项受保护的原始成果哈希不变。66 项 unittest 通过，shell 语法检查与 `git diff --check` 通过。

## 打开 MuJoCo 窗口

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/65_view_dynamic_contact_gpu.sh
```

这是优化专家动作的物理重放，不是新训练策略；默认循环播放，Ctrl+C 退出。已实际运行 `--episodes 1 --speed 3`，窗口正常创建、播放结束，观测重放误差为 0。

离屏截图入口仍报 `Failed to initialize OpenGL`，本轮没有生成仿真截图；不把该问题隐瞒为已解决。它不影响已验证的交互窗口和数值物理检查。

## 复现命令

```bash
conda activate dexmv
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

python scripts/64_optimize_dynamic_contact.py \
  --candidate data/processed/seq_dexycb_002/contact_inspection_v7/admission.json \
  --rollout data/processed/seq_dexycb_002/continuation_v8_root_effort_tracking/gain_2/diagnostic_rollout.pkl \
  --initial data/processed/seq_dexycb_002/dynamic_contact_v9_fine/best.json \
  --method powell --max-nfev 20 --max-evaluations 1500 \
  --output data/processed/seq_dexycb_002/dynamic_contact_v9_powell_rerun

python scripts/64_optimize_dynamic_contact.py \
  --candidate data/processed/seq_dexycb_002/contact_inspection_v7/admission.json \
  --rollout data/processed/seq_dexycb_002/continuation_v8_root_effort_tracking/gain_2/diagnostic_rollout.pkl \
  --verify-actions data/processed/seq_dexycb_002/dynamic_contact_v9_powell/best_actions.npy \
  --output data/processed/seq_dexycb_002/dynamic_contact_v9_verify_rerun

python scripts/06_validate_demo.py data/demonstrations/relocate-mug-second-dynamics-v9.pkl --warn-saturation .01
python -m unittest discover -s tests -q
```

所有新输出目录必须不存在；不要覆盖冻结结果。优化程序不调用训练或泛化入口。下一步须先向用户汇报并等待是否进入训练/数据扩展的决定。
