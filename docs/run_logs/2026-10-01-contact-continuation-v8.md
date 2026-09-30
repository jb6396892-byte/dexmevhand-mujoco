# v8：第二视频连续接触与动作可执行性

## 范围与准入状态

本轮只修复第二视频当前场景。不启动 BC、DAPG、长轮次训练或新的泛化实验；不改变原始数据、示范和 checkpoint。`transport_v5` 原稳定三指版本继续保留。

**第二视频尚未通过持续四指接触，不能纳入合格训练示范。** 静态姿态可接触、控制动作可运行和自由杯子物理抓取通过，分别记录，不互相替代。

## 修正旧诊断依据

同一保存状态（控制步 850），仅改变检测 margin，旧版 MuJoCo 的凸体接触距离也会明显变化：

- 原近表面参数下，最大穿透约 0.375 mm。
- 将检测 margin 扩大至 30 mm 后，最大穿透约 6.080 mm。
- 同一食指/杯子碰撞对，距离从 -0.375 mm 变成约 -4.866 mm。

这不是动态运动造成的差异。因此 v7 扩大 margin 的静态搜索只能作为历史实验，不能再用于证明可达性或精确间隙。`62_probe_ring_reachability.py` 默认拒绝运行旧方法，只有显式 `--legacy-expanded-margin` 才能复现历史诊断。[对照证据](evidence/v8-margin-diagnostic.json)。

新脚本 `63_plan_contact_transition.py` 恢复保存的模型与原近表面碰撞参数，整个几何求解不扩大 margin。远距离使用胶囊轴采样点到杯子网格顶点的近似距离；进入接触时采用原生碰撞查询。前者只是优化引导，不是精确的凸碰撞距离或真实接触证据。

## 实施方法

1. 从已执行的三指稳定回放抽取状态，以第 850 步为不变过渡锚点，每 50 步设置一个关键帧。
2. 对手腕和手指做 SLSQP 连续求解，保留拇指、食指、中指支撑，逐步缩小无名指间隙。约束全手碰撞、肌腱范围、关节范围和原始人手标签误差。
3. 区分优化器“收敛”和结果“独立可行”；对肌腱进行线性投影后重新检查约束。采用 PCHIP 插值连接关键帧。关键帧可行不代表插值全程或动力学可行。
4. 将轨迹增量换成执行器动作，只经 `env.step(action)` 执行。动态杯子始终自由运动，不逐帧写入杯子位姿，不改变质量、摩擦或网格。
5. 依次试验关节反馈、仅无名指反馈、有限局部驱动力、杯子相对位姿补偿和基于实际回放的第二轮重规划。
6. 进一步在规划中约束预估归一化动作余量，限制无名指侧摆，并允许手整体姿态参与。预估动作余量不替代实际饱和统计。

## 已验证的失败原因

第一条原参数规划共 13 个关键帧全部几何可行，末段无名指原生接触距离约 -0.18 mm，规划最大穿透约 0.661 mm。但动态回放没有持续无名指接触。

基础跟踪候选通过物理门槛，目标误差 15.42 mm、最大穿透 0.954 mm，但末段仍只有拇指、食指、中指接触。2 N 上限局部驱动力候选的末帧无名指中节近似间隙仍约 3.89 mm。此时多数关节已接近规划值，杯子实际位姿却已相对规划变化，不能继续把误差归为“关节跟踪不够强”。

全量杯子相对位姿随动导致约 281 mm 的目标偏离，已拒绝；限幅 25% 随动仍无无名指接触，目标误差约 24–26 mm，也不采用。

第二轮以实际回放重新规划，静态全路径再次可行，但动态目标误差升至约 23 mm。低增益候选的动作饱和比例约 1.12%，超过 1% 门槛，主要在拇指。加入有界间隙积分后，无名指侧摆和拇指均出现饱和，总比例约 1.81%，仍无持续无名指接触。继续增大局部驱动力不是已验证的修复方法。

本轮没有任何候选因“视觉上接近”而获准训练，也没有降低四指标准。

## 最终结果

加入整体手姿态和驱动余量的规划也得到 13/13 可行关键帧：末帧无名指原生距离 -0.388 mm，平均指尖误差约 3.47 mm，规划全程关键帧最大穿透约 0.932 mm。关键帧预估动作绝对值最大 0.865，无名指侧摆最大 0.298 rad，均留有余量。它仍然不是动态成功。

| 动态反馈增益 | 物理门槛 | 目标误差 | 无名指末段接触 |
| --- | --- | ---: | ---: |
| 0 | 通过 | 24.35 mm | 0% |
| 0.5 | 通过 | 21.95 mm | 0% |
| 2 | 通过 | 19.84 mm | 0% |

本轮共 21 次当前场景开发回放，12 次物理门槛通过，0 次四指抓法通过。没有新种子或新场景评估，也没有启动训练。[逐组摘要与输入哈希](evidence/v8-continuation-summary.json)。这些结果没有超越原稳定三指版本，不替换它。

三项受保护的原始成果哈希检查全部不变，输入回放哈希不变。62 项 unittest 通过，`git diff --check` 通过；旧 MANO 依赖有弃用和只读数组警告，但没有测试失败。本轮未打开新的 MuJoCo 可视化窗口；上述结论来自数值物理回放。

结论是**问题尚未解决，停止于修复阶段**。当前证据支持的下一步是把杯子动力学和接触力平衡纳入动作优化，在三指支撑向四指支撑转换时预测自由杯子的运动。不能再只对固定杯子姿态做 IK，再期待开环动作自然形成同一接触。接触力平衡是下一步待验证的改进方向，不是本轮已证实的唯一根因。

本轮没有合格候选，因此没有为失败版本继续做保存动作一致性、半步长或新场景验收。后续候选至少先通过当前场景物理及四指门槛，再做保存动作与半步长检查；开始长训练或泛化前，停下通知用户。

## 文件与复现

输入保留：

- `data/processed/seq_dexycb_002/contact_inspection_v7/admission.json`
- `data/processed/seq_dexycb_002/contact_inspection_v7/best/diagnostic_rollout.pkl`
- 原回放 SHA-256：`0958189e161674d0b5a0899b14c73dd663a5ba7f872ba8efdc1837920ff1df23`

所有新产物在 `data/processed/seq_dexycb_002/continuation_v8*` 独立目录。`plan_admission.json` 中的 `planning_ready` 只表示几何规划结果；`status.json` 中训练准入仍为 false。失败目录也保留，不挑选性删除。

```bash
conda activate dexmv
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

python scripts/63_plan_contact_transition.py \
  --candidate data/processed/seq_dexycb_002/contact_inspection_v7/admission.json \
  --rollout data/processed/seq_dexycb_002/contact_inspection_v7/best/diagnostic_rollout.pkl \
  --free-root --effort-margin .02 \
  --output data/processed/seq_dexycb_002/continuation_v8_root_effort_rerun

python scripts/63_plan_contact_transition.py \
  --candidate data/processed/seq_dexycb_002/contact_inspection_v7/admission.json \
  --rollout data/processed/seq_dexycb_002/contact_inspection_v7/best/diagnostic_rollout.pkl \
  --plan data/processed/seq_dexycb_002/continuation_v8_root_effort_rerun \
  --gains 0 .5 2 \
  --output data/processed/seq_dexycb_002/continuation_v8_root_effort_tracking_rerun

python -m unittest discover -s tests -q
```

只有 `planning_ready=true` 的完整规划允许传给 `--plan` 进行物理跟踪；输出目录必须不存在。尚未通过的动态候选不进行新种子扩展。

## 额度与停止边界

修复了额度读取器把过期会话记录误当当前额度的问题：只接受五分钟内、尚未重置的五小时窗口数据。本轮读取结果为 `null`，即未知，不能报告剩余百分比或保证准确在剩余 20% 时停止。有有效记录时，脚本在已用 80% 停止新增候选。

本轮采用有限批次控制实验。训练和新场景泛化都保持关闭；达到准入条件后也需要先通知用户。
