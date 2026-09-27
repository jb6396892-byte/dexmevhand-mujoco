# 近表面物理抓取优化

## 结论与范围

`surface_grasp_v2` 已通过本页定义的近表面接触验收，以及 6 个窄初态和半步长检查。它不是硬件安全认证，不代表任意初态都能抓取，也不代表忠实复现视频抓法。严格视频忠实度仍未通过，`training_ready=false`，本轮没有训练神经网络。

原版本全部保留。旧的“物理通过”仅代表旧门槛；旧版存在约 2.40–2.83 mm 的承力间距，不应再据此宣称真实贴合抓取合格。

## 优化方法

1. 将手、杯碰撞几何的接触余量统一为 0.2 mm，`gap=0`。不移动杯子掩盖误差，不放大网格，不提高摩擦，不改质量与重力。
2. 保留视频全轨迹及原 Adroit 腱约束，使用原生位置执行器、关节阻尼、偏置力补偿，以及相对实际杯子的指尖位置/速度反馈。动作只做一次原生归一化。
3. 分两轮搜索共 18 组闭合偏置与反馈增益。第一轮标称成功候选在扰动初态下饱和率超限，因此没有直接采纳。第二轮选择闭合偏置 0.10 rad、指尖反馈增益 60 N/m，时间伸缩参数 4；不是简单增加夹紧力度。
4. 初始化后只通过 `env.step(action)` 执行，杯子自由运动。每个物理子步审计手杯距离、穿透、关节限位和数值有效性；末尾一秒检查杯子相对手掌的位置漂移。
5. 保存 `geom_margin`、`geom_gap` 到 rollout 的 `physics_model` 字段，查看器显式恢复。原有 DexMV 模型快照不保存这些参数，不能直接用旧加载器重放或训练。
6. 固定参数后测试种子 0–5，杯子初始 XY 各在 ±2 mm 内扰动；进一步将 2 ms 物理步长减半至 1 ms，分别验证保存动作和闭环控制。

## 标准与结果

这些是本项目声明的工程验收阈值，不是通用物理学定律。接触力阈值为单个接触法向力 >0.01 N；正间距和穿透不能互相抵消。

| 检查项 | 门槛 | 原步长 6 个初态结果 |
| --- | --- | --- |
| 承力正间距 | ≤0.5 mm | 最大约 0.200 mm |
| 手杯穿透 | ≤1 mm，全子步检查 | 最大 0.675 mm |
| 连续接触抬杯 | ≥1 s，杯底 >15 mm | 6.28–6.47 s |
| 最后一秒杯底高度 | 始终 >15 mm | 最终约 84.9–86.2 mm |
| 最后一秒接触 | 至少两处承力；拇指、食指接触比例各 ≥80% | 拇指、食指、无名指均持续承力 |
| 动作饱和占比 | <1% | 0.259–0.374% |
| 关节限位误差 | ≤0.02 原生关节单位（转动 rad、平移 m） | 0 |
| 末尾杯子相对手掌平移漂移 | ≤5 mm | 最大 0.022 mm |
| 最终目标位置距离 | <100 mm，仅初级抓取门槛 | 约 55 mm，不是精确放置 |
| 数值有效性 | qpos/qvel 无 NaN/Inf | 通过 |
| 同步长保存动作重放 | 最大观测误差 <1e-8 | 0 |

关节门槛沿用模型原生单位，当前没有测得越界；它不等价于硬件速度、加速度或电机热限制。末尾漂移是杯子相对手掌的平移代理量，不是所有接触点的切向微滑移证明。上述检查没有认证整个动作的全网格自碰撞或全部可视网格有符号交叠。

半步长下保存动作与闭环控制均通过同一近表面门槛，最大穿透分别约 0.316/0.316 mm。没有要求不同步长下轨迹逐点完全一致。

标称末态额外计算了承力点附近显示网格距离：食指约 0.768 mm、无名指 0.468 mm、拇指 0.272 mm。它是接触邻域最近可视点距离，不是完整网格最小有符号距离；证明了该邻域不再存在原来的 2–3 mm 悬空承力，但不代表网格几何精确无误。

## 查看与复现

在 Ubuntu 桌面终端打开新物理版本：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/36_view_surface_grasp_gpu.sh
```

默认循环，`Ctrl+C` 退出。本轮 GPU 窗口已完整运行 1184 步并正常退出。旧入口 28、31 仍保留旧结果，不能用它们的默认播放代替本版本。

以下命令在已设置运行库的 `dexmv` 环境、项目根目录执行，输出必须使用新目录：

```bash
python scripts/33_optimize_surface_grasp.py \
  --output data/processed/seq_dexycb_001/surface_reproduction \
  --closures 0.04 0.07 0.1 --gains 40 60 100
python scripts/34_audit_visible_contact.py \
  --rollout data/processed/seq_dexycb_001/surface_reproduction/best/diagnostic_rollout.pkl \
  --output data/processed/seq_dexycb_001/surface_reproduction/visual_contact.json
python scripts/35_check_surface_timestep.py \
  --result data/processed/seq_dexycb_001/surface_reproduction
python -m unittest discover -s tests -q
```

## 产物与下一道门槛

结果位于 `data/processed/seq_dexycb_001/surface_grasp_v2/`：`search.json`、`admission.json`、`best/`、`replay/`、`seed_1/` 至 `seed_5/`、`visual_contact.json`、`timestep_check.json` 和 `review/`。旧资产 SHA-256 检查通过。

当前中指和小指未持续承力，末尾平均指尖误差约 20.5 mm，严格视频忠实度未通过。下一步应在不放宽近表面标准的前提下优化各指接触位置与参与程度，再考虑生成训练示范。训练环境也必须应用相同接触设置，不能把带自定义接触参数的诊断 rollout 直接送入未适配的旧训练链路。
