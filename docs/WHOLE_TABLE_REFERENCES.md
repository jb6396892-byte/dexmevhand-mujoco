# 整桌移动平台：开源模型与论文参考

检索日期：2026-10-06。以下均核对原作者仓库、项目页或官方文档；“拟采用”不代表已经安装或复现。没有找到可直接替换本项目、同时包含 XYZ 平台 + Adroit + 两视频抓法的现成兼容模型或权重。本项目仍复用原 Adroit 和 YCB 杯子资产。

## 优先级和实际用途

| 优先级 | 开源项目 / 论文 | 解决什么问题 | 如何用在本项目 | 不能直接照搬的部分 |
| --- | --- | --- | --- | --- |
| P0 | [MuJoCo MJCF 文档](https://mujoco.readthedocs.io/en/2.1.2/XMLreference.html) | 三轴机构、惯性、执行器、关节行程 | 建三个 slide joint 和有界执行器，保留原手部模型 | 文档为 2.1.2，实际是 2.0；新增 XML 必须在本机编译验证 |
| P0 | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 可参考的开源机器人模型组织与碰撞资产 | 参考独立 robot/scene XML、关节命名、质量/惯性配置方式 | 不是现成 Adroit 龙门架；各模型最低版本和许可证分别核对，不换成另一只手 |
| P0 | [OMPL](https://ompl.kavrakilab.org/) / [代码](https://github.com/ompl/ompl) | 障碍物散布全桌后寻找可达路径 | 优先 RRTConnect，接 MuJoCo 规划副本作有效性回调 | OMPL 核心不替代场景碰撞检查，也不保证执行动力学可行 |
| P1 | [Ruckig](https://github.com/pantor/ruckig) / [RSS 2021 论文](https://arxiv.org/abs/2105.04830) | 平台启动/刹车过猛，带杯运动易滑落 | 局部 state-to-state 轨迹满足速度、加速度、jerk 限制 | 不负责避障；Community 中间 waypoint 功能可能调用云，不作为本地依赖使用 |
| P1 | [MimicGen，CoRL 2023](https://mimicgen.github.io/) / [代码](https://github.com/NVlabs/mimicgen) | 演示只覆盖原杯位，如何迁移到新位姿 | 借鉴 object-centric 子任务与参考段变换，再重新执行验证 | 不是把成功示范平移就获得成功；需要本项目环境接口，输出失败仍保留 |
| P1 | [DexTrack，ICLR 2025](https://arxiv.org/abs/2502.09614) / [代码](https://github.com/Meowuu7/DexTrack) | 新位置或扰动下保持手部参考跟踪 | 借鉴参考条件残差、单轨迹到多轨迹的分层验证 | 仓库用 IsaacGym，支持 Allegro / LEAP+Franka；权重不能直接用于 Adroit/MuJoCo |
| P2 | [DexMachina 论文](https://arxiv.org/abs/2505.24853) / [代码](https://github.com/MandiZhao/dexmachina) | 形态差异下接触与物体运动的一致性 | 只在 F2 接触迁移卡住时参考接触映射/课程思想 | 方法含逐渐衰减的虚拟物体控制器；不能引入本项目无助力验收，不能以辅助物体运动算抓取成功 |

## 为什么这个组合

**规划与学习分工，而不是先扩大网络。** 平台先解决大范围可达和避障，局部模型解决闭合、接触和保持。先验证纯坐标/机构适配是否足够，再判断是否需要纠偏学习。这是本项目的工程方案选择，不是以上论文已证明本项目 >80%。

OMPL 接入时必须同时实现状态和运动段有效性，不能只验证端点。参考 [官方状态有效性说明](https://ompl.kavrakilab.org/stateValidation.html)。检查对象包括滑台、手掌、手指和抓取后的杯子；粗包围盒可用于加速，但最终验收需用实际碰撞几何与足够细的扫掠检查。

Ruckig 首版可对无碰撞折线路径逐段进行本地状态到状态时间处理，在拐点停车；会慢一些，但不依赖商业/云端中间点服务。处理后的实际轨迹还需碰撞复核，不能认为平滑后自动安全。[官方功能边界](https://github.com/pantor/ruckig#intermediate-waypoints)

MimicGen 的环境接口要求提供子任务相关物体姿态等信息，适合本轮“位置抓取前已知”的设定，但现有 DexMV 控制维度、抓法和物理执行都要自己适配。[官方接入教程](https://github.com/NVlabs/mimicgen/blob/main/docs/tutorials/getting_started.md)

## 依赖与许可证处理

- 本轮只查阅资料，不下载模型、不安装 OMPL/Ruckig、不克隆大型仓库。
- Menagerie 最低 MuJoCo 版本按每个模型 README 判断，模型许可也按模型目录核对；不能把仓库公共代码许可当作全部 mesh 许可。
- Ruckig 仓库标为 MIT，但 Community / Pro 的功能范围不同。只计划用本地开源功能。
- OMPL、MimicGen、DexTrack、DexMachina 在真正复制源码或再分发资产前固定 commit 并核对对应 LICENSE；本任务书不替代许可证审查。
- 当前 Python 3.7 / mujoco-py 2.0 不能假设兼容最新轮子。F1 先沿用旧引擎；F3 再评估隔离规划环境，不直接升级 dexmv。

## 预期对照实验

1. 固定参考与现有学习残差，在相同平台和同一整桌布局上比较，明确专家数据成本。
2. 仅坐标迁移 vs 加入路径规划，比较碰撞和可达性，不归因于新抓取学习。
3. 平台直接位置阶跃 vs 限速度/加速度轨迹，只在安全开发集比较，不故意执行已知碰撞路径。
4. 如确需训练，报告训练预算、新增示范数量和纠偏帧数；不能以更低跟踪误差代替更高完整任务成功率。

暂不选机械臂 IK、具身大模型或新语言模型训练；它们不是当前三轴平台方案的必要依赖。
