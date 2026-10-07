# 无平台 Adroit 导航与两视频抓取适配

2026-10-07。当前方向按用户最新要求改为**直接移动 Adroit，不添加 XYZ 机构**。旧 gantry 源码和证据作为历史保留，不再作为新方案的运行入口。旧 v6 Qt 默认入口、模型权重和冻结报告未替换。

## 交付状态

| 工作 | 本轮结果 | 尚未完成 |
| --- | --- | --- |
| 手参考中心与空手导航 | 原有 30 执行器；六个随机物品布局通过；四路点绕墙通过；堵路和三个非法目标拒绝 | 全桌抓取成功率统计 |
| 两视频局部策略适配 | 第一视频杯子 X=+0.20 m、第二视频 X=-0.20 m 均抓住并抬杯 | 多区域、拥挤场景，以及从统一起点连续交接到预抓姿态 |
| 带杯运动 | 第一视频约 42 cm 搬运通过；第二视频约 45 cm 到位保持，但动态门槛失败 | 第二视频交接加速度；带杯真正绕障 |

这些是开发组件测试，不是新的独立留出集，也不是整桌 >80% 的验收。没有启动训练。新 Qt 模式仍留在后续 F4。

最终软件测试 270/270 通过；旧 v6 四干扰物完整抓取搬运回归通过。

## 运行

在 Ubuntu 桌面终端执行：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/180_launch_adroit_navigation.sh --seed 9 --count 4
# 有阻挡墙的空手导航，结束后窗口保持打开：
bash scripts/180_launch_adroit_navigation.sh --wall --count 0
```

目标是**手掌参考中心**的世界 XYZ，单位米，不是杯子目标：

```bash
bash scripts/180_launch_adroit_navigation.sh --goal 0.25 0.15 0.38
```

该窗口演示空手导航，不自动抓杯。8 秒窗口检查渲染了 401 帧，无 `Missing GL version`。离屏证据另做像素非空检查。

局部学习抓取加物理搬运，输出目录必须不存在：

```bash
bash scripts/137_tabletop_gpu.sh scripts/181_check_free_hand_local.py \
  --video first --shift 0.2 0 --carry-goal -0.2 0.1 0.20 \
  --output /media/smgbro/shared/visual_grasp/adroit-navigation-v1/my-first
```

第二视频使用 `--video second --shift -0.2 0 --carry-goal 0.2 0.1 0.20`。当前它会保存实际轨迹和失败报告并非零退出，不能只看截图判断通过。`--transit` 检查统一起点到预抓姿态的连续交接，目前两视频均被安全余量拒绝。

## 方法

1. **参考中心标定。** 用 `C_palm0` 中心，新增导航模型中对应 `S_nav`，掌内坐标 `[-0.008, 0, 0.038]` m。读取实际平移关节 Jacobian 得到世界轴映射，而不是把 ARTx/ARTy/ARTz 名字直接当世界 XYZ。空手模型固定零点，不随杯位置改变。
2. **局部坐标适配。** 初始化时同时平移手的安装参考、杯子和来源位姿，保留原关节坐标、30 维归一化动作及 139 维输入。相对位置特征在共同平移下不变；新增单元测试验证这一点。初始化安装变换不是“已经完成导航”，因此独立标注。其他物体不随杯子平移。
3. **几何规划。** 前臂、手腕、手掌、各手指分别构成保守 AABB；载杯时再加入真实杯模型。用障碍物膨胀构造配置空间，SciPy Dijkstra 搜索 25 mm 网格，所有边及简化线段都连续检查；不能仅检查路径节点。
4. **执行控制。** 五次静止到静止轨迹限制速度、加速度和加加速度。世界目标经标定映射为原平移关节的米制目标；位置伺服加重力补偿，再由 MuJoCo 积分。旧学习策略仍只缩放一次归一化动作。
5. **负载和交接。** 抬杯后保持 3 秒，记录一次杯子位置及质量，用 `Jp.T @ F + Jr.T @ torque` 补偿负载重力，通过手的执行器发力，不给杯子加外力。保留学习策略的手指控制，根部控制器用 2 秒五次曲线混合力输出。搬运时允许手指弹性变化 ≤0.05 rad，腕部仍 ≤0.025 rad；实际几何、接触和穿透同时审计。

带杯部分是**学习抓取 + 模型控制搬运**，不是新训练出的全桌策略。杯子保持原 freejoint；没有执行期 qpos 写入、weld、吸附或额外物体助力。障碍几何和初始杯位姿已知，不能表述为 RGB-D 端到端成功。

## 验收与阻碍

- 空手位置误差 <3 mm，跟踪误差 <15 mm，规划余量 25 mm、执行余量 8 mm，无手与环境碰撞。
- 搬运命令速度 ≤[0.04,0.04,0.03] m/s，加速度 ≤[0.06,0.06,0.05] m/s²；实测判定分别有 0.005 m/s、0.02 m/s² 数值裕量。记录全部峰值，不只检查平滑的目标曲线。
- 杯子误差 ≤20 mm；末秒持续拇指加两指对握；穿透 ≤1 mm；接触丢失不可持续 0.15 秒；不要求松手放下。
- 第一视频最终误差约 0.75 mm，穿透峰值约 0.61 mm，末秒支持率 100%。这是从局部预抓初态开始的单个开发用例。
- 第二视频最终误差约 0.48 mm、末秒支持率 100%，但 Z 加速度峰值约 0.292 m/s²，高于 0.07 m/s² 验收值，故总体失败；不能称为已通过。
- 预抓取姿态的部分保守包围盒距桌面不足 25 mm，甚至跨过桌沿高度。规划器拒绝近桌面交接，不代表 MuJoCo 已测得同等深度的实体穿透。下一步需要碰撞几何细化和专用低速接近段，不能简单去掉桌面碰撞或清零安全余量。
- 空手绕墙和无障碍带杯分开验证；尚未验证带杯绕障，也未做两视频九位置覆盖。根据用户“困难可以先跳过”的授权，本轮保留这些阻碍，不继续长时间调参。

## 文件与证据

- 配置：`configs/adroit-navigation-v1.json`；导航框架的 `grasp_enabled=false` 表示空手入口不抓取，脚本 181 是独立扩展实验。
- 实现：`whole_table/hand_scene.py`、`navigation.py`、`navigation_runner.py`、`local_adapter.py`、`loaded_motion.py`。
- 完整实验：`/media/smgbro/shared/visual_grasp/adroit-navigation-v1/`，不混入语言模型 `lora` 目录。
- [量化结果与源码哈希](presentation/adroit_navigation/results.json)；[答辩提纲和三张截图](presentation/adroit_navigation/README.md)。

## 开源依据

- [Farama Adroit Relocate](https://robotics.farama.org/main/envs/adroit_hand/adroit_relocate/)：现有 Adroit 平移及手部动作结构；没有引入新机器人资产。
- [Lozano-Perez, Spatial Planning (1983)](https://lis.csail.mit.edu/pubs/tlp/spatial-planning.pdf)：用机器人形状构造配置空间障碍的思路。
- [SciPy Dijkstra](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html)：直接复用已有稀疏图搜索，不引入 ROS2 或升级旧环境。
- [MuJoCo 动力学说明](https://mujoco.readthedocs.io/en/stable/computation.html)：偏置力与 Jacobian 转置映射。本项目将负载重力通过手的执行器补偿；杯子仍由接触力运动。

本轮是已有方法的工程组合，不将坐标平移、Dijkstra 或重力补偿本身作为论文创新。
