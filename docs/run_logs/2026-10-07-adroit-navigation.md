# 无平台导航、两视频适配与带杯开发记录

## 需求和范围

用户取消 XYZ 模型，改用手掌参考中心直接导航，并要求继续两视频局部抓取适配、避障与带杯运动；困难可先跳过。保留旧工作成果，本轮不重训、不接 Qt 新模式、不做独立整桌成功率宣称。

## 执行过程

1. 检查原 Adroit 模型：已有 ARTx/ARTy/ARTz，保留 30 执行器，无需新增平台。标定真实轴方向和掌心，而非假定关节名等于世界轴。
2. 复用全桌合法采样和真实 YCB 资产，实现分部包围盒配置空间、Dijkstra、连续边有效性和五次限速轨迹。
3. 空手六个开发布局完成，绕墙四路点完成；全高堵路与非法目标在仿真推进前拒绝。原生窗口 8 秒 401 帧。
4. 局部适配初次失败是 `mj_setConst` 重置数据，修正为初始化时保留并恢复状态。两视频各一例在 X 平移 20 cm 后完成局部抬杯。
5. 预抓导航被桌面余量拒绝，记录具体障碍与包围盒。没有移除桌面碰撞、重新抽简单布局或跳过校验。
6. 初版带杯控制遗漏载荷补偿，引起腕部偏差和约 5 mm 稳态位置误差。参考 MuJoCo 文档，通过执行器施加 Jacobian 转置负载补偿；接触力仍决定杯子运动。
7. 区分腕姿态保持与手指弹性保持；腕部门槛 0.025 rad 不变，手指允许 0.05 rad，并同时监测对握、丢失、1 mm 穿透和实际几何。此变更公开记录，不把旧失败改判。
8. 新控制器硬切换曾有约 4.27 m/s² 峰值。采用 2 秒五次力混合及 3 秒交接前稳定。第一视频通过；第二视频剩余约 0.292 m/s²，仍超过 Z 验收 0.07 m/s²，保留失败。

## 复现命令

工作目录 `/home/smgbro/mujoconew/GITHUB`；实验前缀 `/media/smgbro/shared/visual_grasp/adroit-navigation-v1/`，所有输出目录必须是新目录。

```bash
bash scripts/137_tabletop_gpu.sh scripts/178_check_adroit_navigation.py --output <new-navigation-directory>
bash scripts/180_launch_adroit_navigation.sh --seconds 8 --count 4
bash scripts/137_tabletop_gpu.sh scripts/181_check_free_hand_local.py --video first --shift .2 0 --carry-goal -.2 .1 .20 --output <new-first-directory>
bash scripts/137_tabletop_gpu.sh scripts/181_check_free_hand_local.py --video second --shift -.2 0 --carry-goal .2 .1 .20 --output <new-second-directory>
bash scripts/137_tabletop_gpu.sh -m unittest discover -s tests
bash scripts/137_tabletop_gpu.sh scripts/182_publish_adroit_navigation.py
```

`--transit` 另验统一起点交接。目前拒绝为 `goal_in_collision_margin`。第二视频带杯总体失败时脚本返回非零值；不要把 `completed=[reach,grasp,lift]` 当作完整搬运通过。

## 测试环境排错

系统 Python 3.12 没有 NumPy，第一次裸跑测试失败；直接使用 dexmv 但不带项目环境时又遇到旧 mujoco-py 构建锁只读。最终统一用现有 `137_tabletop_gpu.sh`，没有安装新库、没有升级环境。原 268 项通过后，新增平移特征不变性与分部包络两项测试，再做最终回归。

最终回归 **270/270 通过**。另运行旧 v6 第二视频、seed=30、四件干扰物的完整任务：reach/grasp/lift/transport 全部通过，穿透峰值 0.718 mm。新入口最终第一视频退出码 0、第二视频退出码 1，后者明确为加速度验收失败。原生窗口自动退出正常，未出现 GL 错误。

## 产物和后续

源码和小型报告进入 Git；实验全量轨迹仍在共享盘。三张截图分别标注空手绕障、第一视频成功、第二视频未通过动态门槛。结果与源文件哈希由脚本 182 生成。

后续先修低速近桌面交接和第二视频控制器状态衔接，再做带杯障碍用例、双视频多区域覆盖。完成后才接 Qt 新模式与冻结留出测试；旧 v6 的成功率不能移植到整桌模式。
