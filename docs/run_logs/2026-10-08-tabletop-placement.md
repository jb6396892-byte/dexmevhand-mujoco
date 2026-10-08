# 桌面放置与返回起点开发记录

## 任务范围

新增连续“搬运到目标 XY → 落桌 → 松手 → 撤离 → 验证站稳 → 返回手的起始中心”。保持旧搬运模式和权重，不训练新策略。原 18/20 搬运结果不能当作放置验收。

## 当前开发证据

输出根目录：`/media/smgbro/shared/visual_grasp/tabletop-placement-v1/`。

1. `diagnostic-first`：第一抓法搬运后杯子倾角约 5.38 度。
2. `dev-first-01`：直接下降时掌部先碰桌，杯底还距桌面 2.41 mm；停止，不删除碰撞。
3. `dev-second-01`：第二抓法搬运结束杯子倾角约 13.61 度，超过目标直立要求。
4. 小角度腕部纠姿使用独立 FK 副本：搜索满足杯轴方向和掌指桌面净空的姿态，再通过真实执行器平滑执行。第一抓法调整后倾角约 6.93 度，第二约 8.07 度。
5. `dev-first-02` / `dev-second-02`：直接恢复原始手型，分别导致小指碰桌和杯子被推偏。弃用该释放方式。
6. 释放改为实际接触点外法向分离 IK，固定腕关节，限制手指关节与执行器共同可达范围。只在 FK 副本写关节状态；实际手由控制输入驱动，杯子始终自由动力学。
7. `dev-first-04`、`dev-second-03`：空旷开发场景连续完成放置和返航。最终 XY 误差约 4.11 / 5.80 mm，最终杯子倾角均小于 0.02 度，无手杯接触，桌面向上承重约为杯重，返航中心误差均小于 0.004 mm。

上述两个成功案例是在增加全放置阶段运动峰值检查之前的开发记录，不等同于最终冻结验收。新的多障碍开发测试会记录每次预演、全部失败和运动限值；不能选择性只保留成功结果。

## 已实现的软件部分

- `placement_metrics.py`：杯轴、杯底、接触力、速度、独立稳定窗口。接触力从接触坐标系转到世界坐标，并按 geom1/geom2 正确处理方向。
- `placement.py`：纠姿、下降、桌面承重确认、前馈卸载、接触分离 IK、撤离、站稳确认、返航；阶段失败报告与截图。
- `task.py` / `loaded_motion.py`：同一物理场景内连续交接，完整候选预演后最多一次实际执行。
- `placement_planning.py` / `196_plan_placement.py`：明确的规则指令白名单，不修改冻结 LoRA 模型和原指令准入，不把新放置功能说成语言模型学会的新技能。
- Qt 新完成模式、下降速度、释放时间和撤离距离；显示放置状态、倾角、杯底间隙和支撑力。
- 第一轮软件回归 293 项通过。物理和 Qt 最终验收尚在进行。

## 触桌、撤离与返航修复

- 杯底距桌面 5 mm 内将下降速度降至 0.5 mm/s；3 mm 内启用接触感知执行器加速度限制。使用有限差分线性化、限幅和回溯线搜索，只修改手的执行器输入，不修改杯子状态。未使用线搜索的早期版本曾产生修正过冲，失败保存在 `dev-second-06`，已弃用。
- 释放前先确认桌面承重并平滑卸载。释放采用接触法向 IK 与 5 s 五次平滑插值，释放后必须无手杯接触。
- 原竖直撤离会碰杯沿，改为沿“杯子指向掌心”的水平分量后退 60 mm，同时升高 120 mm。撤离时保留释放末态的手指控制，避免控制器切换造成重新闭合。
- 撤离并站稳后，再在安全距离外切换为空手定姿伺服，重算包络，调用原避障规划器返回记录的手中心。这里只恢复中心位置，不要求手指与腕姿恢复成初始状态。
- `dev-first-11`、`dev-first-13`、`dev-second-07` 为带运动门槛的完整成功开发案例。第二抓法 `dev-second-07` 最终 XY 误差约 5.67 mm，放置阶段峰值加速度约 0.496 m/s²，峰值穿透约 0.465 mm。它们不计入最终留出集。
- `qt-controls-01`：真实 Qt 停止、锁定、非法指令拒绝全部通过。停止响应约 0.635 s，无残留工作进程。
- 早期 `development-6201` 与 `qt-development` 在控制器修改期间中止，保留部分记录；不是完成的独立验收，不计成功率。
- 放置目标随机采样只在执行前排除杯体 footprint 越桌和落点被其他物体占用；不依据控制结果重采样。几何合法但抓取、下降或返航失败仍算失败。

## 数值一致性与 Qt 回归

1. `dev-second-09` 暴露预测加速度 0.203 m/s²、实际一步 0.515 m/s² 的差别。旧求解器尚未完全收敛时，额外 `mj_forward` 会改变 warmstart；有限差分和真实步不再是同一初值。修复为保存、恢复 `qacc_warmstart`，所有候选控制和实际积分使用同一个数值求解初值。没有修改关节位置、杯子状态、质量、摩擦或求解器类型。
2. `dev-second-10` 在同一失败落点通过，峰值加速度约 0.347 m/s²；`dev-first-15` 在此前失败落点也通过，峰值约 0.350 m/s²。
3. `qt-complete-01` 出现空白帧。旧 Cython 渲染器不会自动为每次绘制恢复上下文，析构时还会在当前上下文释放缓冲。现在每个仿真复用一个自有 GLFW 上下文，绘制和释放前显式切换，避免跨预演的缓冲冲突。
4. `qt-complete-02` 第一例预演成功、实时失败，原因是遥测额外调用 `sim.forward()`，改变了物理计算路径。删除该调用，渲染上下文统一在预演和实际执行的初始化阶段建立。
5. `qt-complete-03` 两项完整 Qt 任务均通过，分别收到 1994 / 1543 张非空白变化帧；两例最终杯位和手关节位置与对应预演结果逐数值相同，最大差为 0。界面响应检查通过，无遗留工作进程。
6. `scripts/197_check_render_context.py` 实际交替渲染两个含自由刚体的场景，并销毁旧场景再绘制；5 次像素标准差均约 90.68，非空白检查和上下文复用通过。
7. 软件全量回归 **300/300** 通过。`development-6251` 旧开发批为 1/4，`development-6261` 为 2/3；均在开发修改期间执行、源码哈希变化，不是独立验收。失败保留，不能用后来的成功覆盖这些记录。

独立验收使用新种子 6401-6420，配置、源码和权重哈希在运行前冻结；结果另行归档，不把上述开发例计入。

## 方法来源

- [MuJoCo 官方计算文档](https://mujoco.readthedocs.io/en/2.1.2/computation.html)：接触法向是接触系第一轴，力由第一个几何体指向第二个几何体。这里用于桌面承重审计，不把夹持力当作承重。
- [MoveIt Task Constructor 放置教程](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html)：借鉴放置、张手、撤离、返回的阶段组织；不采用物体绑定代替抓握。
- [Dexterous Manipulation of Unknown Objects Using Virtual Contact Points](https://www.mdpi.com/2218-6581/8/4/86)：参考接触点目标与逆运动学的分层思路。本项目使用当前仿真接触法向生成小幅分离目标，并非复现该论文的完整算法。
- [MuJoCo warmstart 文档](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#warmstarts)：未完全收敛时，求解初值会影响结果。这里据此固定预测与实际步的数值初值，不用增加闭合力掩盖差异。
- [mujoco-py 上游渲染实现](https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjrendercontext.pyx)：对照本机旧版本检查上下文与缓冲生命周期，修复预演、截图和 Qt 串流之间的上下文切换。

## 复现入口

```bash
bash scripts/137_tabletop_gpu.sh scripts/195_run_placement.py --direct --video first --output /media/smgbro/shared/visual_grasp/tabletop-placement-v1/new-development
bash scripts/137_tabletop_gpu.sh scripts/192_evaluate_navigation_task.py --place --config configs/tabletop-placement-v1.json --start-seed 6201 --count 6 --label development --output /media/smgbro/shared/visual_grasp/tabletop-placement-v1/new-development-batch
python3 scripts/193_check_navigation_qt.py --place --cases first second stop locked rejected --output /media/smgbro/shared/visual_grasp/tabletop-placement-v1/new-qt-check
```

输出路径必须使用新目录，不覆盖已有证据。最终统计、截图和未解决问题将在验收完成后补充。
