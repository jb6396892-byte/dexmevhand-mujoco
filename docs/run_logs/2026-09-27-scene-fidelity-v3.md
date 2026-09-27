# v3：完整视频示范准入与短训练

## 结论

视频专家控制器已通过本项目现有的完整场景物理与视频忠实度门槛，能够进入训练。
20 次 DAPG 短训完成，但神经网络策略仅 2/11 个评估初态抬杯并保持，0/11 通过完整任务验收。
示范可执行、训练链路可运行、策略学会任务是三个不同结论。

## 优化方法与证据

1. 增加逐碰撞几何对的峰值穿透、发生时间及逐执行器饱和次数记录。
2. 将已保存的逐指修正与时间伸缩参数 5 组合，首次达到 6/6 个初态完整场景物理通过；末段指尖误差仍为 19.28 mm。
3. 提前 0.15 秒的源视频时间开始闭合与关节修正，改善抓住杯子后才抬手的时序。10/10 个初态物理通过，标称末段误差降至 15.38 mm。
4. 在接近阶段增加物体相对指尖的阻抗反馈，作用于根部、手腕和手指，并在抓取阶段平滑退出。增益 25 N/m，末段反馈保留 75 N/m。杯子初始化后始终由动力学自由运动。
5. 更强拇指反馈导致执行器饱和；接近增益 75/150 或根部增益 150/300 会导致部分初态穿透超过 1 mm，未选用。
6. 增益 45 及单独根部增益 75 的候选虽可通过部分扩展检查，但物理余量更小；正式导出沿用增益 25 的保守候选。

没有调整杯子质量、摩擦、碰撞网格或原始数据，没有逐帧重写杯子位姿，也没有放宽验收阈值。
采用之前已验证的 0.2 mm 接触边距，并确保训练与回放恢复同一接触参数。

## 正式示范结果

正式候选：`data/processed/seq_dexycb_001/scene_fidelity_v3/approach_feedback/`。
导出准入：`data/processed/seq_dexycb_001/scene_fidelity_v3/verified_export/admission.json`。

| 指标 | 标称初态结果 | 门槛 |
| --- | --- | --- |
| 全程平均指尖误差 | 17.74 mm | <20 mm |
| 末段平均指尖误差 | 14.01 mm | <15 mm |
| 拇指末段误差 | 19.49 mm | 每指 <25 mm |
| 稳定抬杯持续时间 | 8.35 s | >=1 s |
| 末段最低杯底高度 | 137.79 mm | >15 mm |
| 末段相对手掌滑移 | 0.0357 mm | <=5 mm |
| 最大承力正间距 | 0.200 mm | <=0.5 mm |
| 动作饱和率 | 0.299% | <1% |
| 保存动作重放最大观测误差 | 0 | <1e-8 |

- 种子 0–9 的候选检查均通过；其中 0、2、5 用于参数选择，其他旧种子也曾在前轮观察，不能全部视作全新测试。
- 冻结参数后，独立种子 10–19 全部通过；加上 6 个正式示范初态，最坏场景穿透为 0.690 mm。
- 物理步长从 2 ms 减半到 1 ms 后，保存动作重放和闭环控制均通过物理及忠实度检查。
- 末段拇指、食指、中指、无名指持续接触比例均为 100%，小指没有持续受力。
- 初始阶段平均指尖误差仍约 31.35 mm。当前门槛是整段统计通过，不等于每帧像素级复刻。
- 泛化范围仍只是同一条视频、同一个杯子、XY 约 +/-2 mm 初态扰动；不是跨视频、跨杯子或实机成功率。

正式示范：`data/demonstrations/relocate-mug-video-faithful-v3.pkl`，6 条，每条 1417 步。
SHA-256：`c1a7bb96952bf149e7bfb6975ec518db37eee7a18ccdf56de021702f186c9e80`。
旧基线受保护文件哈希保持一致。

## 短训及策略结果

配置：`configs/dapg-mug-video-faithful-v3-smoke.yaml`。
训练目录：`training_log/dapg_relocate-mug-0.8_relocate-mug-video-faithful-v3_0.1_6_video_faithful_v3_smoke20_seed200/`。
策略：上述目录的 `iterations/best_policy.pickle`。

- 训练入口恢复示范的 `geom_margin/geom_gap`，支持真实轨迹长度；复位与采样重放误差均为 0。
- GPU 检查通过，使用 RTX 4060 Laptop GPU。上游价值网络使用 GPU，MuJoCo 采样与策略更新仍在 CPU。
- 5 轮 BC 初始化加 20 次 DAPG 更新完成，日志 20 行，无 NaN；记录的确定性评估回报由 -36.40 到 -13.54。
- 独立纯策略评估没有专家纠偏：种子 1、5 可连续抬杯约 11 秒，末态目标距离分别 111.3、106.0 mm；其余 9 组没有有效抬杯。
- 因目标距离、部分场景穿透和忠实度不合格，完整验收 0/11。未启动 2000 次完整训练。
- 评估报告及回放：`data/processed/seq_dexycb_001/scene_fidelity_v3/policy_evaluation/`。

上游 `mjrl/utils/train_agent.py` 有硬编码向 `~/Desktop/trpo_params/trpo_iter_N_params.npy` 导出的旧逻辑，本次训练已触发，无法确认此前同名文件是否被覆盖。现已在本项目入口增加作用域内的路径重定向，后续写入各训练任务的 `parameter_exports/`，不修改上游源码。原有主 checkpoint 和受保护示范未覆盖。

## 可复现命令

最终 33 项单元测试通过，`git diff --check` 通过。新增测试覆盖接触参数恢复及旧桌面参数导出的隔离。

以下 Python 命令在配置好运行库的 `dexmv` 环境执行，工作目录为项目根目录。输出目录须使用新名称，脚本拒绝覆盖。

```bash
python scripts/41_optimize_grasp_timing.py \
  --candidate data/processed/seq_dexycb_001/scene_fidelity_v3/baseline_audit/admission.json \
  --output NEW_SEARCH_DIR --leads .15 --thumb-weights 1 --approach-gains 25 --quota-stop 85
python scripts/35_check_surface_timestep.py --result NEW_SEARCH_DIR
python scripts/42_export_video_demonstrations.py \
  --candidate NEW_SEARCH_DIR/admission.json --output NEW_EXPORT_DIR --demo NEW_DEMO.pkl
```

直接在 Ubuntu 桌面终端查看已通过的示范：

```bash
cd /home/smgbro/mujoconew/GITHUB
bash scripts/43_view_video_faithful_v3_gpu.sh
```

这是保存动作经 `env.step(action)` 的物理示范回放，不是神经网络策略。
GLX 窗口已完成一次 1417 步回放并正常退出，观测误差 0。
GPU EGL 离屏仍初始化失败；软件离屏成功，8 张源视频/仿真对照图保存在 `scene_fidelity_v3/visual_review_cpu/`。

训练预检命令：

```bash
bash scripts/27_train_verified_smoke.sh \
  --cfg configs/dapg-mug-video-faithful-v3-smoke.yaml \
  --admission data/processed/seq_dexycb_001/scene_fidelity_v3/verified_export/admission.json
```

本轮短训已完成，不要直接重复加 `--train`；如要新实验，先修改配置中的任务名和示范路径。
策略验收入口为 `scripts/44_evaluate_video_policy.py`，必须提供 policy、candidate、独立 output。

## 后续可执行任务

1. 对保存策略计算示范动作误差和闭环首次偏离帧；用同样种子比较更充分的 BC 初始化，不能只看离线 MSE。
2. 收集偏离示范状态上的专家纠偏数据，验证 DAgger 或残差策略；明确观测是否需要速度、技能阶段等信息。
3. 策略稳定抬杯后重点修正搬运目标误差，再做新的 20 次短训；保留专家与策略两套独立验收。
4. 扩展独立视频、杯子初始朝向和任务位置；小指接触与前段可见性仍需进一步核验。

本轮持续读取本地 300 分钟额度记录，85% 为停止新搜索阈值；最终记录达到 83% 时停止新实验并收尾。新搜索每个候选落盘，旧候选和原始大数据均保留；大数据未下载或迁移。
