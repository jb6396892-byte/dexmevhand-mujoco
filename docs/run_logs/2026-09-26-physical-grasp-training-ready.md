# 连续动作与闭合保持修复：已达到短训练准入条件

日期：2026-09-26。本记录接续 geometry-physics 诊断记录；旧记录中的失败结果仍保留。

## 结论与边界

已经得到只通过 `env.step(action)` 控制、杯子自由运动的成功物理示范，并通过独立动作重放和训练采样器预检。
可以开始**示范附近初态课程的 20 次 DAPG 短训练**，本轮按“直到可以开始训练为止”的要求，未启动训练。

这不是已训练神经网络的新成绩。当前成功的是从 DexYCB 姿态构建、经仿真搜索验证的示范控制器。
6 条示范来自同一真实序列的初态扰动，不是 6 个独立真实视频。范围是右手/YCB mug/scale=0.8，杯子 XY 位置仅扰动 ±2 mm；不能外推到默认随机复位、其他杯子或完整长任务。

## 修复了什么

1. 不再直接执行原逐帧重定向的 PCHIP 全轨迹。保留源初始手姿态、源帧 36 的抓持参考姿态和源序列终点，重建“静止、闭合、抬升搬运、保持”连续轨迹。
2. 用五次平滑函数连接各阶段，使参考轨迹阶段边界速度、加速度为零；从零初速度开始。
3. 根关节位移通过实际平移雅可比从世界坐标映射。Adroit 根 body 自带旋转，不能把关节名 ARTz 当成世界 z。
4. 使用原模型的位置执行器增益、仿射偏置和阻尼，加入重力/偏置与参考速度阻尼补偿；不采用此前额外高增益 PD。动作只归一化一次。
5. 在自由杯子动力学中搜索手指闭合、拇指对掌和毫米级手部目标调整。调整的是手的控制目标，不是逐帧移动杯子。原数据和旧示范未改。
6. 把正式训练回合从上游默认 400 步改为独立课程入口的 1000 步，复位到已验证示范初态。原环境工厂和训练入口文件不修改。

轨迹时间：0–0.5 s 静止；0.5–2.5 s 缓慢闭合；2.5–3 s 保持；3–4.5 s 向源视频任务终点搬运；4.5–10 s 保持。
动作搜索完整记录在 `data/processed/seq_dexycb_001/physical_grasp_v2/search.json`。
v1 是根关节坐标映射尚未修正的失败实验，不用于训练。

## 物理验收

最终报告：`data/processed/seq_dexycb_001/physical_grasp_verified_v1/admission.json`。

| 检查 | 结果 |
| --- | --- |
| 初态种子 0–5 | 6/6 通过 |
| 独立保存动作重放 | 6/6 通过，逐步观测最大差 0 |
| 未参与搜索的种子 6–10 | 5/5 通过 |
| 连续有力接触抬杯 | 6.24–6.34 s |
| 最终杯底离桌 | 98.8–114.7 mm |
| 最终杯中心到源任务目标距离 | 29.2–45.5 mm |
| 全部种子最大手杯穿透，含物理子步 | 3.844 mm |
| 动作饱和率 | 不超过 2.58% |

标称种子 0 保持末态的受力接触几何为 `C_ffdistal`、`C_mfdistal`、`C_thdistal`，不是手掌或前臂托住杯子；该条最大负接触距离为 0。
环境保留原训练参数：friction=(1,0.5,0.01)、solref='-6000 -300'、margin=3 mm。正接触距离也可能有接触力，因此不以 dist<=0 作为是否抓持的判据。
所有种子的 3.844 mm 最大穿透低于本轮 5 mm 门槛，但不能称为严格零穿透；这是保留软接触模型下的验收结果。

准入要求在 `src/fromrealhand/grasp_gate.py`：持续抬杯 >=1 s；最后 1 s 杯底始终 >15 mm、有至少两个手部接触区域和 >0.05 N 接触力；最大穿透 <=5 mm；动作饱和率 <10%；最终目标距离 <0.1 m。
回放只在开头初始化手、杯状态，之后全部 env.step，不修改物体 qpos、设置焊接约束、关闭重力或增大摩擦作弊。

## 正式产物

- `data/demonstrations/relocate-mug-physics-verified-v1.pkl`：6 条、每条 1000 步，39 维观测、30 维动作，真实物理 rollout 的前状态/动作/后奖励。
- `data/processed/seq_dexycb_001/physical_grasp_verified_v1/admission.json`：准入报告、参数、各次回放指标、示范 SHA-256。
- 同目录 `seed_0/rollout.mp4`：成功物理控制视频；`replay_0/rollout.mp4`：保存动作的独立重放。
- 同目录 `training_preflight.json`：训练接口和 GPU 检查结果。
- `configs/dapg-mug-physics-verified-smoke.yaml`：20 次迭代，6 条示范，每轮 5 条采样、3 条评估，单进程初期验证，64×64 策略网络。

## 训练接口验证

不仅检查 pkl 格式，还用正式 `mjrl.samplers.base_sampler.do_rollout` 跑满 1000 步，复现示范观测最大误差为 0。六次课程复位的初始观测误差也均为 0。
GPU 前向与反向检查通过：NVIDIA GeForce RTX 4060 Laptop GPU。
设备分工保持现有上游实现：价值基线使用 GPU；MuJoCo 和策略主要计算仍在 CPU，不能称为全 GPU DAPG。
课程加载器要求 admission 通过并校验 pkl SHA-256，拒绝未准入或改动过的示范。
训练包装器拒绝覆盖已有同名训练目录，限制本入口 NUM_ITER=20、NUM_CPU=1。

## 可执行命令

```bash
cd /home/smgbro/mujoconew/GITHUB

# 默认只检查，不训练。
bash scripts/27_train_verified_smoke.sh

# 开始新的 20 次 DAPG 课程短训练。
bash scripts/27_train_verified_smoke.sh --train
```

不要直接用原 `train.py --cfg` 启动此配置，否则不会安装课程复位工厂，且默认 400 步无法完整覆盖新示范。
下一轮策略评估也必须使用 `make_verified_environment`；普通默认环境应作为另一个分布外评估单独报告。

本轮执行过的核心命令（使用 dexmv 环境及既有 LD_LIBRARY_PATH/LD_PRELOAD）：

```bash
python scripts/24_search_physical_grasp.py --trials 16 --output data/processed/seq_dexycb_001/physical_grasp_v2
python scripts/25_export_verified_grasps.py --output data/processed/seq_dexycb_001/physical_grasp_verified_v1 --demo data/demonstrations/relocate-mug-physics-verified-v1.pkl
python scripts/06_validate_demo.py data/demonstrations/relocate-mug-physics-verified-v1.pkl
python scripts/26_train_verified_curriculum.py
PYTHONPATH=src python -m unittest discover -s tests -q
```

示范质量检查通过；19 项测试通过。旧 chumpy/NumPy 弃用警告和 Gym float32 边界警告仍存在，未为消除警告升级旧依赖。

## 修改范围与后续

本轮新增 scripts/24–27、grasp_gate.py、trajectory_control.py、verified_curriculum.py、对应三个测试文件、短训练配置和本文；原始数据、旧 checkpoint、已有 dirty worktree 保留；未提交或推送 Git。

下一步是 20 次课程短训练并用多个独立种子评估**学到的策略**，单独统计接触力、杯底高度和目标距离。
只有策略在这个窄分布上通过，才扩展更远的接近初态、更多真实序列和更宽的位置/朝向扰动。当前物理控制器通过，不保证 BC/DAPG 自动学会，也不代表阶段 3 的策略验收已完成。
