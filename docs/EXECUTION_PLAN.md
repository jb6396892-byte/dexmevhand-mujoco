# 实际操作执行计划

这份文档用于后续逐步实施。每一步都包含前置条件、具体操作、产出和验收标准。
上一阶段没有通过验收前，不进入下一阶段。

## 使用规则

- 所有命令默认在 `/home/smgbro/mujoconew/GITHUB` 执行。
- 每条真实轨迹使用唯一编号，例如 `seq_dexycb_001`。
- 所有平移统一使用米，旋转统一保存为 `3x3` 矩阵或 `4x4` 齐次矩阵。
- 原始数据、模型、`.npy`、`.pkl`、checkpoint 和日志不提交 Git。
- 每次实验使用 [实验记录模板](WORK_LOG_TEMPLATE.md) 保存数据来源、命令和结果。
- 每完成一个可验证阶段就提交一次 Git，避免多个问题混在同一个提交中。

## 总任务清单

- [x] A0：跑通原始 `relocate-mug` 训练和可视化。
- [x] A1：建立真实视频到 DexMV 的基础项目框架。
- [ ] B1：准备一条 DexYCB `025_mug` 真实 RGB-D 轨迹。
- [ ] B2：编写 DexYCB 数据扫描和转换程序。
- [ ] B3：验证手、杯子和相机坐标。
- [ ] B4：生成并回放第一条真实视频 demonstration。
- [ ] B5：使用真实 demonstration 完成一次短训练。
- [ ] C1：扩展到多条真实抓杯轨迹。
- [ ] C2：把完整轨迹拆成低层技能。
- [ ] C3：分别训练低层技能策略。
- [ ] C4：实现技能注册表和顺序执行器。
- [ ] D1：实现规则高层规划器。
- [ ] D2：训练语言模型生成结构化技能计划。
- [ ] E1：录制或获取倒水视频并增加新技能。
- [ ] E2：建立 `pour-mug` 环境并完成完整评估。

---

## B1：准备第一条 DexYCB 真实轨迹

### 目标

获得一条真正拍摄的右手抓取 `025_mug` 的 RGB-D 序列，并保留标定与位姿标签。

### 前置条件

- 磁盘至少预留 30 GB，避免下载和解压同时占满磁盘。
- `bash scripts/00_check_env.sh` 能通过。
- 确认 DexYCB 数据仅用于符合其许可的研究用途。

### 操作

1. 从 DexYCB 官方项目页下载：
   `https://dex-ycb.github.io/`
2. 第一批只下载一个受试者包、`calibration.tar.gz` 和 `models.tar.gz`。
3. 解压到：

```text
data/external/dexycb/
  20200709-subject-01/
  calibration/
  models/
```

4. 不要直接下载全部 119 GB 数据。
5. 记录数据版本、下载日期和文件校验值。
6. 在受试者的 100 条序列中筛选：
   - 目标物体为 `025_mug`。
   - `mano_sides` 为右手。
   - 杯子和手尽量无遮挡。
   - 选一个画面稳定的固定相机视角。

### 产出

```text
data/external/dexycb/
data/real_data/relocate_mug/seq_dexycb_001/meta.json
```

`meta.json` 至少记录原始 subject、sequence、camera serial、帧率、手侧和物体 ID。

### 验收标准

- RGB 和 depth 帧数一致。
- 能读取相机内参和相机外参。
- 能读取 MANO/3D 手关节和 `025_mug` 6D 位姿。
- 连续播放时能看到完整的“接近、抓住、抬起”动作。

### 常见问题

- 没有 `025_mug`：检查序列 `meta.yml` 中的 `ycb_ids` 和抓取目标索引。
- 找不到右手：检查 `mano_sides`，不要仅凭图像左右判断。
- 数据太大：先保留一个视角的工作副本，原始包继续放在 external 目录。

---

## B2：编写 DexYCB 扫描和转换程序

### 目标

把 DexYCB 原始格式转换成本项目已经约定的目录和 `.npy` 格式。

### 需要新增的程序

```text
scripts/09_scan_dexycb.py
scripts/10_convert_dexycb.py
scripts/11_visualize_source_pose.py
src/fromrealhand/dexycb_io.py
```

### `09_scan_dexycb.py` 需要完成

- 遍历 subject 下的所有序列。
- 读取每条序列的元数据。
- 输出目标物体、手侧、帧数和可用相机。
- 支持 `--object 025_mug --hand-side right` 筛选。
- 输出 JSON 或 CSV 清单，不能只打印终端文本。

建议调用：

```bash
python scripts/09_scan_dexycb.py \
  --root data/external/dexycb \
  --object 025_mug \
  --hand-side right \
  --output data/processed/dexycb_mug_sequences.json
```

### `10_convert_dexycb.py` 需要完成

- 复制或链接选定视角的 RGB 和 depth 帧。
- 保存 `camera_matrix.npy`、畸变参数和 `camera_to_world.npy`。
- 将 3D 手关节转换成 `joints_*.npy`。
- 将 DexYCB MANO/全局手部参数转换成 `results_global_*.npy`。
- 将目标杯子 6D 位姿保存到 `object_pose/*.npy`。
- 保存源帧编号映射，避免跳帧后标签错位。
- 对低置信度或无效帧写入 `valid_frames.npy`，不能直接忽略。

建议调用：

```bash
python scripts/10_convert_dexycb.py \
  --root data/external/dexycb \
  --sequence SUBJECT/SEQUENCE \
  --camera CAMERA_SERIAL \
  --output data/real_data/relocate_mug/seq_dexycb_001
```

### 产出

目录必须符合 [数据格式说明](DATA_FORMAT.md)。

### 验收标准

- RGB、depth、hand pose 和 object pose 数量一致。
- 每帧文件编号能够一一对应。
- 所有平移数值已经换算成米。
- `meta.json` 明确写出每种位姿当前所在坐标系。

---

## B3：验证坐标、尺度和位姿

### 目标

证明转换结果在几何上正确，再进入 retarget。

### 操作

1. 用 `camera_matrix.npy` 把 3D 手关节投影回 RGB 图像。
2. 把 YCB mug mesh 按物体 6D 位姿渲染到原图。
3. 在 3D 坐标中同时显示：
   - 相机坐标轴。
   - 世界坐标轴。
   - 桌面平面。
   - 手骨架。
   - 杯子 mesh。
4. 绘制手腕和杯子平移轨迹，检查逐帧速度尖峰。
5. 随机抽查开头、中间和结尾至少 10 帧。

### 产出

```text
data/processed/seq_dexycb_001/overlay/
data/processed/seq_dexycb_001/trajectory_report.json
```

报告至少包含有效帧比例、最大位移跳变、手腕范围和杯子范围。

### 验收标准

- 手关节投影与真实手基本重合。
- 杯子 mesh 的位置、方向和尺度与画面一致。
- 桌面法向方向正确，杯子起始位置位于桌面上方。
- 没有左右镜像、轴交换、毫米/米混用和矩阵方向错误。

### 失败时检查顺序

1. 内参是否对应当前分辨率和相机。
2. `camera_to_world` 是否误用了 `world_to_camera`。
3. 行向量/列向量和矩阵乘法顺序是否一致。
4. DexYCB 标签单位是否已经转换成米。
5. RGB、depth 和 label 是否错开一帧。

---

## B4：retarget 并生成第一条 demonstration

### 目标

让真实视频中的动作在 MuJoCo Adroit 手中可视化和回放。

### 操作 1：retarget

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv \
  python scripts/03_retarget_one.py \
  --hand-dir data/real_data/relocate_mug/seq_dexycb_001/hand_pose \
  --output data/real_data/relocate_mug/seq_dexycb_001/retargeting.pkl
```

检查输出帧数、关节范围和 NaN。

### 操作 2：可视化对齐

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv \
  python scripts/04_visualize_retargeting.py \
  --retargeting data/real_data/relocate_mug/seq_dexycb_001/retargeting.pkl \
  --object-dir data/real_data/relocate_mug/seq_dexycb_001/object_pose \
  --camera-to-world data/real_data/relocate_mug/seq_dexycb_001/calib/camera_to_world.npy
```

### 操作 3：生成 demonstration

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv \
  python scripts/05_generate_demo.py \
  --sequence-dir data/real_data/relocate_mug/seq_dexycb_001 \
  --output data/demonstrations/relocate-mug-real.pkl \
  --trajectory-id seq_dexycb_001
```

### 操作 4：检查

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv \
  python scripts/06_validate_demo.py \
  data/demonstrations/relocate-mug-real.pkl
```

### 验收标准

- `retargeting.pkl` 和视频有效帧数一致。
- Adroit 手和杯子整体位置对齐，手指闭合时间合理。
- demonstration 包含 `observations/actions/rewards/sim_data/model_data`。
- action 没有大量持续饱和到 `-1` 或 `1`。
- MuJoCo 回放中杯子不会在抓取前跳动或穿过手掌。

---

## B5：用真实 demonstration 做短训练

### 目标

先证明真实数据能完成采样和策略更新，不立即进行长时间训练。

### 操作

1. 新建 `configs/dapg-mug-real-smoke.yaml`。
2. 从正式配置复制参数，只把：
   - `NUM_ITER` 改成 `20`。
   - `NUM_CPU` 改成适合当前机器的小值。
   - `JOB_DIR` 指向独立的 smoke 目录。
3. 保持 `BC_INIT: true` 和 `USE_DAPG: true`。
4. 训练前再次运行 `scripts/06_validate_demo.py`。
5. 运行 smoke training，确认行为克隆、采样和参数更新都执行。

### 产出

```text
training_log/relocate-mug-real-smoke/
docs/run_logs/<日期>-real-demo-smoke.md
```

### 验收标准

- 训练完成 20 次迭代，无 NaN 和 shape 错误。
- policy、日志和迭代结果正常保存。
- reward 没有立即崩溃为异常大值或全零。
- 能加载训练后的策略进行可视化。

通过后再恢复 `NUM_ITER: 2000` 进行正式训练。

---

## C1：扩展到多条真实抓杯轨迹

### 目标

避免策略只记住单条轨迹。

### 操作

- 第一轮增加到至少 5 条经过人工检查的右手 mug 轨迹。
- 第二轮扩展到不同受试者、视角和杯子初始位姿。
- 使用 `--append` 把每条轨迹加入同一个 demonstration 文件。
- 每加入一条都单独验证，不能最后一起排查。
- 记录成功轨迹、剔除轨迹和剔除原因。

### 验收标准

- 多条轨迹字段维度一致。
- 轨迹长度分布合理，没有明显异常短轨迹。
- 训练和测试按 subject 或原始 sequence 分开，避免相邻帧泄漏。

---

## C2：拆分低层技能

### 目标

把完整抓杯动作拆成可复用的 `reach`、`grasp`、`lift`、`transport`。

### 需要新增

```text
scripts/12_segment_skills.py
scripts/13_review_skill_segments.py
scripts/14_export_skill_demos.py
src/fromrealhand/skill_segmentation.py
configs/skills/*.yaml
```

### 初始切分规则

- `reach`：手腕持续接近杯子，杯子基本静止。
- `grasp`：指尖距离减小，手与杯子开始建立稳定相对位姿。
- `lift`：杯子离开桌面，高度持续增加。
- `transport`：杯子高度稳定或缓慢变化，同时水平移动到目标区域。

### 产出

```text
annotations/skill_segments.json
data/demonstrations/reach-mug.pkl
data/demonstrations/grasp-mug.pkl
data/demonstrations/lift-mug.pkl
data/demonstrations/transport-mug.pkl
```

### 验收标准

- 每个片段都能独立回放。
- 技能之间没有缺帧或重复执行造成的明显跳变。
- 自动边界已经人工审核并标记 `reviewed: true`。

---

## C3：训练低层技能策略

### 目标

得到四个可以单独调用的可靠策略。

### 操作顺序

1. 为每个技能建立独立 MuJoCo task/config。
2. 先行为克隆，确认策略能复现平均轨迹。
3. 使用 DAPG 微调。
4. 随机化手和杯子的初始位置、方向及目标位置。
5. 每个技能实现独立成功条件和失败分类。
6. 保存每个技能的最佳策略和评估报告。

### 验收标准

- `reach` 不碰倒杯子并到达预抓取区域。
- `grasp` 建立稳定抓取，短时间内不滑落。
- `lift` 把杯子抬到目标高度且不过度旋转。
- `transport` 把杯子移动到目标位姿范围且不掉落。

---

## C4：实现技能执行器

### 目标

通过固定计划顺序调用多个策略，并在每一步检查结果。

### 需要新增

```text
src/fromrealhand/hierarchy/skill_registry.py
src/fromrealhand/hierarchy/executor.py
src/fromrealhand/hierarchy/conditions.py
src/fromrealhand/hierarchy/affordance.py
scripts/15_run_skill_plan.py
```

统一接口：

```python
result = executor.run(
    skill="grasp",
    object="mug",
    goal={"grasp": "handle"},
)
```

执行器必须记录技能名称、输入状态、成功概率、终止原因、耗时和重试次数。

### 验收标准

- 固定计划 `reach -> grasp -> lift -> transport` 可以完成。
- 任一技能失败时，不继续盲目执行后续技能。
- 支持超时、重试和安全停止。

---

## D1：规则高层规划器

### 目标

在使用语言模型前验证高层与低层接口。

### 示例规则

```text
“抓起杯子”
  -> reach -> grasp -> lift

“把杯子移到目标位置”
  -> reach -> grasp -> lift -> transport
```

### 需要新增

```text
src/fromrealhand/hierarchy/planner.py
src/fromrealhand/hierarchy/plan_schema.py
configs/skill_registry.yaml
scripts/16_run_instruction.py
```

### 验收标准

- 规划结果通过严格 JSON Schema 检查。
- 未注册技能、错误物体和越界参数会被拒绝。
- 每个技能执行后根据新状态决定继续、重试或重新规划。

---

## D2：语言模型高层规划器

### 目标

让不同自然语言表达映射到同一套经过验证的技能计划。

### 操作

1. 从规则规划器和成功执行日志生成训练样本。
2. 输入包含：指令、场景 JSON、可用技能和技能参数约束。
3. 输出只允许是符合 Schema 的技能计划 JSON。
4. 使用预训练小型语言模型做 LoRA/SFT，不从头训练。
5. 用验证集检查指令改写、物体变化和缺失前置条件。
6. 最终技能评分结合语言相关性和低层可行性：

```text
score(skill) = language_score * affordance_success_probability
```

### 验收标准

- 输出 JSON 合法率达到预先设定目标。
- 不生成技能库以外的动作。
- 杯子未抓住时不会直接调用 `lift` 或 `tilt`。
- 同义指令得到一致的技能序列。

---

## E1：增加倒水视频和技能

### 目标

补齐 DexYCB 中缺少的倒水、恢复竖直、放下和松手动作。

### 数据采集

- 相机固定，记录内参和 `camera_to_world`。
- 使用与 MuJoCo mug 尺寸接近的杯子。
- 第一版可以给杯子贴 ArUco 标记，降低 6D 位姿估计难度。
- 完整录制抓取、抬起、移动、倾斜、恢复、放下和松手。
- 同时记录目标容器的位置和尺寸。

### 新技能

```text
tilt(mug, angle)
upright(mug)
place(mug, target_pose)
release(mug)
```

### 验收标准

- 新视频可以通过同一位姿恢复和 retarget 管线。
- 每个新增技能可以独立训练和回放。

---

## E2：建立 `pour-mug` 环境并完整评估

### 目标

执行“拿起杯子并倒入目标容器”的完整层次化任务。

### 第一版倒水成功条件

- 杯口位于目标容器上方。
- 倾斜角进入设定范围。
- 保持指定时间。
- 杯子没有掉落或碰倒目标容器。

第一版不模拟真实液体。几何任务稳定后，再评估粒子液体或外部流体模拟。

### 完整评估维度

- 感知：投影误差、无效帧率、位姿跳变。
- retarget：手杯对齐和接触合理性。
- 技能：每个技能的独立成功率。
- 规划：计划合法率和技能顺序正确率。
- 完整任务：成功率、平均重试次数、恢复率和失败类型。
- 泛化：不同受试者、杯子位姿、相机视角和指令表达。

### 最终验收标准

- 从指定数据和配置可以重复训练。
- 输入自然语言后能生成可解释的技能计划。
- 每个技能的执行结果和失败原因可以追踪。
- 完整“抓杯子 -> 倒水 -> 放回”任务可以在 MuJoCo 中稳定运行。

---

## 当前下一步

现在只执行 B1，不同时开始后面的模型训练：

1. 检查剩余磁盘空间。
2. 下载一个 DexYCB subject、calibration 和 models。
3. 找出一条右手 `025_mug` 序列。
4. 保存序列清单和 `meta.json`。
5. 确认 RGB、depth、手部和杯子标签都能读取。

B1 验收通过后，再开始编写 `09_scan_dexycb.py` 和 `10_convert_dexycb.py`。
