# 数据与标注格式

## 单条轨迹目录

```text
data/real_data/relocate_mug/seq_000/
  rgb/
  depth/
  calib/
    camera_matrix.npy
    dist_coeffs.npy
    camera_to_world.npy
  hand_pose/
    results_global_000.npy
    joints_000.npy
  object_pose/
    000.npy
  annotations/
    skill_segments.json
  meta.json
  retargeting.pkl
```

所有平移统一使用米。`camera_to_world.npy` 是 `4x4` 齐次变换矩阵：

```text
point_world = camera_to_world @ point_camera
pose_world = camera_to_world @ pose_camera
```

使用前必须确认矩阵方向。如果标定程序保存的是 `world_to_camera`，需要先求逆。

## 轨迹元数据

建议的 `meta.json`：

```json
{
  "sequence_id": "seq_000",
  "source": "DexYCB",
  "source_sequence": "subject/sequence",
  "camera_id": "camera_serial",
  "fps": 30.0,
  "object_name": "mug",
  "object_model": "025_mug",
  "object_scale": 0.8,
  "hand_side": "right",
  "pose_frame": "world",
  "translation_unit": "meter"
}
```

关键字段：

- `source_sequence`：可以追溯到原始数据。
- `camera_id`：确定使用的视角和标定参数。
- `hand_side`：第一版只筛选右手轨迹。
- `pose_frame`：防止相机坐标和世界坐标混用。
- `translation_unit`：防止毫米和米混用。

## 技能片段标注

建议的 `annotations/skill_segments.json`：

```json
{
  "instruction": "拿起杯子",
  "reviewed": true,
  "segments": [
    {
      "skill": "reach",
      "start_frame": 0,
      "end_frame": 42,
      "object": "mug",
      "goal": {}
    },
    {
      "skill": "grasp",
      "start_frame": 43,
      "end_frame": 61,
      "object": "mug",
      "goal": {"grasp": "handle"}
    },
    {
      "skill": "lift",
      "start_frame": 62,
      "end_frame": 89,
      "object": "mug",
      "goal": {"height": 0.12}
    }
  ]
}
```

第一版需要保留标注人、审核状态和备注。自动生成的边界不能直接覆盖人工审核
过的边界。

## 手部姿态文件

```text
hand_pose/results_global_*.npy
hand_pose/joints_*.npy
```

文件编号必须与 RGB、深度和物体位姿帧编号一致。缺失或低置信度帧需要明确
标记，不能静默错位。

## 物体位姿文件

```text
object_pose/000.npy
object_pose/001.npy
...
```

每个文件可以是：

- 直接保存的 `4x4` 物体位姿矩阵。
- 包含多个物体位姿的 NumPy 字典，通过 `object_id` 选择 mug。

物体位姿转换遵循：

```text
pose_world = camera_to_world @ pose_camera
```

## DexMV demonstration

DexMV 训练文件顶层是轨迹字典：

```text
dict[trajectory_id] = trajectory_data
```

每条轨迹至少包含：

- `observations`
- `actions`
- `rewards`
- `sim_data`
- `model_data`

生成后使用：

```bash
python scripts/06_validate_demo.py data/demonstrations/relocate-mug-real.pkl
```

## Git 数据规则

GitHub 仓库只保存代码、配置示例和数据格式说明。以下内容不提交：

- 原始视频和抽取帧。
- DexYCB 等外部数据集。
- 相机标定输出和 `.npy` 位姿文件。
- demonstration pickle。
- 训练日志、策略、模型和 checkpoint。

仓库中只记录数据下载来源、许可、版本和校验值。
