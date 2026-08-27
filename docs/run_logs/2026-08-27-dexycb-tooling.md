# DexYCB 数据工具预实现记录

日期：2026-08-27

## 目的

在暂不下载 12 GB DexYCB subject 数据的情况下，提前完成 B2/B3 所需的数据
扫描、单视角转换、坐标验证和报告工具。

## 本次完成

- 新增 `src/fromrealhand/dexycb_io.py`，读取官方 `meta.yml`、相机标定和
  `labels_*.npz`。
- 新增 `scripts/09_scan_dexycb.py`，按抓取目标和手侧输出 JSON/CSV 清单。
- 新增 `scripts/10_convert_dexycb.py`，保存 RGB、depth、标定、手关节、
  手部框架、MANO 参数、物体位姿、帧映射和有效帧掩码。
- 新增 `scripts/11_visualize_source_pose.py`，输出手部重投影、物体坐标轴、
  三维轨迹图和 `trajectory_report.json`。
- 修改 `scripts/03_retarget_one.py`，让手部与物体统一应用
  `camera_to_world.npy`。

## 测试

命令：

```bash
PYTHONPYCACHEPREFIX=/tmp/fromrealhand-pycache \
MPLCONFIGDIR=/tmp/fromrealhand-matplotlib \
/home/smgbro/miniconda3/envs/dexmv/bin/python \
  -m unittest discover -s tests -v
```

结果：2 个测试全部通过。

合成数据包含 3 帧，其中中间帧故意设置为无效手部标签。验证结果：

- 扫描出 1 条右手 `025_mug` 序列。
- 转换结果为 `2/3` 有效帧，原始帧编号没有被删除。
- 无效帧的 `results_global_*.npy` 为 NaN，并由 `valid_frames.npy` 标记。
- 42 个有效手关节点的平均和最大重投影误差均为 `0 px`。
- 成功生成 2 张 overlay 和一张三维轨迹图。

## 仍未验收

- 尚未在真实 DexYCB subject 上检查目录、帧数和相机完整性。
- 尚未人工选择遮挡最少的 `025_mug` 固定相机视角。
- 当前 `results_global_*.npy` 从 `joint_3d` 骨架方向推导；正式训练前需要
  MANO 官方模型恢复精确旋转并进行对比。
- 尚未使用真实转换结果运行 retarget、MuJoCo 回放和 DAPG smoke training。

因此 B1、B2 和 B3 仍保持未完成状态。
