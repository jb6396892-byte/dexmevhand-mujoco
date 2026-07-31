# From Real Hand Video to DexMV Relocate-Mug

This project is a lightweight workspace for turning real mug-grasping videos into DexMV demonstrations, then training a MuJoCo Adroit-hand policy with imitation learning and reinforcement learning.

本项目从真实人手抓取视频出发，将手部和杯子运动恢复到统一世界坐标，
再通过 DexMV retargeting 生成 MuJoCo demonstration，使用行为克隆和 DAPG
训练灵巧手策略。后续目标是在可靠的低层技能之上增加可解释的层次化控制：
高层语言规划器输出结构化技能计划，低层策略执行精确动作。

## Project Status

Completed:

- Local `relocate-mug` training and policy visualization.
- Environment validation and reusable video-to-demonstration scripts.
- Demonstration validation and DAPG training entry points.

Next milestone:

- Convert one real DexYCB right-hand `025_mug` sequence.
- Replay it through the existing retargeting and demonstration pipeline.
- Segment it into `reach`, `grasp`, `lift`, and `transport` skills.

Project documents:

- [Hierarchical imitation-learning architecture](docs/ARCHITECTURE.md)
- [Implementation roadmap](docs/ROADMAP.md)
- [Data and annotation format](docs/DATA_FORMAT.md)

It reuses the existing local projects:

- `/home/smgbro/dexmv-sim`
- `/home/smgbro/dexmv-learn`
- conda env `dexmv`

It does not copy the full DexMV assets or pretrained models.

## Pipeline

```text
real video
  -> frame extraction
  -> hand 3D pose estimation
  -> mug 6D pose estimation
  -> camera/world coordinate transform
  -> human hand to Adroit hand retargeting
  -> DexMV demonstration pickle
  -> BC + DAPG/TRPO training
  -> policy visualization and rollout evaluation
```

DexMV does not estimate hand pose or object pose from raw video. This project expects those estimation results as input.

The planned hierarchical extension is:

```text
instruction + scene state
  -> high-level planner
  -> validated skill plan
  -> skill executor and affordance checks
  -> low-level MuJoCo policies
  -> state feedback and replanning
```

## Expected Real-Data Layout

Each real trajectory should look like:

```text
data/real_data/relocate_mug/seq_000/
  rgb/
  calib/
    camera_to_world.npy
  hand_pose/
    results_global_000.npy
    joints_000.npy
    results_global_001.npy
    joints_001.npy
  object_pose/
    000.npy
    001.npy
  meta.json
```

Minimum required files for demo generation:

- `hand_pose/results_global_*.npy`
- `hand_pose/joints_*.npy`
- `object_pose/*.npy`
- optional `calib/camera_to_world.npy` if object poses are still in camera coordinates

Object pose files may contain either a raw `4x4` pose matrix, or a Python dict saved with NumPy where one entry is the mug pose.

## Environment

The default local paths are in `.env.example`. If needed, create a local `.env` with overrides. The important MuJoCo variables are:

```bash
LD_LIBRARY_PATH=/home/smgbro/.mujoco/mujoco200/bin:/usr/lib/x86_64-linux-gnu
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Check the environment:

```bash
bash scripts/00_check_env.sh
```

## Basic Usage

Create a sequence directory template:

```bash
python scripts/02_prepare_pose_dirs.py --seq seq_000
```

Extract frames from a video:

```bash
python scripts/01_extract_frames.py \
  --video data/raw_videos/seq_000.mp4 \
  --output data/real_data/relocate_mug/seq_000/rgb
```

After external hand/object pose estimation has filled `hand_pose/` and `object_pose/`, run retargeting:

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/03_retarget_one.py \
  --hand-dir data/real_data/relocate_mug/seq_000/hand_pose \
  --output data/real_data/relocate_mug/seq_000/retargeting.pkl
```

Visualize retargeted hand and mug:

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/04_visualize_retargeting.py \
  --retargeting data/real_data/relocate_mug/seq_000/retargeting.pkl \
  --object-dir data/real_data/relocate_mug/seq_000/object_pose \
  --camera-to-world data/real_data/relocate_mug/seq_000/calib/camera_to_world.npy
```

Generate a DexMV demonstration:

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/05_generate_demo.py \
  --sequence-dir data/real_data/relocate_mug/seq_000 \
  --output data/demonstrations/relocate-mug-real.pkl \
  --trajectory-id seq_000
```

Validate the demonstration:

```bash
/home/smgbro/miniconda3/bin/conda run -n dexmv python scripts/06_validate_demo.py \
  data/demonstrations/relocate-mug-real.pkl
```

Train with DexMV:

```bash
bash scripts/07_train_dapg.sh
```

Visualize a trained policy:

```bash
bash scripts/08_visualize_policy.sh /path/to/best_policy.pickle
```

## What Is Still Missing

- A downloaded or self-recorded real mug-grasping sequence.
- Camera intrinsics and `camera_to_world` extrinsics.
- Hand pose estimation output in DexMV-compatible `.npy` naming.
- Mug 6D pose estimation output per frame.
- A decision on whether the real mug can be approximated by the YCB mug.
- Enough successful trajectories to train robustly; start with 1 for debugging, then collect many.
- Skill segmentation, skill success predicates, and a common skill executor.
- A high-level instruction-to-plan model.
- A separate MuJoCo environment and demonstrations for pouring.

## Notes

- Start with state-input policy training. Do not train an end-to-end visual policy until state-based training is stable.
- Keep `object_scale=0.8` for the first real-data experiments because `relocate-mug` has already been verified locally with that scale.
- Do not manually scale policy actions before calling `YCBRelocate.step()`. DexMV scales normalized actions internally.
- Raw recordings, datasets, calibration output, policies, and checkpoints are intentionally excluded from Git.
