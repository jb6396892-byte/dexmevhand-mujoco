# Data and Annotation Format

## Sequence Layout

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

All metric translations use meters. `camera_to_world.npy` is a homogeneous
`4x4` transform:

```text
point_world = camera_to_world @ point_camera
pose_world = camera_to_world @ pose_camera
```

## Sequence Metadata

Suggested `meta.json`:

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

## Skill Segments

Suggested `annotations/skill_segments.json`:

```json
{
  "instruction": "pick up the mug",
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

The first dataset version should keep manual review metadata such as annotator,
review status, and notes. Automatically generated boundaries must not silently
replace reviewed boundaries.

## Demonstration Pickle

DexMV training expects a top-level trajectory dictionary. Every trajectory
contains:

- `observations`
- `actions`
- `rewards`
- `sim_data`
- `model_data`

Large pickle files, raw video, labels, model files, and calibration outputs are
excluded from Git. Store download instructions and checksums in the repository,
not the dataset bytes.
