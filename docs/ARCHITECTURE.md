# Hierarchical Imitation Learning Architecture

## Goal

The project learns dexterous manipulation from real human videos while keeping
language reasoning separate from high-frequency motor control.

```text
natural-language instruction             real RGB/RGB-D video
             |                                      |
             v                                      v
      high-level planner <--------- structured scene state
             |
             | validated skill plan
             v
  skill executor + affordance scorer + termination checks
             |
             v
  low-level policies: reach / grasp / lift / transport / tilt / place
             |
             v
        MuJoCo Adroit actions
             |
             +---------------- state feedback and replanning
```

The language model never emits joint torques or the 30-dimensional normalized
Adroit action directly. It emits a plan in a constrained skill language.

## Layer 1: Perception and Video Reconstruction

Inputs:

- RGB or RGB-D frames.
- Camera intrinsics and camera-to-world extrinsics.
- A metric mug model or a known YCB mug model.

Outputs for every frame:

- 3D hand joints and MANO/global hand pose.
- Mug 6D pose.
- A shared world coordinate frame in meters.
- Confidence and validity flags.

Before training, project the reconstructed hand and object back onto the source
images and inspect the sequence in 3D. Retargeting should not be used to hide
coordinate, scale, or tracking errors.

## Layer 2: High-Level Planner

The planner maps an instruction and a structured scene description to a skill
plan:

```json
{
  "instruction": "pick up the mug and pour",
  "plan": [
    {"skill": "reach", "object": "mug"},
    {"skill": "grasp", "object": "mug", "grasp": "handle"},
    {"skill": "lift", "height": 0.12},
    {"skill": "transport", "target": "above_container"},
    {"skill": "tilt", "angle_deg": 100},
    {"skill": "upright"},
    {"skill": "place", "target": "table"},
    {"skill": "release"}
  ]
}
```

Every plan is schema-validated. Unknown skills, objects, and unsafe parameters
are rejected before execution.

For each candidate skill, selection combines:

```text
task relevance from the language model
    x
predicted probability that the skill can succeed in the current state
```

This prevents semantically plausible but physically impossible calls, such as
tilting a mug before it is grasped.

Start with a deterministic rule planner. Fine-tune a pretrained language model
only after the skill interface and execution traces are stable. Training data
for the planner consists of:

- Instruction and scene-state inputs.
- Valid skill-plan JSON outputs.
- Paraphrased instructions.
- Failed plans and recovery plans.
- Skill success estimates collected from simulation rollouts.

## Layer 3: Skill Executor

The executor owns the runtime contract:

```python
result = executor.run(
    skill="grasp",
    object="mug",
    goal={"grasp": "handle"},
)
```

Each skill defines:

- Initiation conditions.
- Goal parameters.
- A low-level policy.
- Termination and success predicates.
- Timeout and recovery behavior.

The executor observes the scene again after every skill. It can retry, call a
recovery skill, or ask the planner to produce a new plan.

## Layer 4: Low-Level Policies

The initial state-input policy is:

```text
pi_low(action | simulation_state, skill_id, skill_goal)
```

The simulation state can include hand joint positions and velocities, object
pose and velocity, target pose, and contacts. Demonstrations come from:

```text
video pose reconstruction
  -> DexMV human-to-Adroit retargeting
  -> MuJoCo demonstration generation
  -> skill segmentation
```

The first implementation should train one policy per skill:

- `reach(mug)`
- `grasp(mug)`
- `lift(mug, height)`
- `transport(mug, target_pose)`

Each policy is initialized with behavior cloning and refined with DAPG in
MuJoCo. Once the individual policies are reliable, they can share one backbone
conditioned on `skill_id` and `skill_goal`.

## Video Segmentation

A complete trajectory is divided into reusable options:

```text
hand approaches object        -> reach
fingers close                 -> grasp
object leaves support         -> lift
object follows the hand       -> transport
object rotates over target    -> tilt
object returns upright        -> upright
object contacts support       -> place
fingers open and move away    -> release
```

Automatic boundaries can use hand-object distance, fingertip distance, object
velocity, height, tilt angle, and support contact. Initial annotations should be
manually reviewed.

## Pouring Environment

The existing `relocate-mug` task is sufficient for reach, grasp, lift, and
transport. Pouring requires a new environment with a target container.

The first version should use a geometric success condition:

- Mug rim is above the target container.
- Mug tilt is within a configured range.
- The pose is maintained for a minimum duration.
- The mug is not dropped.

Fluid simulation can be added later. It is not required to validate the
hierarchical policy architecture.

## Evaluation

Evaluate each layer separately:

- Pose reconstruction: projection error, continuity, and valid-frame ratio.
- Retargeting: hand-object alignment and contact plausibility.
- Skills: success rate across randomized initial states.
- Planning: valid-plan rate and correct skill ordering.
- End to end: task success, retries, recovery rate, and failure category.

Dataset splits should separate subjects, initial mug poses, camera views, and
instruction paraphrases.
