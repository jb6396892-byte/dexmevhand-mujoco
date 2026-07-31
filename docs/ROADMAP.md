# Implementation Roadmap

## Completed Foundation

- `relocate-mug` training and policy visualization are working locally.
- DexMV simulation and dexmv-learn DAPG are reused instead of copied.
- Environment validation, frame extraction, retargeting, visualization,
  demonstration generation, validation, training, and policy visualization
  entry points exist in this repository.

## Phase 1: Acquire One Real Mug Sequence

Tasks:

- Download one DexYCB subject package, calibration, and YCB models.
- Select a right-hand `025_mug` trajectory and one fixed camera view.
- Register the sequence metadata and source license.

Acceptance:

- One real RGB-D sequence can be played from relaxed hand to lifted mug.
- Intrinsics, extrinsics, depth, hand labels, and object labels are available.

## Phase 2: Convert Video Labels to World-Space Poses

Tasks:

- Add a DexYCB sequence scanner and `025_mug` filter.
- Convert MANO/3D joints and mug 6D poses into this project's layout.
- Convert all frames to one metric world coordinate frame.
- Add 2D overlay and 3D trajectory visualization.

Acceptance:

- Hand and mug projections align with source frames.
- World-space trajectories are continuous and use meters.

## Phase 3: Build a Real-Video MuJoCo Demonstration

Tasks:

- Feed converted hand poses into DexMV retargeting.
- Align the mug trajectory with the MuJoCo table and YCB mug model.
- Generate and validate `relocate-mug-real.pkl`.
- Replay the trajectory in MuJoCo.

Acceptance:

- The Adroit hand reaches and moves the mug without major penetration,
  coordinate flips, or frame jumps.

## Phase 4: Segment Reusable Skills

Tasks:

- Define the skill annotation schema.
- Detect and review boundaries for `reach`, `grasp`, `lift`, and `transport`.
- Export one demonstration set per skill.
- Define initiation, success, termination, and timeout predicates.

Acceptance:

- Every segment can be replayed independently.
- Segment boundaries correspond to observable physical events.

## Phase 5: Train Low-Level Skill Policies

Tasks:

- Train behavior-cloning initialization for each skill.
- Fine-tune with DAPG and randomized initial conditions.
- Collect rollout success data for an affordance model.
- Add skill-level evaluation reports.

Acceptance:

- Each skill succeeds reliably across held-out initial poses.
- Failures are classified rather than reported only as total reward.

## Phase 6: Implement the Skill Runtime

Tasks:

- Add a skill registry and common execution API.
- Add observation, timeout, retry, recovery, and logging support.
- Chain `reach -> grasp -> lift -> transport` with feedback between skills.

Acceptance:

- A fixed symbolic plan completes the mug-relocation workflow.
- The runtime stops or recovers when a skill fails.

## Phase 7: Add the High-Level Planner

Tasks:

- Implement a deterministic instruction-to-plan baseline.
- Define a strict JSON schema for plans.
- Build instruction, scene, plan, and recovery examples.
- Fine-tune a pretrained language model after enough plan data exists.
- Rank proposed skills using both language relevance and affordance scores.

Acceptance:

- Instruction paraphrases produce valid and physically executable plans.
- Every executed step and its success score are visible in logs.

## Phase 8: Add Pouring

Tasks:

- Record real `tilt`, `upright`, `place`, and `release` demonstrations.
- Add a target container and a `pour-mug` MuJoCo environment.
- Implement geometric pouring success conditions.
- Train the new low-level skills and extend the planner vocabulary.

Acceptance:

- The instruction "pick up the mug and pour into the container" produces and
  executes a complete skill sequence.

## Phase 9: End-to-End Evaluation

Tasks:

- Test unseen subjects, mug poses, camera views, and language paraphrases.
- Measure perception, retargeting, skill, planner, and end-to-end failures.
- Add domain randomization before considering real-robot transfer.

Acceptance:

- Results are reproducible from versioned configs and documented datasets.
- Each failure can be assigned to perception, planning, or control.

## Immediate Next Deliverable

Implement the DexYCB converter and validate one right-hand `025_mug` trajectory
through the existing retargeting and demonstration scripts.
