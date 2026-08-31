# From trajectory following to an end-effector metric

Status: **step A done and validated** (2026-08-13, commit `603c1dc`); step B in
progress. Written 2026-08-12, revised 2026-08-13 after checking what crocoddyl
and agimus actually do — see "What the reference stacks do", which corrects a
claim the first version of this plan made.

## The goal

Today the pregrasp OCP follows a *joint* trajectory. The next step is to
optimize directly against an **end-effector metric** — the OCP should be told
where the gripper must be, not which joint path to take to get there.

Long-term target: pick-and-place from (almost) any start configuration to
(almost) any goal object pose. That is what makes the EE metric worth the work
— `goal_q` is IK on the goal pose, and IK for an arbitrary target can fail,
pick a different branch, or land far from the current configuration.

## Plan of record

| step | content | state |
|---|---|---|
| A | cost registry: the yaml term list becomes the controller | **done**, `603c1dc` |
| B | `ee_translation` residual + goal on userdata | in progress |
| C | `ee_rotation` (SO(3) log) — needed for a real grasp | not started |
| D | `path_deviation` as an alternative to time-indexed tracking | not started |

Step A shipped `hydrax/cost_residuals.py` (the registry), the
`costs.running` / `costs.terminal` yaml schema, `doc/cost_residuals.md` and
`tests/test_cost_residuals.py`. It was a deliberate no-op on behaviour:
the cost matched the old formula to 9.65e-8, and the ROS sim reproduced the
baseline convergence curve (`pregrasp_mujoco_20260813T162515Z` vs
`...20260812T113718Z`: 10 mm at 6.1 s both, 1 mm at 39.5 vs 39.6 s, terminal
1.54 vs 1.46 mm).

## What the reference stacks do (verified from source, 2026-08-13)

**crocoddyl's canonical reach** (`minimal_examples_crocoddyl/ocp_kuka_reaching.py`):
`ResidualModelFrameTranslation` weight 10, `ResidualModelState` against a
*constant* `x0` weight 1e-1, `ResidualModelControlGrav` weight 1e-4; terminal
drops the control term. **dt 1e-2, T = 250 — a 2.5 s horizon.** The whole
motion fits inside it, which is why a fixed goal and no plan works there. Ours
is 8 x 0.04 = 0.32 s.

**agimus-controller** builds its OCP from yaml (`ocp_croco_generic.py`) and
feeds it `set_reference_weighted_trajectory()` — one `WeightedTrajectoryPoint`
per node, carrying both the reference *and* its own weights, written into
`ActivationModelWeightedQuad.weights` at runtime. It ships both a
`ocp_goal_reaching.yaml` (control_reg + state_reg + `ResidualModelFramePlacement`)
and a trajectory-tracking variant.

**Their deployed panda pick-and-place uses no frame cost at all**:
`ocp_definition_file.yaml` is control_reg + state_reg only, with
`w_q: 3.0, w_qdot: 0.12, w_qddot: 1e-6, w_robot_effort: 8e-4` and
**`w_pose: 0.0`**. It is pure joint-trajectory tracking — because the reference
comes from a global motion planner (HPP), so the trajectory is already
collision-free and feasible for arbitrary start/goal.

**This corrects the first version of this plan**, which claimed an EE cost
"matches the reference architecture". That is true of crocoddyl's minimal
example and false of agimus's deployed pick-and-place.

## Why the plan stays

Not legacy — it is what lets a 0.32 s horizon execute a 7.5 s motion, which is
the same division of labour agimus gets from HPP. Three concrete jobs:

1. **The only velocity ceiling.** `max_velocity_fraction: 0.20` caps the
   quintic's peak speed. Joint 2 tripped `joint_velocity_violation` on hardware
   once; a pure pose cost has no such guard.
2. **The warm start.** `initial_knots` is inverse dynamics along the plan.
3. **Rank.** The frame residual's Gauss-Newton curvature is `JᵀWJ`, rank <= 3
   for translation alone in a 7-DoF space. Take the joint weight to zero and
   the cost has a null space the sampler explores with no cost signal, where
   `K = du/dx0` is pure noise. A small-but-nonzero joint weight is what
   conditions the gain — the whole point of the project.

So: EE term owns *arrival* (the true task metric, correcting IK and model
error), joint+velocity terms own *transport* (rate, warm start, conditioning),
with most of the EE weight on the terminal node as crocoddyl does.

## Rejected, with the mechanism

**Using the planner as a sampling prior** (several plausible paths, MPPI
samples around those modes). Legitimate technique, wrong for this project:
`samples_delta = rollouts.controls[batch, 0] - rollouts.controls[0, 0]`
(`feedback_mppi.py`) measures `du` against the zero-noise nominal, so `K` is a
covariance around a *single* cloud. A multi-modal sampler makes it measure mode
separation instead of `du/dx0`. Given the gain's tail index of ~2, that is the
last thing to inject. Also moot for now: contacts are disabled in the planning
model, so there are no obstacles to have modes around.

## Where we are now (what the next session inherits)

The controller is F-MPPI in `hydrax` driving the FR3 through
`linear_feedback_controller` at 1 kHz, `planner_mode: exact_feedback`, the
bridge publishing at `1/dt`. The running cost is, in crocoddyl terms, a
weighted quadratic on three residuals (`hydrax/tasks/panda_pregrasp.py`):

| term | weight | residual |
|---|---|---|
| configuration | 10.0 | `q - q_ref(t)`, the quintic plan |
| velocity | 0.1 | `v` |
| control | 1e-4 | `u - qfrc_bias` |

`q_ref(t)` is a minimum-jerk joint plan from the measured `q0` to `goal_q`,
where `goal_q` is itself obtained by **IK on the goal pose** (`_solve_ik`). So
the task is already defined by an EE pose — it is just immediately projected
into joint space and then tracked. That projection is what this plan removes.

The plan travels to the cost as `mjx.Data.userdata` (a traced argument), so it
is re-parameterizable at runtime without recompiling — see `plan_userdata` /
`_plan_params`.

## Why change it

- **It over-constrains the redundancy.** IK picks one of infinitely many arm
  configurations reaching the pose; tracking it at a *tracking* weight forbids
  the solver from using the null space, which is the freedom MPPI could
  exploit.
- **It optimizes a proxy.** Joint error stands in for the task metric, which is
  gripper pose error. They disagree whenever the arm is off-path, and they
  disagree at convergence by exactly the IK and kinematic-model error — which
  is where the residual millimetres live.
- **It does not generalize to an arbitrary goal.** With a vision-driven target,
  `goal_q` must be re-solved per goal and can flip branch. If the joint term
  dominates, a bad branch fights the task.

## Design sketch (revised 2026-08-13)

**Not** "replace the configuration residual" — that was the first version of
this plan, before the horizon argument above. Add the frame residual and demote
the joint one:

- `ee_translation`: `r = p_gripper(q) - p_goal` (3), **fixed goal pose**, since
  the goal will come from vision. Moderate running weight, large terminal
  weight, following crocoddyl. A huge *running* weight on a nonlinear residual
  is what makes the sampled cost landscape spiky.
- `ee_rotation`: `r = log(R_goalᵀ R_ee)` (3), step C. The one genuinely new
  piece of maths (an SO(3) log in JAX). Required for a real grasp.
- Keep `joint_position_plan` at a **regularizer** weight, not a tracker weight,
  and keep `joint_velocity_plan` and `control_grav`.
- No separate posture regularizer is needed while the joint plan is present:
  past its end the plan holds `goal_q`, so the joint term *is* the posture
  regularizer. One becomes necessary only if step D removes the joint
  reference.

`trace_sites=["gripper"]` already puts the site position in the rollout
(`state.site_xpos[site_id]`), so the residual is cheap; the rotation term needs
`site_xmat`.

### Goal on userdata

`goal_pos` and `goal_rot` are construction constants today, so a new goal is a
new HLO and a full recompile — unusable for vision. `spec.nuserdata` is set by
the task itself, so the layout extends freely:

```
userdata = [q0 (nq) | duration (1) | goal_q (nq) | goal_pos (3)]
```

`goal_q` has to travel too, because `_reference` interpolates toward it; host
-side IK re-solves at the same place `set_plan_start` already latches. Anything
handed to `optimize` must be **device-committed** or the solver recompiles
mid-run.

## Risks to watch

- **Null-space drift / elbow wandering** — the posture regularizer above.
- **Velocity limiting.** The quintic currently caps plan speed at
  `max_velocity_fraction` of the joint limits. A pure pose cost has no such
  guard, and joint-2 already tripped `joint_velocity_violation` on hardware
  once. Keep an explicit velocity cost or limit.
- **What `K` means.** F-MPPI gains stay `du/dx` in joint space; only the cost
  changes. Verify the gain magnitudes do not shift (`gain_norm`, `gain_ess`).
- **The steady-state friction offset does not go away** — it is an actuation
  problem, independent of the cost.

## Facts already established (do not re-derive)

- **Validation is the ROS launch, only.** See `docs/HANDOFF.md` §8 and the
  metric table there. Strip the JIT lead-in and frozen post-reflex rows before
  comparing runs.
- `with_frictionloss: true` at 25 Hz beats frictionless + friction feedforward
  at 50 Hz on every metric — see [[sbmpc-friction-feedforward-works]].
- The plan anchors on the measured configuration at arming
  (`_plan_anchored`, `set_plan_start`); it used to always start at `start_q`.
- Everything handed to `optimize` must be **device-committed** or JAX
  recompiles the solver mid-run.
