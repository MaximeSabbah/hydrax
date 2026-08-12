# From trajectory following to an end-effector metric

Status: **not started**. Written 2026-08-12 as the pick-up point for the next
working session, so it can begin without re-deriving the current state.

## The goal

Today the pregrasp OCP follows a *joint* trajectory. The next step is to
optimize directly against an **end-effector metric** — the OCP should be told
where the gripper must be, not which joint path to take to get there.

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
  configurations reaching the pose; tracking it forbids the solver from using
  the null space, which is exactly the freedom MPPI could exploit.
- **It optimizes the wrong thing.** Joint error is a proxy; the task metric is
  gripper pose error. They disagree whenever the arm is off-path.
- **It matches the reference architecture.** Crocoddyl's canonical reach uses
  `ResidualModelFrameTranslation` / `ResidualModelFramePlacement` on the EE
  frame plus state/control regularization — not a joint trajectory.

## Design sketch

Replace the configuration residual with a frame residual at the `gripper`
site, keeping the regularizers:

- `r_pos = p_gripper(q) - p_goal` (3), and `r_rot` as the log of the relative
  rotation (3), weighted separately — the task already computes both
  (`_ee_pose`, `goal_pos`, `goal_rot`, and the reported
  `position_error`/`orientation_error`).
- Keep velocity and control regularization; **add a posture regularizer**
  (small weight on `q - q_nominal`) to stop null-space drift once the joint
  reference is gone.
- Terminal cost weights the frame residual much higher than the running one.
- Decide what remains time-varying: either a pose *trajectory* (EE-space
  min-jerk, keeping the smooth ramp that currently limits velocity) or a
  constant goal pose with the horizon doing the shaping. The first is the
  smaller step and keeps `max_velocity_fraction`'s safety role.

`trace_sites=["gripper"]` already makes the site position available inside the
rollout, so the residual is cheap; the rotation term needs the site xmat.

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
