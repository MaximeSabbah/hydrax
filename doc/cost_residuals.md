# Cost residuals — what a controller can be built from

The OCP cost is crocoddyl's `CostModelSum`: a sum of weighted, activated
squared residuals,

```
J  =  Σ_terms  a( Σ_i w_i · r_i² )
```

with one term per entry of the `costs.running` / `costs.terminal` blocks in
`hydrax/configs/pregrasp.yaml`. **The term list is the controller.** Changing
which residuals appear there changes what is optimized, with no code change.

The registry is `hydrax/cost_residuals.py`; this file is the same table
for someone choosing weights rather than editing code. The formulas are
repeated in each residual's `ResidualSpec`, so an error message and this doc
cannot drift apart.

## The residuals

| yaml name | residual `r` | dim | reference from | crocoddyl equivalent |
|---|---|---|---|---|
| `joint_position` | `q − q_ref` | nq | `reference:` (required) | `ResidualModelState` (q part) |
| `joint_position_plan` | `q − q_ref(t)` | nq | plan on userdata | `ResidualModelState`, per-node ref |
| `joint_velocity` | `v − v_ref` | nv | `reference:` (required) | `ResidualModelState` (v part) |
| `joint_velocity_plan` | `v − v_ref(t)` | nv | plan on userdata | `ResidualModelState`, per-node ref |
| `control` | `(u − u_ref) / τ_max` | nu | `reference:` (required) | `ResidualModelControl` |
| `control_grav` | `(u − qfrc_bias) / τ_max` | nu | the state | `ResidualModelControlGrav` |
| `ee_translation` | `p_gripper(q) − p_goal` | 3 | goal on userdata | `ResidualModelFrameTranslation` |
| `ee_rotation` | `log3(R_goalᵀ · R_gripper)` | 3 | goal on userdata | `ResidualModelFrameRotation` |

Planned, documented here so the names are fixed before they exist:

| yaml name | residual `r` | dim | reference from | status |
|---|---|---|---|---|
| `path_deviation` | `q − proj_[q₀,goal](q)` | nq (rank nq−1) | plan segment on userdata | not implemented |

`ee_translation` + `ee_rotation` together are crocoddyl's
`ResidualModelFramePlacement`, kept as two entries so they can be weighted
apart — they have different units (m vs rad) and different task tolerances.

### Notes per residual

- **`_plan` suffix** means the reference is the time-varying quintic joint
  plan, carried on `mjx.Data.userdata` as `(q₀, duration)` and evaluated in
  closed form at `state.time`. It is re-anchored on the measured configuration
  at arming without recompiling.
- **Both control residuals divide by `τ_max`**, so the strong shoulder joints
  (87 N·m) and the wrist (12 N·m) compare — equivalently a weighted quadratic
  with `w_i = 1/τ_max_i²`. A yaml `reference` for `control` is in N·m, i.e.
  before that scaling.
- **`control_grav`** is crocoddyl's `ResidualModelControlGrav` up to one
  difference: `qfrc_bias` is MuJoCo's full bias force `C(q,v)v + g(q)`, where
  crocoddyl uses `computeGeneralizedGravity(q)` alone. Measured along this
  task's plan the two differ by at most 0.048% of `τ_max`.
- **`ee_translation`** is the task metric itself, and the only residual whose
  value is not a joint quantity. It reads `state.site_xpos[gripper]`, which
  `trace_sites=["gripper"]` already puts in the rollout, so it costs a lookup
  rather than a kinematics pass. Its goal comes from userdata so a moving
  (vision) target does not recompile. **It is 3 rows in a 7-DoF space**, so its
  Gauss-Newton curvature `JᵀWJ` has rank ≤ 3 and leaves a 4-dimensional null
  space with no cost signal; keep a non-zero joint or posture term alongside
  it, or `K = du/dx₀` is noise in those directions.
- **`ee_rotation`** is the geodesic error on SO(3), `θ·axis`. It is **not**
  optional for a grasp: with `ee_translation` alone, orientation is held only
  implicitly by the joint term, and dropping that term's weight 10 → 1 measured
  terminal orientation 2× worse with a 0.291 rad excursion mid-reach
  (`pregrasp_mujoco_20260813T164721Z`, 2026-08-13).

  Two properties worth knowing, both measured in **float32**, the dtype the
  rollout actually runs in.

  **Its Jacobian diverges as `1/(π−θ)`.** Intrinsic to `log3`, not to this
  implementation — pinocchio's `Jlog3` does the same. Measured `max|∂log/∂R|`:
  0.5 at θ ≤ 1 rad, 1.1 at 2 rad, 10.6 at 3 rad, 1.7e4 within 1e-4 of π.
  Harmless at this task's operating point (errors ≤ 0.3 rad → 0.51), but a
  route to a large `∂J/∂x₀` if orientation errors near 180° become reachable.

  **Accuracy degrades smoothly toward θ = π**, it does not explode: worst error
  against `scipy.as_rotvec` is 4.8e-7 rad at θ ≤ 3.0, 1.7e-6 at π−0.04,
  5.6e-5 at π−1.6e-3, 8.5e-4 at π−9.3e-5. Within ~1e-6 rad of π the axis is
  lost outright and the magnitude falls toward 0 instead of π — the genuine
  singularity, which pinocchio special-cases via the symmetric part of R.
- **`path_deviation`** will penalize distance to the *path* rather than to the
  time-indexed trajectory, so falling behind schedule costs nothing. Because
  the plan is a straight segment in joint space the projection is closed-form
  (`s* = clip(((q−q₀)·d)/‖d‖², 0, 1)`, `d = goal_q − q₀`), needing neither a
  search nor an augmented state. Its residual spans the subspace orthogonal to
  `d`, leaving progress along the path unconstrained.

## Writing an entry

An entry is either the weight on its own:

```yaml
control_grav: 0.0001
joint_position_plan: [10, 10, 10, 10, 10, 10, 1.0]   # per-joint weights
```

or a mapping, when the residual needs a constant reference:

```yaml
joint_position:
  weight: 1.0
  reference: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
```

**Weights** may be a scalar (broadcast to every component) or a vector of the
residual's length. The vector form is crocoddyl's `ActivationModelWeightedQuad`
— the weights live *inside* the quadratic form — which is what lets a
null-space joint be weighted down without touching the rest of the term. Joint
7 is the concrete case: it contributes 0% of this task's end-effector error,
being wrist roll in the null space.

**References** have no defaults. A residual whose reference must come from the
yaml requires it explicitly; regularizing toward a value nobody chose is how a
cost quietly stops meaning what its name says.

## What is rejected, and when

At load (`parse_cost_terms`):

- an unknown residual name — the error lists the registry;
- a missing `reference` on a residual that has no other source;
- a `reference` on a residual that already has one (`_plan`, `control_grav`,
  the `ee_*` terms) — two sources for one reference is exactly the confusion
  this naming scheme exists to prevent;
- unknown keys inside a term mapping.

At task construction (`build_cost_terms`), where the model dimensions are
known:

- a weight or reference vector of the wrong length;
- a control-dependent residual in `costs.terminal`. There is no control at the
  terminal node to regularize, and crocoddyl's terminal `CostModelSum` drops
  its control regularization for the same reason.

A term the yaml omits is absent from the compiled graph, not multiplied by
zero inside it.

## Activation

`costs.activation` selects the activation for every term:

- `quadratic` → `0.5 · Σ w_i r_i²`, crocoddyl's `ActivationModelWeightedQuad`,
  constant Hessian. The default, and what the committed weights are tuned for.
- `saturated` → `1 − exp(−Σ w_i r_i²)`, bounded to [0, 1). Available, but its
  curvature vanishes as the error grows: measured at the state where the real
  FR3 parked, `d²J/du²` was ~1e-4 **with three negative entries**, so the
  optimizer could not see the control at all. Selecting it requires retuning
  the weights and `solver.temperature` together.

Note the weights sit inside the activation. For `quadratic` that is identical
to a scalar weight outside it; for `saturated` it is not, and inside is the
only position where a weight changes that term's influence at all.

## Reference configurations

Today's committed controller — a joint-trajectory tracker, 1.6 mm terminal in
sim:

```yaml
running:
  joint_position_plan: 10.0
  joint_velocity_plan: 0.1
  control_grav: 0.0001
terminal:
  joint_position_plan: 10.0
  joint_velocity_plan: 0.1
```

Crocoddyl's canonical reach (`ocp_kuka_reaching.py`, weights 10 / 1e-1 / 1e-4)
becomes expressible directly once `ee_translation` exists — note its state
regularization references a *constant* `x₀`, not a trajectory:

```yaml
running:
  ee_translation: 10.0
  joint_position: {weight: 0.1, reference: [...x0 q...]}
  joint_velocity: {weight: 0.1, reference: [0, 0, 0, 0, 0, 0, 0]}
  control_grav: 0.0001
terminal:
  ee_translation: 10.0
  joint_position: {weight: 0.1, reference: [...x0 q...]}
  joint_velocity: {weight: 0.1, reference: [0, 0, 0, 0, 0, 0, 0]}
```
