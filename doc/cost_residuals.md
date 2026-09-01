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

The **name** says what is measured; **`reference`** says against what. They are
orthogonal, so every term states both.

| yaml name | residual `r` | dim | `reference:` | crocoddyl equivalent |
|---|---|---|---|---|
| `joint_position` | `q − q_ref` | nq | `plan` or a vector | `ResidualModelState` (q part) |
| `joint_velocity` | `v − v_ref` | nv | `plan` or a vector | `ResidualModelState` (v part) |
| `control` | `(u − u_ref) / τ_max` | nu | `gravity` or a vector | `ResidualModelControlGrav` / `ResidualModelControl` |
| `ee_translation` | `p_gripper(q) − p_goal` | 3 | `goal` | `ResidualModelFrameTranslation` |
| `ee_rotation` | `log3(R_goalᵀ · R_gripper)` | 3 | `goal` | `ResidualModelFrameRotation` |

Reference sources:

| keyword | meaning |
|---|---|
| `plan` | the time-varying quintic on `userdata`; for `joint_velocity` that is the bell-shaped velocity profile, exactly zero once the plan ends |
| `gravity` | `qfrc_bias`, MuJoCo's full bias force `C(q,v)v + g(q)` |
| `goal` | the goal pose on `userdata`, so vision can move it without recompiling |
| *a vector* | a constant written inline, e.g. `reference: [0, -0.785, 0, -2.356, 0, 1.571, 0.785]` |

## Activations

Per **term**, as in crocoddyl where a cost is
`CostModelResidual(state, activation, residual)`. There is no problem-wide
activation.

| `activation:` | formula | crocoddyl |
|---|---|---|
| `quadratic` *(default)* | `0.5 Σ wᵢ rᵢ²` | `ActivationModelWeightedQuad` |
| `smooth_l1` | `Σ wᵢ (√(rᵢ² + knee²) − knee)` | `ActivationModelSmooth1Norm` |
| `saturated` | `1 − exp(−Σ wᵢ rᵢ²)` | — |

`smooth_l1` requires **`knee`**, in the residual's own units (m, rad, or
N·m/τ_max). Below it the term is quadratic and the arm settles; above it the
pull is a constant `w`. **It is the accuracy you choose to stop at**, so it has
no default.

Why it matters: a quadratic pull is `w·r`, which fades as the error shrinks, so
below some radius it can no longer command breakaway torque and the joint
stops — measured on hardware at `|τ − gravity| / breakaway = 0.62` on *every*
run. The smooth-L1 pull is `w·r/√(r² + knee²)`, constant above the knee, so
there is no friction-set stall radius; what replaces it is the knee, which is
chosen. Scale-matched at `w = 1.4` this gave 2.9× *less* drive at arming (a
smaller lunge) and 8.4× *more* at the stall radius.

`saturated` is kept selectable but not recommended: its curvature vanishes as
the error grows, which is what once made this OCP blind to the control —
measured `d²J/du² ≈ 1e-4` with three negative entries where the real FR3
parked.

Planned, documented here so the names are fixed before they exist:

| yaml name | residual `r` | dim | status |
|---|---|---|---|
| `path_deviation` | `q − proj_[q₀,goal](q)` | nq (rank nq−1) | not implemented |

`ee_translation` + `ee_rotation` together are crocoddyl's
`ResidualModelFramePlacement`, kept as two entries so they can be weighted
apart — they have different units (m vs rad) and different task tolerances.

### Notes per residual

- **`reference: plan`** is the quintic on `mjx.Data.userdata`, evaluated in
  closed form at `state.time` and re-anchored on the measured configuration at
  arming without recompiling.
- **`control` divides by `τ_max`**, so the strong shoulder joints (87 N·m) and
  the wrist (12 N·m) compare — equivalently a weighted quadratic with
  `w_i = 1/τ_max_i²`. A constant `reference` is in N·m, before that scaling.
  With `reference: gravity` it is crocoddyl's `ResidualModelControlGrav` up to
  one difference: `qfrc_bias` is MuJoCo's full bias force `C(q,v)v + g(q)`,
  where crocoddyl uses `computeGeneralizedGravity(q)` alone. Measured along
  this task's plan the two differ by at most 0.048% of `τ_max`.
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

Every entry states its weight, its reference and its activation:

```yaml
joint_position: {weight: 1.0, reference: plan, activation: quadratic}
control: {weight: 1e-4, reference: gravity, activation: quadratic}
ee_translation:
  weight: 1.4
  reference: goal
  activation: smooth_l1
  knee: 0.001                       # metres
```

`reference` is either a source keyword or a constant vector written inline:

```yaml
joint_position:
  weight: 1.0
  reference: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
  activation: quadratic
```

A bare number is accepted as weight-only shorthand, but only for a residual
with exactly one possible source, so nothing is ambiguous.

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

- an unknown residual name, activation, or reference source — each error lists
  the valid ones;
- a missing `reference` on a residual with more than one possible source;
- a constant vector on a residual that accepts none (the `ee_*` terms);
- `knee` on an activation that does not use one, or a missing `knee` on
  `smooth_l1`;
- a non-positive `knee`;
- unknown keys inside a term mapping.

At task construction (`build_cost_terms`), where the model dimensions are
known:

- a weight or reference vector of the wrong length;
- a control-dependent residual in `costs.terminal`. There is no control at the
  terminal node to regularize, and crocoddyl's terminal `CostModelSum` drops
  its control regularization for the same reason.

A term the yaml omits is absent from the compiled graph, not multiplied by
zero inside it.

## Reference configurations

The controller that ran on hardware 2026-08-31 (7.27 mm settled, recovering
from a 29 mm disturbance in 0.3 s):

```yaml
running:
  joint_position: {weight: 1.0, reference: plan, activation: quadratic}
  joint_velocity: {weight: 0.1, reference: plan, activation: quadratic}
  control: {weight: 0.0001, reference: gravity, activation: quadratic}
  ee_translation:
    {weight: 1.4, reference: goal, activation: smooth_l1, knee: 0.001}
  ee_rotation:
    {weight: 0.3, reference: goal, activation: smooth_l1, knee: 0.01}
terminal: (the same, without `control`)
```

The joint-trajectory tracker that still holds the best hardware number
(4.30 mm settled, `pregrasp_real_20260811T085838Z_REFERENCE`):

```yaml
running:
  joint_position: {weight: 10.0, reference: plan, activation: quadratic}
  joint_velocity: {weight: 0.1, reference: plan, activation: quadratic}
  control: {weight: 0.0001, reference: gravity, activation: quadratic}
terminal: (the same, without `control`)
```

Crocoddyl's canonical reach (`ocp_kuka_reaching.py`, weights 10 / 1e-1 / 1e-4)
— note its state regularization references a *constant* `x₀`, not a trajectory:

```yaml
running:
  ee_translation: {weight: 10.0, reference: goal, activation: quadratic}
  joint_position:
    {weight: 0.1, reference: [...x0 q...], activation: quadratic}
  joint_velocity:
    {weight: 0.1, reference: [0, 0, 0, 0, 0, 0, 0], activation: quadratic}
  control: {weight: 0.0001, reference: gravity, activation: quadratic}
```
