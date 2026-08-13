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

Planned, documented here so the names are fixed before they exist:

| yaml name | residual `r` | dim | reference from | status |
|---|---|---|---|---|
| `ee_translation` | `p_ee(q) − p_goal` | 3 | goal on userdata | step B |
| `ee_rotation` | `log(R_goalᵀ · R_ee)` | 3 | goal on userdata | not implemented |
| `path_deviation` | `q − proj_[q₀,goal](q)` | nq (rank nq−1) | plan segment on userdata | not implemented |

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
