"""The residuals a task's cost can be assembled from, selectable by name.

The cost is crocoddyl's ``CostModelSum``: a sum of weighted, activated
squared residuals,

    J = Σ_terms  a( Σ_i w_i · r_i² )

with one term per entry of the ``costs.running`` / ``costs.terminal`` blocks
of the tuning yaml. Each entry names a residual from ``RESIDUALS`` below and
gives its weight; that pairing is the whole controller definition, so a
different controller is a yaml edit rather than a code change.

Two rules make an entry readable without opening this file:

* **the name states where the reference comes from.** A ``_plan`` suffix
  means the time-varying reference carried on ``mjx.Data.userdata``; the
  unsuffixed name means a constant that the yaml must give explicitly. There
  is no default reference — regularizing toward a value nobody chose is how
  a cost silently stops meaning what its name says.
* **the weight may be a scalar or a per-component vector.** The vector form
  is crocoddyl's ``ActivationModelWeightedQuad`` (the weights live inside the
  quadratic form), which is what lets a null-space joint be weighted down
  without touching the rest of the term.

``doc/cost_residuals.md`` carries the same table for readers who are picking
weights rather than editing code.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.task_base import Task

# Activation applied to each term's weighted squared residual — crocoddyl's
# pluggable ActivationModel, in the two forms this OCP has used. Selected once
# for the whole cost by `costs.activation`.
ACTIVATIONS = {
    # ActivationModelWeightedQuad: a(r) = 0.5 rᵀ diag(w) r, constant Hessian
    "quadratic": lambda weighted_square: 0.5 * weighted_square,
    # Bounds each term to [0, 1); its curvature -> 0 as the error grows
    "saturated": lambda weighted_square: 1.0 - jnp.exp(-weighted_square),
}


@dataclass(frozen=True)
class ResidualSpec:
    """One residual available to the cost, and everything needed to check it.

    Attributes:
        formula: The residual in maths, for error messages and the doc.
        dim: Length of r, as a model attribute name ("nq", "nv", "nu") or an
             explicit int. Resolved against the task's model at build time,
             and what a yaml weight/reference vector must match.
        reference: Where the reference comes from — "yaml" (a constant the
             yaml must supply), "plan" (time-varying, from userdata), "state"
             (derived from the state itself) or "goal" (from userdata).
        uses_control: True if r depends on the control, which makes the term
             illegal in the terminal block: there is no control to regularize
             at the terminal node, and crocoddyl drops it there for the same
             reason.
        fn: (task, state, control, reference) -> r.
    """

    formula: str
    dim: str | int
    reference: str
    uses_control: bool
    fn: "Residual"


# Every residual takes the same arguments so the registry can call them
# uniformly: the task (for its model constants and its plan), the state, the
# control (None at the terminal node) and the constant reference from the yaml
# (None when the residual has its own source). Each uses what it needs.
Residual = Callable[
    [Task, mjx.Data, Optional[jax.Array], Optional[jax.Array]], jax.Array
]


def _joint_position(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    return state.qpos - reference


def _joint_position_plan(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    q_ref, _ = task._reference(state)
    return state.qpos - q_ref


def _joint_velocity(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    return state.qvel - reference


def _joint_velocity_plan(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    _, v_ref = task._reference(state)
    return state.qvel - v_ref


def _control(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    return (control - reference) / task.tau_max


def _control_grav(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    return (control - state.qfrc_bias) / task.tau_max


def _ee_translation(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    return state.site_xpos[task.gripper_site_id] - task._goal_pos(state)


# Floor on |v|^2 below. Two jobs, both about the differentiated path rather
# than the forward value: jnp.linalg.norm has a NaN gradient at zero, and the
# division below is 0/0 at zero rotation. Squared, so it is (1e-12)^2.
_LOG_EPS_SQ = 1e-24


def so3_log(rotation: jax.Array) -> jax.Array:
    """Rotation vector of a rotation matrix — pinocchio/crocoddyl ``log3``.

    Returns ``theta * axis`` (3,), the geodesic error on SO(3).

    Written as ``theta * v / |v|`` where ``v = vee(skew(R)) = sin(theta)*axis``,
    rather than the textbook ``theta/(2 sin theta) * v``. The two agree
    exactly, but this form keeps the MAGNITUDE equal to ``theta`` by
    construction instead of leaving it to a small-angle series, so it is right
    at theta = 0 with no Taylor branch and it degrades gracefully rather than
    silently reporting zero error at a large one.

    ``theta`` comes from ``arctan2(|v|, cos theta)``, not ``arccos``: arccos is
    ill-conditioned exactly where this residual spends its life (small errors,
    trace near 3) and its derivative is infinite there.

    **Limit:** within ~1e-6 rad of theta = pi, ``v`` underflows the floor and
    the axis direction is lost (the magnitude degrades toward 0 rather than
    pi). That neighbourhood is the genuine singularity of log3 — pinocchio
    special-cases it via the symmetric part of R. Resolving it here would need
    a Shepperd-style quaternion branch; worth adding if orientation errors near
    180 degrees ever become reachable for this task.
    """
    v = 0.5 * jnp.stack(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    # Floored BEFORE the sqrt: sqrt has an infinite derivative at 0, so
    # clamping afterwards would still produce a NaN gradient at zero rotation.
    norm = jnp.sqrt(jnp.maximum(jnp.sum(v * v), _LOG_EPS_SQ))
    cos_theta = 0.5 * (jnp.trace(rotation) - 1.0)
    theta = jnp.arctan2(norm, cos_theta)
    return theta * v / norm


def _ee_rotation(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    rotation = state.site_xmat[task.gripper_site_id]
    return so3_log(task._goal_rot(state).T @ rotation)


# name -> spec. The single source of what a yaml may ask for; an unknown name
# is an error listing this table, so a typo can never become a silent no-op.
RESIDUALS: Dict[str, ResidualSpec] = {
    "joint_position": ResidualSpec(
        formula="q - q_ref",
        dim="nq",
        reference="yaml",
        uses_control=False,
        fn=_joint_position,
    ),
    "joint_position_plan": ResidualSpec(
        formula="q - q_ref(t)",
        dim="nq",
        reference="plan",
        uses_control=False,
        fn=_joint_position_plan,
    ),
    "joint_velocity": ResidualSpec(
        formula="v - v_ref",
        dim="nv",
        reference="yaml",
        uses_control=False,
        fn=_joint_velocity,
    ),
    "joint_velocity_plan": ResidualSpec(
        formula="v - v_ref(t)",
        dim="nv",
        reference="plan",
        uses_control=False,
        fn=_joint_velocity_plan,
    ),
    # Both control residuals are divided by tau_max so the strong shoulder
    # joints (87 N.m) and the wrist (12 N.m) compare — equivalently a weighted
    # quadratic with w_i = 1/tau_max_i². A yaml `reference` is in N.m, before
    # that scaling.
    "control": ResidualSpec(
        formula="(u - u_ref) / tau_max",
        dim="nu",
        reference="yaml",
        uses_control=True,
        fn=_control,
    ),
    # crocoddyl's ResidualModelControlGrav, up to one difference: qfrc_bias is
    # MuJoCo's full bias force C(q,v)v + g(q), where crocoddyl uses
    # computeGeneralizedGravity(q) alone. Measured along this task's plan the
    # two differ by at most 0.048% of tau_max.
    "control_grav": ResidualSpec(
        formula="(u - qfrc_bias) / tau_max",
        dim="nu",
        reference="state",
        uses_control=True,
        fn=_control_grav,
    ),
    # crocoddyl's ResidualModelFrameTranslation on the gripper site. The goal
    # is read from userdata, not from a construction constant, so a goal that
    # moves (vision) does not produce a new HLO and a full recompile.
    #
    # It is 3 rows in a 7-DoF space, so its Gauss-Newton curvature J^T W J has
    # rank <= 3: on its own it leaves a 4-dimensional null space the sampler
    # explores with NO cost signal, where K = du/dx0 is noise. Keep a
    # non-zero joint or posture term alongside it -- that is what conditions
    # the gain, which is the point of the whole project.
    "ee_translation": ResidualSpec(
        formula="p_gripper(q) - p_goal",
        dim=3,
        reference="goal",
        uses_control=False,
        fn=_ee_translation,
    ),
    # crocoddyl's ResidualModelFrameRotation. Together with ee_translation this
    # is ResidualModelFramePlacement, split so the two can be weighted apart --
    # they have different units (m vs rad) and different task tolerances.
    #
    # Orientation is NOT optional for a grasp: with ee_translation alone it is
    # held only implicitly by the joint term, and dropping that term's weight
    # 10 -> 1 measured terminal orientation 2x worse and a 0.291 rad excursion
    # mid-reach (2026-08-13, pregrasp_mujoco_20260813T164721Z).
    "ee_rotation": ResidualSpec(
        formula="log3(R_goal^T R_gripper)",
        dim=3,
        reference="goal",
        uses_control=False,
        fn=_ee_rotation,
    ),
}


@dataclass(frozen=True)
class CostTerm:
    """A residual name paired with the weight (and reference) from the yaml.

    Structurally validated at load; the lengths are checked at build time,
    where the model dimensions are known.
    """

    name: str
    weight: Tuple[float, ...]
    reference: Optional[Tuple[float, ...]] = None
    epsilon: Optional[float] = None


# What build_cost_terms hands to cost_sum: per term, the broadcast weight
# vector, the constant reference (None unless the yaml gave one) and the
# residual itself. Resolved once at construction, so the compiled cost
# contains exactly the terms the yaml asked for.
BuiltTerms = Tuple[
    Tuple[jax.Array, Optional[jax.Array], Optional[float], "Residual"], ...
]


def _as_tuple(value: object, what: str, where: str) -> Tuple[float, ...]:
    if isinstance(value, bool):
        raise ValueError(f"{where}: {what} must be a number or a list")
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(float(v) for v in value)
    raise ValueError(f"{where}: {what} must be a number or a list")


def parse_cost_terms(block: object, label: str) -> Tuple[CostTerm, ...]:
    """Parse one ``costs.running`` / ``costs.terminal`` yaml block.

    An entry is either the weight on its own::

        control_grav: 0.0001
        joint_position_plan: [10, 10, 10, 10, 10, 10, 1.0]

    or a mapping when a constant reference is needed::

        joint_position:
          weight: 1.0
          reference: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    Everything that could silently change what a term means is an error here:
    an unknown name, a missing reference on a residual that has no other
    source, and a reference on one that already has a source.
    """
    if block is None:
        return ()
    if not isinstance(block, dict):
        raise ValueError(
            f"'{label}' must be a mapping of residual name -> weight"
        )

    terms = []
    for name, entry in block.items():
        where = f"'{label}.{name}'"
        if name not in RESIDUALS:
            raise ValueError(
                f"{where}: unknown residual; expected one of "
                f"{sorted(RESIDUALS)}"
            )
        spec = RESIDUALS[name]

        epsilon = None
        if isinstance(entry, dict):
            unknown = set(entry) - {"weight", "reference", "epsilon"}
            if unknown:
                raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
            if "weight" not in entry:
                raise ValueError(f"{where}: missing 'weight'")
            weight = _as_tuple(entry["weight"], "weight", where)
            reference = entry.get("reference")
            if "epsilon" in entry:
                epsilon = float(entry["epsilon"])
                if epsilon <= 0.0:
                    raise ValueError(
                        f"{where}: epsilon must be > 0 (it is the SQUARE of "
                        f"the residual value where the cost turns quadratic)"
                    )
        else:
            weight = _as_tuple(entry, "weight", where)
            reference = None

        if spec.reference == "yaml":
            if reference is None:
                hint = (
                    f" Use '{name}_plan' for the time-varying plan reference."
                    if f"{name}_plan" in RESIDUALS
                    else ""
                )
                raise ValueError(
                    f"{where}: residual '{name}' (r = {spec.formula}) has no "
                    f"reference of its own, so 'reference' is required.{hint}"
                )
            reference = _as_tuple(reference, "reference", where)  # noqa: PLW2901
        elif reference is not None:
            raise ValueError(
                f"{where}: residual '{name}' takes its reference from "
                f"'{spec.reference}', so 'reference' must not be set — two "
                f"sources for one reference is how a cost stops meaning "
                f"its name."
            )

        terms.append(
            CostTerm(
                name=name,
                weight=weight,
                reference=reference,
                epsilon=epsilon,
            )
        )
    return tuple(terms)


def build_cost_terms(
    task: Task,
    terms: Sequence[CostTerm],
    label: str,
    allow_control: bool,
) -> "BuiltTerms":
    """Resolve parsed terms against a task into (weight, reference, fn) triples.

    Vector lengths are checked here, where the model dimensions are known; a
    scalar weight is broadcast to the residual's length so the cost is one
    expression for both forms.
    """
    built = []
    for term in terms:
        spec = RESIDUALS[term.name]
        where = f"'{label}.{term.name}'"
        dim = (
            spec.dim
            if isinstance(spec.dim, int)
            else getattr(task.model, spec.dim)
        )

        if spec.uses_control and not allow_control:
            raise ValueError(
                f"{where}: '{term.name}' depends on the control, which does "
                f"not exist at the terminal node; crocoddyl drops its control "
                f"regularization there for the same reason."
            )

        if len(term.weight) == 1:
            weight = jnp.full((dim,), term.weight[0], dtype=jnp.float32)
        elif len(term.weight) == dim:
            weight = jnp.asarray(term.weight, dtype=jnp.float32)
        else:
            raise ValueError(
                f"{where}: weight has {len(term.weight)} entries, expected 1 "
                f"or {dim} (the length of r = {spec.formula})"
            )

        reference = None
        if term.reference is not None:
            if len(term.reference) != dim:
                raise ValueError(
                    f"{where}: reference has {len(term.reference)} entries, "
                    f"expected {dim} (the length of r = {spec.formula})"
                )
            reference = jnp.asarray(term.reference, dtype=jnp.float32)

        built.append((weight, reference, term.epsilon, spec.fn))
    return tuple(built)


def cost_sum(
    task: Task,
    built: "BuiltTerms",
    state: mjx.Data,
    control: Optional[jax.Array],
    activation: str,
) -> jax.Array:
    """Sum the built terms, each either quadratic or smooth-L1.

    A term with no ``epsilon`` uses the global activation on its weighted
    squared residual, ``a(Σ_i w_i r_i²)``. The weights sit INSIDE the
    activation, crocoddyl's ActivationModelWeightedQuad convention: for
    "quadratic" that is identical to a scalar weight outside it; for
    "saturated" it is not, since that form bounds each term to [0, 1)
    whatever its weight.

    A term WITH ``epsilon`` uses crocoddyl's ActivationModelSmooth1Norm,
    ``Σ_i w_i (√(r_i² + ε) − √ε)``, per component. The subtraction only makes
    the term vanish at r = 0 so costs stay comparable; it changes no gradient.

    The point of the smooth-L1 form is the gradient: ``w·r/√(r² + ε)``, which
    tends to a CONSTANT w for |r| ≫ √ε instead of decaying to zero like the
    quadratic's ``w·r``. A quadratic cost's pull fades as the error shrinks,
    so below some radius it can no longer command the breakaway torque and the
    joint stops -- measured on hardware at |tau - gravity| / breakaway = 0.62
    on every run. A constant pull has no such radius; what replaces it is √ε,
    which is chosen rather than imposed by friction.
    """
    activate = ACTIVATIONS[activation]
    total = jnp.zeros(())
    for weight, reference, epsilon, fn in built:
        residual = fn(task, state, control, reference)
        if epsilon is None:
            total = total + activate(jnp.sum(weight * jnp.square(residual)))
        else:
            smooth_abs = jnp.sqrt(jnp.square(residual) + epsilon)
            total = total + jnp.sum(
                weight * (smooth_abs - jnp.sqrt(epsilon))
            )
    return total
