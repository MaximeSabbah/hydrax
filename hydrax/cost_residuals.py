"""The residuals a task's cost can be assembled from, selectable by name.

The cost is crocoddyl's ``CostModelSum``: a sum of weighted, activated
residuals, one per entry of the ``costs.running`` / ``costs.terminal`` blocks
of the tuning yaml. That term list is the whole controller definition, so a
different controller is a yaml edit rather than a code change.

Three rules make an entry readable without opening this file:

* **the name says WHAT is measured, ``reference`` says AGAINST WHAT.** The two
  are orthogonal: ``reference`` is either a source keyword (``plan``,
  ``goal``, ``gravity``) or a constant vector written straight into the yaml.
  It is always required — regularizing toward a value nobody chose is how a
  cost silently stops meaning what its name says.
* **the activation belongs to the term**, as it does in crocoddyl, where a
  cost is ``CostModelResidual(state, activation, residual)``. There is no
  global activation: reading one entry tells you that entry's shape.
* **the weight may be a scalar or a per-component vector.** The vector form
  is crocoddyl's ``ActivationModelWeightedQuad`` (the weights live inside the
  activation), which is what lets one joint be weighted differently from the
  rest of the term.

``doc/cost_residuals.md`` carries the same tables for readers picking weights
rather than editing code.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.task_base import Task


def _quadratic(
    weight: jax.Array, residual: jax.Array, knee: Optional[float]
) -> jax.Array:
    """0.5 Σ w_i r_i² — crocoddyl's ActivationModelWeightedQuad."""
    return 0.5 * jnp.sum(weight * jnp.square(residual))


def _saturated(
    weight: jax.Array, residual: jax.Array, knee: Optional[float]
) -> jax.Array:
    """1 − exp(−Σ w_i r_i²), bounded to [0, 1).

    Its curvature vanishes as the error grows, which is what once made this
    OCP blind to the control: measured where the real FR3 parked, d²J/du² was
    ~1e-4 with three NEGATIVE entries. Kept selectable, not recommended.
    """
    return 1.0 - jnp.exp(-jnp.sum(weight * jnp.square(residual)))


def _smooth_l1(
    weight: jax.Array, residual: jax.Array, knee: Optional[float]
) -> jax.Array:
    """Σ w_i (√(r_i² + knee²) − knee) — crocoddyl's ActivationModelSmooth1Norm.

    Quadratic below ``knee``, linear above it, analytic everywhere (no branch).
    The point is the gradient, ``w·r/√(r² + knee²)``: it tends to a CONSTANT w
    above the knee instead of decaying to zero like the quadratic's ``w·r``. A
    quadratic pull fades as the error shrinks, so below some radius it can no
    longer command the breakaway torque and the joint stops — measured on
    hardware at |tau − gravity| / breakaway = 0.62 on every run. A constant
    pull has no such radius; what replaces it is ``knee``, which is chosen
    rather than imposed by friction.
    """
    return jnp.sum(
        weight * (jnp.sqrt(jnp.square(residual) + knee**2) - knee)
    )


# name -> (function, whether it requires a `knee`). Per TERM, not per problem.
ACTIVATIONS: Dict[str, Tuple[Callable[..., jax.Array], bool]] = {
    "quadratic": (_quadratic, False),
    "saturated": (_saturated, False),
    "smooth_l1": (_smooth_l1, True),
}
DEFAULT_ACTIVATION = "quadratic"


@dataclass(frozen=True)
class ResidualSpec:
    """One residual available to the cost, and everything needed to check it.

    Attributes:
        formula: The residual in maths, for error messages and the doc.
        dim: Length of r, as a model attribute name ("nq", "nv", "nu") or an
             explicit int. Resolved against the task's model at build time,
             and what a yaml weight/reference vector must match.
        sources: Reference source keyword -> the function implementing it.
             The yaml's ``reference`` picks one of these by name, or gives a
             constant vector, which selects "constant". The NAME says what is
             measured and ``reference`` says against what, so the two are
             orthogonal and every term states both.
        uses_control: True if r depends on the control, which makes the term
             illegal in the terminal block: there is no control to regularize
             at the terminal node, and crocoddyl drops it there for the same
             reason.
    """

    formula: str
    dim: str | int
    sources: Dict[str, "Residual"]
    uses_control: bool


# Every residual takes the same arguments so the registry can call them
# uniformly: the task (for its model constants and its plan), the state, the
# control (None at the terminal node) and the constant reference from the yaml
# (None unless the source is "constant"). Each uses what it needs.
Residual = Callable[
    [Task, mjx.Data, Optional[jax.Array], Optional[jax.Array]], jax.Array
]


def _joint_position_constant(
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


def _joint_velocity_constant(
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


def _control_constant(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    return (control - reference) / task.tau_max


def _control_gravity(
    task: Task,
    state: mjx.Data,
    control: Optional[jax.Array],
    reference: Optional[jax.Array],
) -> jax.Array:
    return (control - state.qfrc_bias) / task.tau_max


def _ee_translation_goal(
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


def _ee_rotation_goal(
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
        sources={
            "plan": _joint_position_plan,
            "constant": _joint_position_constant,
        },
        uses_control=False,
    ),
    "joint_velocity": ResidualSpec(
        formula="v - v_ref",
        dim="nv",
        sources={
            "plan": _joint_velocity_plan,
            "constant": _joint_velocity_constant,
        },
        uses_control=False,
    ),
    # Divided by tau_max so the strong shoulder joints (87 N.m) and the wrist
    # (12 N.m) compare -- equivalently a weighted quadratic with
    # w_i = 1/tau_max_i^2. A constant `reference` is in N.m, before that
    # scaling. Source "gravity" is crocoddyl's ResidualModelControlGrav, up to
    # one difference: qfrc_bias is MuJoCo's full bias force C(q,v)v + g(q),
    # where crocoddyl uses computeGeneralizedGravity(q) alone. Measured along
    # this task's plan the two differ by at most 0.048% of tau_max.
    "control": ResidualSpec(
        formula="(u - u_ref) / tau_max",
        dim="nu",
        sources={
            "gravity": _control_gravity,
            "constant": _control_constant,
        },
        uses_control=True,
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
        sources={"goal": _ee_translation_goal},
        uses_control=False,
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
        sources={"goal": _ee_rotation_goal},
        uses_control=False,
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
    reference_source: str
    reference: Optional[Tuple[float, ...]] = None
    activation: str = DEFAULT_ACTIVATION
    knee: Optional[float] = None


# What build_cost_terms hands to cost_sum: per term, the broadcast weight
# vector, the constant reference (None unless the yaml gave one) and the
# residual itself. Resolved once at construction, so the compiled cost
# contains exactly the terms the yaml asked for.
BuiltTerms = Tuple[
    Tuple[
        jax.Array,
        Optional[jax.Array],
        Optional[float],
        Callable[..., jax.Array],
        "Residual",
    ],
    ...,
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

    Every entry is a mapping stating its weight, its reference and its
    activation, so the file describes the OCP without anyone opening the code::

        joint_position: {weight: 1.0, reference: plan, activation: quadratic}
        control: {weight: 1e-4, reference: gravity, activation: quadratic}
        ee_translation:
          weight: 1.4
          reference: goal
          activation: smooth_l1
          knee: 0.001

    ``reference`` is either a SOURCE KEYWORD from the residual's ``sources``
    (``plan``, ``goal``, ``gravity``) or a CONSTANT VECTOR written inline::

        joint_position:
          weight: 1.0
          reference: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    A bare number is still accepted as shorthand for weight-only, but then the
    residual must have exactly one possible source, so nothing is ambiguous.

    Everything that could silently change what a term means is an error here:
    an unknown name, activation or source, a missing reference, a knee on an
    activation that does not use one, or a missing knee on one that does.
    """
    if block is None:
        return ()
    if not isinstance(block, dict):
        raise ValueError(
            f"'{label}' must be a mapping of residual name -> entry"
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

        knee = None
        activation = DEFAULT_ACTIVATION
        if isinstance(entry, dict):
            unknown = set(entry) - {"weight", "reference", "activation", "knee"}
            if unknown:
                raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
            if "weight" not in entry:
                raise ValueError(f"{where}: missing 'weight'")
            weight = _as_tuple(entry["weight"], "weight", where)
            raw_reference = entry.get("reference")
            activation = str(entry.get("activation", DEFAULT_ACTIVATION))
            if activation not in ACTIVATIONS:
                raise ValueError(
                    f"{where}: unknown activation {activation!r}; expected "
                    f"one of {sorted(ACTIVATIONS)}"
                )
            if "knee" in entry:
                knee = float(entry["knee"])
                if knee <= 0.0:
                    raise ValueError(f"{where}: knee must be > 0")
        else:
            weight = _as_tuple(entry, "weight", where)
            raw_reference = None

        # knee and activation must agree: a knee no activation reads, or a
        # shape whose transition point nobody chose, are both silent errors.
        needs_knee = ACTIVATIONS[activation][1]
        if needs_knee and knee is None:
            raise ValueError(
                f"{where}: activation '{activation}' requires 'knee' -- the "
                f"residual value below which it turns quadratic and settles. "
                f"It is in the residual's own units (m, rad or N.m/tau_max), "
                f"and it is a deliberate accuracy choice, so it has no default."
            )
        if knee is not None and not needs_knee:
            raise ValueError(
                f"{where}: activation '{activation}' does not use 'knee'"
            )

        # The reference: a source keyword, an inline constant vector, or -- for
        # a residual with only one possible source -- omitted.
        named = sorted(k for k in spec.sources if k != "constant")
        reference = None
        if raw_reference is None:
            if len(spec.sources) != 1:
                raise ValueError(
                    f"{where}: 'reference' is required; expected one of "
                    f"{named} or a constant vector of length {spec.dim}"
                )
            source = next(iter(spec.sources))
        elif isinstance(raw_reference, str):
            source = raw_reference
            if source not in spec.sources:
                raise ValueError(
                    f"{where}: unknown reference {source!r} for '{name}'; "
                    f"expected one of {named}"
                    + (" or a constant vector" if "constant" in spec.sources else "")
                )
            if source == "constant":
                raise ValueError(
                    f"{where}: reference 'constant' is selected by GIVING the "
                    f"vector, e.g. reference: [0.0, ...]"
                )
        else:
            if "constant" not in spec.sources:
                raise ValueError(
                    f"{where}: '{name}' takes no constant reference; expected "
                    f"one of {named}"
                )
            source = "constant"
            reference = _as_tuple(raw_reference, "reference", where)

        terms.append(
            CostTerm(
                name=name,
                weight=weight,
                reference_source=source,
                reference=reference,
                activation=activation,
                knee=knee,
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

        activate, _ = ACTIVATIONS[term.activation]
        residual_fn = spec.sources[term.reference_source]
        built.append((weight, reference, term.knee, activate, residual_fn))
    return tuple(built)


def cost_sum(
    task: Task,
    built: "BuiltTerms",
    state: mjx.Data,
    control: Optional[jax.Array],
) -> jax.Array:
    """Σ over the built terms, each with its OWN activation.

    Each term applies its activation to its own weighted residual, exactly as
    crocoddyl's CostModelResidual pairs one activation with one residual.
    There is no problem-wide activation to look up: the yaml entry states the
    shape of that entry.

    The weights sit INSIDE the activation (crocoddyl's WeightedQuad
    convention). For "quadratic" that is identical to a scalar weight outside
    it; for "saturated" it is not, since that form bounds each term to [0, 1)
    whatever its weight.
    """
    total = jnp.zeros(())
    for weight, reference, knee, activate, fn in built:
        residual = fn(task, state, control, reference)
        total = total + activate(weight, residual, knee)
    return total
