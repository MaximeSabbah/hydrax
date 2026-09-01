"""The cost registry's contract: what a yaml may ask for, and what it gets.

The term list in ``costs.running`` / ``costs.terminal`` IS the controller, so
these tests cover the promises that list relies on — that a declared residual
dimension is the real one, that the composition is the weighted quadratic the
doc claims, and that every way of writing a term wrong is refused rather than
silently reinterpreted. They deliberately do not pin cost *values*: which
weights are right is a tuning question answered by the ROS simulation, not by
a unit test.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from hydrax import ROOT
from hydrax.configs import load_pregrasp_config
from hydrax.cost_residuals import (
    RESIDUALS,
    CostTerm,
    build_cost_terms,
    cost_sum,
    parse_cost_terms,
    so3_log,
)
from hydrax.tasks.panda_pregrasp import PandaPregrasp

DOC = Path(ROOT).parent / "doc" / "cost_residuals.md"


@pytest.fixture(scope="module")
def task():
    """One task per module: construction runs IK and derives a model."""
    return PandaPregrasp()


@pytest.fixture(scope="module")
def state(task):
    """A state well away from the plan, so no residual is accidentally zero."""
    rng = np.random.default_rng(0)
    data = mjx.make_data(task.model)
    return mjx.forward(
        task.model,
        data.replace(
            qpos=jnp.asarray(
                rng.uniform(-1.0, 1.0, task.model.nq), jnp.float32
            ),
            qvel=jnp.asarray(
                rng.uniform(-1.0, 1.0, task.model.nv), jnp.float32
            ),
            time=jnp.float32(1.0),
        ),
    )


@pytest.fixture(scope="module")
def control(task):
    """A control well away from gravity compensation."""
    rng = np.random.default_rng(1)
    return jnp.asarray(rng.uniform(-20.0, 20.0, task.model.nu), jnp.float32)


@pytest.mark.parametrize("name", sorted(RESIDUALS))
def test_declared_dimension_is_the_real_one(name, task, state, control):
    """Every residual returns a vector of the length its spec advertises.

    The declared dim is what validates a yaml weight or reference vector, so a
    wrong one would accept a wrongly-sized weight and misweight the term. This
    runs over the whole registry, so a residual added later is covered without
    anyone remembering to add a test.
    """
    spec = RESIDUALS[name]
    dim = (
        spec.dim
        if isinstance(spec.dim, int)
        else getattr(task.model, spec.dim)
    )
    for source, fn in spec.sources.items():
        reference = jnp.zeros(dim) if source == "constant" else None
        residual = fn(task, state, control, reference)
        assert residual.shape == (dim,), (
            f"'{name}' source '{source}' (r = {spec.formula}) declares dim "
            f"{dim} but returned {residual.shape}"
        )


def test_cost_is_the_weighted_quadratic_the_doc_claims(task, state, control):
    """cost_sum == Σ_terms 0.5 · Σ_i w_i r_i², with the weights INSIDE.

    Pins the composition itself: the per-term weight vector, the sum over
    terms, and the 0.5 of crocoddyl's ActivationModelWeightedQuad.
    """
    terms = (
        CostTerm("joint_position", (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0), "plan"),
        CostTerm("joint_velocity", (0.25,), "constant",
                 reference=(0.0,) * task.model.nv),
        CostTerm("control", (7.0,), "gravity"),
    )
    built = build_cost_terms(task, terms, "costs.running", allow_control=True)

    q_ref, _ = task._reference(state)
    expected = (
        0.5 * jnp.sum(jnp.asarray(terms[0].weight) * (state.qpos - q_ref) ** 2)
        + 0.5 * 0.25 * jnp.sum(state.qvel**2)
        + 0.5
        * 7.0
        * jnp.sum(((control - state.qfrc_bias) / task.tau_max) ** 2)
    )

    got = cost_sum(task, built, state, control)

    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_scalar_weight_matches_the_uniform_vector(task, state, control):
    """A scalar weight is exactly the uniform vector, not a rescaled one."""
    scalar = build_cost_terms(
        task,
        (CostTerm("joint_position", (10.0,), "plan"),),
        "costs.running",
        allow_control=False,
    )
    vector = build_cost_terms(
        task,
        (CostTerm("joint_position", (10.0,) * task.model.nq, "plan"),),
        "costs.running",
        allow_control=False,
    )

    assert float(
        cost_sum(task, scalar, state, control)
    ) == pytest.approx(
        float(cost_sum(task, vector, state, control)), rel=1e-6
    )


@pytest.mark.parametrize(
    "block, because",
    [
        (
            {"joint_postion": 1.0},
            "a misspelled name is not silently dropped",
        ),
        ({"joint_position": 1.0}, "an ambiguous reference must be stated"),
        (
            {"ee_translation": {"weight": 1.0, "reference": [0.0] * 3}},
            "ee_translation takes no constant reference",
        ),
        (
            {"control": {"weight": 1e-4, "ref": [0.0] * 7}},
            "unknown sub-key",
        ),
        ({"control": {"reference": [0.0] * 7}}, "a term needs a weight"),
        ({"control": "cheap"}, "a weight must be numeric"),
        ([("control", 1e-4)], "a block is a mapping, not a list"),
    ],
)
def test_load_rejects(block, because):
    """Every way of writing a term wrong is refused at load."""
    with pytest.raises(ValueError):
        parse_cost_terms(block, "costs.running")


@pytest.mark.parametrize(
    "term, allow_control, because",
    [
        (
            CostTerm("control", (1e-4,), "gravity"),
            False,
            "no control exists at the terminal node to regularize",
        ),
        (
            CostTerm("joint_velocity", (1.0, 2.0, 3.0), "plan"),
            True,
            "a weight vector must match the residual's length",
        ),
        (
            CostTerm("joint_velocity", (0.1,), "constant", reference=(0.0, 0.0, 0.0)),
            True,
            "a reference must match the residual's length",
        ),
    ],
)
def test_build_rejects(task, term, allow_control, because):
    """Dimension and node-type violations are refused at construction."""
    with pytest.raises(ValueError):
        build_cost_terms(task, (term,), "costs.terminal", allow_control)


def test_userdata_round_trips(task):
    """What plan_userdata packs is what the cost reads back.

    The layout is the one thing here that is expensive to change later — the
    ROS adapter writes it and the residuals read it — so it is pinned by value,
    not by construction.
    """
    q0 = np.array([0.1, -0.9, 0.0, -2.2, 0.0, 1.4, 0.7])
    goal_q = np.asarray(task.goal_q) + 0.05
    goal_pos = np.array([0.45, 0.02, 0.20])

    packed = task.plan_userdata(q0, 6.0, goal_q, goal_pos)
    assert len(packed) == task.model.nuserdata

    data = mjx.make_data(task.model)
    state = data.replace(userdata=jnp.asarray(packed, data.userdata.dtype))
    got_q0, got_duration, got_goal_q = task._plan_params(state)

    assert np.allclose(got_q0, q0, atol=1e-6)
    assert float(got_duration) == pytest.approx(6.0)
    assert np.allclose(got_goal_q, goal_q, atol=1e-6)
    assert np.allclose(task._goal_pos(state), goal_pos, atol=1e-6)


def test_unset_userdata_falls_back_to_construction(task):
    """Zeroed userdata means the construction-time plan and goal.

    Keeps the task usable outside the ROS adapter — the Tier A example and
    every test build one without ever writing userdata.
    """
    state = mjx.make_data(task.model)
    q0, duration, goal_q = task._plan_params(state)

    assert np.allclose(q0, task.start_q, atol=1e-6)
    assert float(duration) == pytest.approx(task.duration)
    assert np.allclose(goal_q, task.goal_q, atol=1e-6)
    assert np.allclose(task._goal_pos(state), task.goal_pos, atol=1e-6)


def test_ee_translation_matches_forward_kinematics(task):
    """The residual is the real gripper position error, not an approximation.

    Checked against an independent MuJoCo forward pass rather than against
    another copy of the same expression.
    """
    q = np.array([0.1, -0.9, 0.0, -2.2, 0.0, 1.4, 0.7])
    goal_pos = np.array([0.45, 0.02, 0.20])

    data = mjx.make_data(task.model)
    packed = task.plan_userdata(q, 6.0, task.goal_q, goal_pos)
    state = mjx.forward(
        task.model,
        data.replace(
            qpos=jnp.asarray(q, jnp.float32),
            userdata=jnp.asarray(packed, data.userdata.dtype),
        ),
    )

    reference = mujoco.MjData(task.mj_model)
    reference.qpos[:] = q
    mujoco.mj_forward(task.mj_model, reference)
    site = mujoco.mj_name2id(
        task.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
    )
    expected = reference.site_xpos[site] - goal_pos

    got = RESIDUALS["ee_translation"].sources["goal"](task, state, None, None)
    assert np.allclose(got, expected, atol=1e-6)


def _log_error(theta, seed=0, count=10):
    """Worst |so3_log - scipy.as_rotvec| at this angle, over random axes.

    float32 throughout, because that is the dtype the rollout runs in — a
    float64 check would certify precision the solver never has.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rotation = Rotation.from_rotvec(theta * axis)
        got = np.asarray(so3_log(jnp.asarray(rotation.as_matrix(), jnp.float32)))
        expected = rotation.as_rotvec()
        # the rotation vector's sign is ambiguous only at exactly pi
        worst = max(
            worst,
            min(
                np.linalg.norm(got - expected),
                np.linalg.norm(got + expected),
            ),
        )
    return worst


@pytest.mark.parametrize("theta", [1e-8, 1e-4, 0.01, 0.3, 1.0, 2.0, 3.0])
def test_so3_log_matches_scipy(theta):
    """so3_log is pinocchio's log3, checked against an independent oracle.

    Spans the small-angle end — where the textbook theta/(2 sin theta) form
    needs a Taylor branch and this one does not — out to 3.0 rad, past which
    float32 precision, not the formula, is the limit (see the next test).
    """
    assert _log_error(theta) < 1e-5, f"theta={theta}"


def test_so3_log_degrades_gracefully_towards_pi():
    """Near the log3 singularity the error grows smoothly, it does not explode.

    The singularity at theta = pi is real (pinocchio special-cases it via the
    symmetric part of R). What matters for a cost function is that approaching
    it stays bounded and small rather than producing a NaN or a wild value:
    measured error runs ~1e-7/(pi - theta), so 5.6e-5 rad at pi - 1.6e-3.
    """
    assert _log_error(np.pi - 1e-1) < 1e-5
    assert _log_error(np.pi - 1e-2) < 1e-4
    assert _log_error(np.pi - 1e-3) < 1e-3


def test_so3_log_gradient_is_finite_at_zero():
    """The zero-rotation gradient must not be NaN.

    ``norm`` has a NaN gradient at the origin and the division is 0/0 there,
    and this residual sits inside the jacfwd path that produces dJ/dx0 — a NaN
    there silently poisons the gain, which is nan_to_num'd downstream.
    """
    jacobian = np.asarray(jax.jacfwd(so3_log)(jnp.eye(3)))
    assert np.all(np.isfinite(jacobian))


def test_ee_rotation_is_zero_at_the_goal_orientation(task):
    """Zero residual exactly when the gripper is at the goal orientation."""
    data = mjx.make_data(task.model)
    q = np.array([0.1, -0.9, 0.0, -2.2, 0.0, 1.4, 0.7])
    reference = mujoco.MjData(task.mj_model)
    reference.qpos[:] = q
    mujoco.mj_forward(task.mj_model, reference)
    site = mujoco.mj_name2id(
        task.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
    )
    reached = reference.site_xmat[site].reshape(3, 3)

    packed = task.plan_userdata(
        q, 6.0, task.goal_q, np.zeros(3), goal_rot=reached
    )
    state = mjx.forward(
        task.model,
        data.replace(
            qpos=jnp.asarray(q, jnp.float32),
            userdata=jnp.asarray(packed, data.userdata.dtype),
        ),
    )

    residual = RESIDUALS["ee_rotation"].sources["goal"](task, state, None, None)
    assert np.allclose(residual, 0.0, atol=1e-5)


def test_smooth_l1_gradient_stops_fading(task, state, control):
    """The point of the knee: the pull becomes constant instead of decaying.

    A quadratic cost's gradient is w·r, so it vanishes as the error shrinks and
    below some radius it can no longer command breakaway torque — measured on
    hardware as a stall at |tau - gravity|/breakaway = 0.62 on every run. The
    smooth-L1 gradient is w·r/√(r²+ε), which flattens to w above √ε.
    """
    weight, knee = 10.0, 1e-3  # knee = 1 mm

    def quadratic(r):
        return 0.5 * weight * r**2

    def smooth_l1(r):
        return weight * (jnp.sqrt(r**2 + knee**2) - knee)

    dq, ds = jax.grad(quadratic), jax.grad(smooth_l1)

    # far above the knee the smooth-L1 pull is flat, the quadratic's is not
    assert float(ds(0.1)) == pytest.approx(weight, rel=1e-3)
    assert float(ds(0.0167)) == pytest.approx(weight, rel=1e-2)
    # and it is far stronger exactly where the hardware stalled
    assert float(ds(0.0167)) / float(dq(0.0167)) > 50
    # below the knee it fades, so the arm settles instead of hunting
    assert float(ds(0.0001)) < 0.15 * weight


def test_knee_term_matches_crocoddyl_smooth_1norm(task, state, control):
    """smooth_l1 == Σ_i w_i (√(r_i² + knee²) − knee), per component."""
    knee = 1e-3
    terms = (CostTerm("ee_translation", (10.0,), "goal", activation="smooth_l1", knee=knee),)
    built = build_cost_terms(task, terms, "costs.running", allow_control=False)

    residual = RESIDUALS["ee_translation"].sources["goal"](task, state, control, None)
    expected = jnp.sum(
        10.0 * (jnp.sqrt(residual**2 + knee**2) - knee)
    )

    got = cost_sum(task, built, state, control)
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_knee_must_be_positive():
    """knee is a residual magnitude; zero or negative is meaningless."""
    with pytest.raises(ValueError):
        parse_cost_terms(
            {"ee_translation": {"weight": 10.0, "activation": "smooth_l1", "knee": 0.0}},
            "costs.running",
        )


def test_committed_yaml_builds():
    """The shipped pregrasp.yaml is a valid controller.

    The schema and the tuning surface are edited separately; this is what stops
    a mismatch between them from reaching a robot.
    """
    options, _ = load_pregrasp_config()
    assert options.running_costs, "the running block defines no cost at all"
    task = PandaPregrasp(options=options)
    assert len(task._running_terms) == len(options.running_costs)
    assert len(task._terminal_terms) == len(options.terminal_costs)


def test_doc_tabulates_every_residual():
    """Every residual has a row in doc/cost_residuals.md's table.

    That table is what a reader consults to pick a name, so a residual missing
    from it is effectively invisible however much prose mentions it. Checking
    the row, not just the name, is what makes this catch a deleted row.
    """
    rows = {
        line.split("|")[1].strip()
        for line in DOC.read_text().splitlines()
        if line.startswith("| `")
    }
    missing = [name for name in RESIDUALS if f"`{name}`" not in rows]
    assert not missing, f"residuals with no table row: {sorted(missing)}"
