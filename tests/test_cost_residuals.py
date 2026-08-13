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

import jax.numpy as jnp
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
    reference = jnp.zeros(dim) if spec.reference == "yaml" else None

    residual = spec.fn(task, state, control, reference)

    assert residual.shape == (dim,), (
        f"'{name}' (r = {spec.formula}) declares dim {dim} but returned "
        f"{residual.shape}"
    )


def test_cost_is_the_weighted_quadratic_the_doc_claims(task, state, control):
    """cost_sum == Σ_terms 0.5 · Σ_i w_i r_i², with the weights INSIDE.

    Pins the composition itself: the per-term weight vector, the sum over
    terms, and the 0.5 of crocoddyl's ActivationModelWeightedQuad.
    """
    terms = (
        CostTerm("joint_position_plan", (3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0)),
        CostTerm("joint_velocity", (0.25,), reference=(0.0,) * task.model.nv),
        CostTerm("control_grav", (7.0,)),
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

    got = cost_sum(task, built, state, control, "quadratic")

    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_scalar_weight_matches_the_uniform_vector(task, state, control):
    """A scalar weight is exactly the uniform vector, not a rescaled one."""
    scalar = build_cost_terms(
        task,
        (CostTerm("joint_position_plan", (10.0,)),),
        "costs.running",
        allow_control=False,
    )
    vector = build_cost_terms(
        task,
        (CostTerm("joint_position_plan", (10.0,) * task.model.nq),),
        "costs.running",
        allow_control=False,
    )

    assert float(
        cost_sum(task, scalar, state, control, "quadratic")
    ) == pytest.approx(
        float(cost_sum(task, vector, state, control, "quadratic")), rel=1e-6
    )


@pytest.mark.parametrize(
    "block, because",
    [
        (
            {"joint_postion_plan": 1.0},
            "a misspelled name is not silently dropped",
        ),
        ({"joint_position": 1.0}, "a constant reference has no default"),
        (
            {"joint_position_plan": {"weight": 1.0, "reference": [0.0] * 7}},
            "a residual with a plan reference must not take a second one",
        ),
        (
            {"control_grav": {"weight": 1e-4, "ref": [0.0] * 7}},
            "unknown sub-key",
        ),
        ({"control_grav": {"reference": [0.0] * 7}}, "a term needs a weight"),
        ({"control_grav": "cheap"}, "a weight must be numeric"),
        ([("control_grav", 1e-4)], "a block is a mapping, not a list"),
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
            CostTerm("control_grav", (1e-4,)),
            False,
            "no control exists at the terminal node to regularize",
        ),
        (
            CostTerm("joint_velocity_plan", (1.0, 2.0, 3.0)),
            True,
            "a weight vector must match the residual's length",
        ),
        (
            CostTerm("joint_velocity", (0.1,), reference=(0.0, 0.0, 0.0)),
            True,
            "a reference must match the residual's length",
        ),
    ],
)
def test_build_rejects(task, term, allow_control, because):
    """Dimension and node-type violations are refused at construction."""
    with pytest.raises(ValueError):
        build_cost_terms(task, (term,), "costs.terminal", allow_control)


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
