"""V-A2 gain-correctness gate (port plan Phase 2).

Finite-difference protocol from doc/feedback_mppi_panda_port_plan.md: fix
the RNG (the same params, hence the same sampled noise, for every
perturbed solve), set num_gain_samples == num_samples so the analytic
gain is exact, central-difference du*(t₀)/dx₀ element-wise (ε = 1e-4 on
q, 1e-3 on q̇), and compare by relative Frobenius error:

  (a) toy task (particle):               ≤ 1 %
  (b) Panda, fresh-seeded at the start:  ≤ 2 %
  (c) Panda, warm-started mid-reach:     ≤ 10 %

plus the Phase 2 timing gate: the solve-with-gains cycle at the
deployment configuration (the tuning-surface values, gain batch 128)
must fit the 25 Hz budget (p95 ≤ 40 ms, post-JIT).

Writes validation/reports/V-A2_gains.json (+ a history.jsonl line), like
every check in the validation protocol. Run in the hydrax uv env:

    uv run pytest tests/test_feedback_gains.py -v -s
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from hydrax.algs import FeedbackMPPI
from hydrax.configs import load_pregrasp_config
from hydrax.tasks.panda_pregrasp import PandaPregrasp
from hydrax.tasks.particle import Particle

REPORT_DIR = Path(__file__).parent.parent / "validation" / "reports"

EPS_Q = 1e-4
EPS_V = 1e-3


@pytest.fixture(scope="session")
def report():
    """Collect gate metrics; write the V-A2 report when the session ends."""
    entries: dict = {}
    yield entries
    if not entries:
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    git_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    verdict = "PASS" if all(e["pass"] for e in entries.values()) else "FAIL"
    report_data = {
        "check": "V-A2",
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha,
        "gates": entries,
        "verdict": verdict,
    }
    (REPORT_DIR / "V-A2_gains.json").write_text(
        json.dumps(report_data, indent=2)
    )
    history_line = {
        "date": report_data["date"],
        "git_sha": git_sha,
        "check": "V-A2",
        "verdict": verdict,
    }
    history_line.update({k: e["value"] for k, e in entries.items()})
    with open(REPORT_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps(history_line) + "\n")


def _fd_gain(ctrl, jit_optimize, state, params) -> np.ndarray:
    """Central-difference du*(t₀)/dx₀ of the produced control.

    Every perturbed solve reuses the same params (same RNG key → the same
    sampling noise), so the only difference is the initial state — exactly
    the sensitivity the analytic gain claims to be.
    """
    nq = ctrl.task.model.nq
    nv = ctrl.task.model.nv
    t0 = float(state.time)
    fd = np.zeros((ctrl.task.model.nu, nq + nv))
    for i in range(nq + nv):
        eps = EPS_Q if i < nq else EPS_V
        actions = []
        for sign in (1.0, -1.0):
            if i < nq:
                x = state.replace(qpos=state.qpos.at[i].add(sign * eps))
            else:
                x = state.replace(qvel=state.qvel.at[i - nq].add(sign * eps))
            perturbed, _ = jit_optimize(x, params)
            actions.append(
                np.asarray(ctrl.get_action(perturbed, t0), dtype=np.float64)
            )
        fd[:, i] = (actions[0] - actions[1]) / (2 * eps)
    return fd


def _relative_frobenius(fd: np.ndarray, analytic: np.ndarray) -> float:
    return float(np.linalg.norm(fd - analytic) / np.linalg.norm(fd))


def test_toy_particle_gains_match_fd(report):
    """(a) Toy task: analytic K vs finite differences, ≤ 1 %."""
    task = Particle()
    # The FD gate checks the gain formula, on smooth dynamics: disable the
    # particle's joint-limit constraint rows. jvp tangents through MJX's
    # constraint solver can go NaN on (even inactive) limit rows — an MJX
    # differentiability quirk, not gain math; with limits enabled 5/64
    # sample gradients here are non-finite and the nan_to_num guard would
    # bias the analytic K. The pregrasp deployment model is constraint-free
    # by design (contacts disabled, limits never active along the reach:
    # 0/1024 non-finite gradients measured at the start state).
    task.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
    task.model = mjx.put_model(task.mj_model)
    ctrl = FeedbackMPPI(
        task,
        num_samples=64,
        noise_std=0.2 * jnp.ones(task.model.nu),
        temperature=0.1,
        num_gain_samples=64,  # == num_samples: the analytic gain is exact
        compute_gains=True,
        plan_horizon=0.5,
        spline_type="cubic",
        num_knots=4,
    )
    # Off-target state so the cost landscape has a slope (at the target the
    # gradients, and K, vanish and the relative error is 0/0).
    state = mjx.make_data(task.model).replace(
        qpos=jnp.array([0.1, -0.08]), qvel=jnp.array([0.05, 0.1])
    )
    params = ctrl.init_params()
    jit_optimize = jax.jit(ctrl.optimize)

    solved, _ = jit_optimize(state, params)
    analytic = np.asarray(solved.gains, dtype=np.float64)
    fd = _fd_gain(ctrl, jit_optimize, state, params)
    rel_err = _relative_frobenius(fd, analytic)
    print(
        f"\n(a) toy: |K|_F={np.linalg.norm(analytic):.4f}  "
        f"rel_err={rel_err:.4%}"
    )
    report["toy_rel_err"] = {"pass": rel_err <= 0.01, "value": rel_err}
    assert rel_err <= 0.01


@pytest.fixture(scope="module")
def panda():
    """The pregrasp task with the tuning-surface configuration."""
    options, config = load_pregrasp_config()
    return PandaPregrasp(options=options), config


def _pregrasp_controller(task, config, num_gain_samples) -> FeedbackMPPI:
    """The pairing glue of the example/adapter, with gains enabled."""
    return FeedbackMPPI(
        task,
        num_samples=config.num_samples,
        noise_std=config.noise_scale * jnp.asarray(task.options.tau_max),
        temperature=config.temperature,
        num_gain_samples=num_gain_samples,
        compute_gains=True,
        plan_horizon=config.plan_horizon,
        spline_type=config.spline_type,
        num_knots=config.num_knots,
        iterations=config.iterations,
    )


def _initial_knots(task, ctrl) -> jax.Array:
    """Warm-start knots from the feedforward torque plan (as the example)."""
    knot_times = np.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
    knot_idx = np.minimum(
        np.round(knot_times / task.dt).astype(int),
        task.reference_ctrl.shape[0] - 1,
    )
    return task.reference_ctrl[knot_idx]


@pytest.fixture(scope="module")
def panda_exact(panda):
    """Pregrasp controller with num_gain_samples == num_samples (exact)."""
    task, config = panda
    ctrl = _pregrasp_controller(
        task, config, num_gain_samples=config.num_samples
    )
    return task, ctrl, jax.jit(ctrl.optimize)


def test_panda_fresh_gains_match_fd(panda_exact, report):
    """(b) Panda at the start state, fresh-seeded: ≤ 2 %."""
    task, ctrl, jit_optimize = panda_exact
    state = mjx.make_data(task.model).replace(
        qpos=jnp.asarray(task.start_q, dtype=jnp.float32), time=0.0
    )
    params = ctrl.init_params(initial_knots=_initial_knots(task, ctrl))

    solved, _ = jit_optimize(state, params)
    analytic = np.asarray(solved.gains, dtype=np.float64)
    fd = _fd_gain(ctrl, jit_optimize, state, params)
    rel_err = _relative_frobenius(fd, analytic)
    print(
        f"\n(b) panda fresh: |K|_F={np.linalg.norm(analytic):.4f}  "
        f"rel_err={rel_err:.4%}"
    )
    report["panda_fresh_rel_err"] = {"pass": rel_err <= 0.02, "value": rel_err}
    assert rel_err <= 0.02


def test_panda_warm_started_gains_match_fd(panda_exact, report):
    """(c) Panda mid-reach with warm-started params: ≤ 10 %."""
    task, ctrl, jit_optimize = panda_exact
    plan_q = np.asarray(task.reference_qpos)
    plan_v = np.asarray(task.reference_qvel)
    data = mjx.make_data(task.model)
    params = ctrl.init_params(initial_knots=_initial_knots(task, ctrl))

    # March the planner at 25 Hz along the reference plan to mid-reach,
    # exactly as deployment warm-starts it (idealized closed loop: the
    # measured state follows the plan).
    for k in range(50):  # t = 0.00 ... 1.96 s
        state = data.replace(
            qpos=jnp.asarray(plan_q[k]),
            qvel=jnp.asarray(plan_v[k]),
            time=k * task.dt,
        )
        warm_params = params
        params, _ = jit_optimize(state, params)

    # Redo the last solve pair (state, warm_params) analytically and by FD.
    solved, _ = jit_optimize(state, warm_params)
    analytic = np.asarray(solved.gains, dtype=np.float64)
    fd = _fd_gain(ctrl, jit_optimize, state, warm_params)
    rel_err = _relative_frobenius(fd, analytic)
    print(
        f"\n(c) panda warm: |K|_F={np.linalg.norm(analytic):.4f}  "
        f"rel_err={rel_err:.4%}"
    )
    report["panda_warm_rel_err"] = {"pass": rel_err <= 0.10, "value": rel_err}
    assert rel_err <= 0.10


def test_cycle_with_gains_fits_the_25hz_budget(panda, report):
    """Phase 2 timing gate: solve + gains at the deployment configuration.

    The plant is irrelevant to solve timing, so the states march along the
    reference plan; what matters is the deployment tuning (the yaml values,
    gain batch num_gain_samples) and the deployment GPU.
    """
    task, config = panda
    ctrl = _pregrasp_controller(task, config, config.num_gain_samples)
    jit_optimize = jax.jit(ctrl.optimize)
    plan_q = np.asarray(task.reference_qpos)
    plan_v = np.asarray(task.reference_qvel)
    data = mjx.make_data(task.model)
    params = ctrl.init_params(initial_knots=_initial_knots(task, ctrl))

    times_ms = []
    for k in range(52):
        state = data.replace(
            qpos=jnp.asarray(plan_q[k]),
            qvel=jnp.asarray(plan_v[k]),
            time=k * task.dt,
        )
        t_start = time.perf_counter()
        params, _ = jit_optimize(state, params)
        jax.block_until_ready(params.gains)
        times_ms.append(1000.0 * (time.perf_counter() - t_start))

    steady = np.asarray(times_ms[2:])  # drop JIT compilation
    mean_ms = float(steady.mean())
    p95_ms = float(np.percentile(steady, 95))
    print(
        f"\n(d) cycle with gains: mean={mean_ms:.1f} ms  p95={p95_ms:.1f} ms  "
        f"(budget 40 ms, gain batch {config.num_gain_samples})"
    )
    report["cycle_mean_ms"] = {"pass": mean_ms <= 40.0, "value": mean_ms}
    report["cycle_p95_ms"] = {"pass": p95_ms <= 40.0, "value": p95_ms}
    assert p95_ms <= 40.0
