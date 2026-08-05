"""Export local Panda Feedback-MPPI OCP approximations for Crocoddyl.

This script runs the controller from ``examples/panda_pregrasp.py`` at selected
points of the minimum-jerk reach, then exports the exact first-order MJX
linearization and second-order Hydrax cost approximation used for an
apples-to-apples Crocoddyl Riccati comparison.

The exported running cost matches Hydrax's rollout convention: the control is
applied first, then ``dt * task.running_cost(x_next, u)`` is accumulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from hydrax.algs import FeedbackMPPI
from hydrax.configs import load_pregrasp_config
from hydrax.tasks.panda_pregrasp import PandaPregrasp


def _parse_fractions(value: str) -> np.ndarray:
    fractions = np.asarray([float(item) for item in value.split(",")])
    if fractions.ndim != 1 or fractions.size == 0:
        raise argparse.ArgumentTypeError("at least one fraction is required")
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise argparse.ArgumentTypeError("fractions must lie in [0, 1]")
    return fractions


def _state_vector(data: mjx.Data) -> jax.Array:
    return jnp.concatenate((data.qpos, data.qvel))


def _replace_state(data: mjx.Data, x: jax.Array, nq: int) -> mjx.Data:
    return data.replace(qpos=x[:nq], qvel=x[nq:])


def _reference_knots(
    task: PandaPregrasp,
    controller: FeedbackMPPI,
    time_sec: float,
) -> jax.Array:
    knot_times = np.linspace(
        time_sec,
        time_sec + controller.plan_horizon,
        controller.num_knots,
    )
    indices = np.clip(
        np.rint(knot_times / task.dt).astype(int),
        0,
        task.reference_ctrl.shape[0] - 1,
    )
    return task.reference_ctrl[indices]


def _make_reference_state(task: PandaPregrasp, time_sec: float) -> mjx.Data:
    index = min(
        int(round(time_sec / task.dt)),
        task.reference_qpos.shape[0] - 1,
    )
    data = mjx.make_data(task.model)
    return data.replace(
        qpos=task.reference_qpos[index],
        qvel=task.reference_qvel[index],
        time=jnp.asarray(time_sec, dtype=task.reference_qpos.dtype),
    )


def _mean_control_sequence(
    controller: FeedbackMPPI,
    params: Any,
) -> jax.Array:
    query_times = jnp.linspace(
        params.tk[0],
        params.tk[-1],
        controller.ctrl_steps,
    )
    return controller.interp_func(
        query_times,
        params.tk,
        params.mean[None, ...],
    )[0]


def _nominal_rollout(
    controller: FeedbackMPPI,
    state: mjx.Data,
    controls: jax.Array,
) -> tuple[mjx.Data, mjx.Data]:
    def step(data: mjx.Data, control: jax.Array):
        next_data = mjx.step(controller.model, data.replace(ctrl=control))
        return next_data, next_data

    final_state, post_states = jax.lax.scan(step, state, controls)
    pre_states = jax.tree.map(
        lambda initial, post: jnp.concatenate(
            (initial[None, ...], post[:-1]), axis=0
        ),
        state,
        post_states,
    )
    return pre_states, final_state


def _linear_quadratic_approximation(
    task: PandaPregrasp,
    controller: FeedbackMPPI,
    pre_states: mjx.Data,
    final_state: mjx.Data,
    controls: jax.Array,
) -> dict[str, np.ndarray]:
    nq = task.model.nq
    nx = task.model.nq + task.model.nv

    x_nominal = jax.vmap(_state_vector)(pre_states)
    z_nominal = jnp.concatenate((x_nominal, controls), axis=1)

    def next_state(template: mjx.Data, z: jax.Array) -> jax.Array:
        x = z[:nx]
        control = z[nx:]
        data = _replace_state(template, x, nq).replace(ctrl=control)
        return _state_vector(mjx.step(controller.model, data))

    def stage_cost(template: mjx.Data, z: jax.Array) -> jax.Array:
        x = z[:nx]
        control = z[nx:]
        data = _replace_state(template, x, nq).replace(ctrl=control)
        next_data = mjx.step(controller.model, data)
        return controller.dt * task.running_cost(next_data, control)

    jacobians = jax.vmap(jax.jacfwd(next_state, argnums=1))(
        pre_states,
        z_nominal,
    )
    # Use forward-over-forward differentiation. Reverse-mode AD through
    # MJX's internal solver loops is unsupported, which is also why the
    # Feedback-MPPI gain implementation uses forward-mode JVPs.
    stage_hessian = jax.jacfwd(
        jax.jacfwd(stage_cost, argnums=1), argnums=1
    )
    hessians = jax.vmap(stage_hessian)(pre_states, z_nominal)
    hessians = 0.5 * (hessians + jnp.swapaxes(hessians, -1, -2))

    final_x = _state_vector(final_state)

    def terminal_cost(x: jax.Array) -> jax.Array:
        return task.terminal_cost(_replace_state(final_state, x, nq))

    terminal_hessian = jax.jacfwd(jax.jacfwd(terminal_cost))(final_x)
    terminal_hessian = 0.5 * (
        terminal_hessian + terminal_hessian.T
    )

    return {
        "A": np.asarray(jacobians[:, :, :nx]),
        "B": np.asarray(jacobians[:, :, nx:]),
        "Q": np.asarray(hessians[:, :nx, :nx]),
        "R": np.asarray(hessians[:, nx:, nx:]),
        "N": np.asarray(hessians[:, :nx, nx:]),
        "Qf": np.asarray(terminal_hessian),
        "x_nominal": np.asarray(
            jnp.concatenate((x_nominal, final_x[None, :]), axis=0)
        ),
        "u_nominal": np.asarray(controls),
    }


def _snapshot(
    task: PandaPregrasp,
    controller: FeedbackMPPI,
    optimize,
    time_sec: float,
    seed: int,
    updates: int,
) -> dict[str, np.ndarray | float]:
    state = _make_reference_state(task, time_sec)
    initial_knots = _reference_knots(task, controller, time_sec)
    params = controller.init_params(initial_knots=initial_knots, seed=seed)
    params = params.replace(
        tk=jnp.linspace(
            time_sec,
            time_sec + controller.plan_horizon,
            controller.num_knots,
        )
    )

    for _ in range(updates):
        params, _ = optimize(state, params)
    jax.block_until_ready(params.mean)

    controls = _mean_control_sequence(controller, params)
    pre_states, final_state = _nominal_rollout(controller, state, controls)
    approximation = _linear_quadratic_approximation(
        task,
        controller,
        pre_states,
        final_state,
        controls,
    )
    approximation.update(
        {
            "time": float(time_sec),
            "K_feedback_mppi": np.asarray(params.gains),
            "gain_ess": float(params.gain_ess),
            "gain_nominal_weight": float(params.gain_nominal_weight),
        }
    )
    return approximation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export exact local MJX dynamics/cost approximations and "
            "Feedback-MPPI gains for a Crocoddyl Riccati comparison."
        )
    )
    parser.add_argument(
        "--fractions",
        type=_parse_fractions,
        default=_parse_fractions("0,0.25,0.5,0.75,1"),
        help="Comma-separated fractions of the reach duration.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=1,
        help="MPPI updates performed from the reference-torque warm start.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-gain-samples",
        type=int,
        default=None,
        help=(
            "Override the configured gain batch. Use the solver sample count "
            "for the exact all-rollout estimator."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/reports/panda_feedback_local_lq.npz"),
    )
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")

    options, config = load_pregrasp_config()
    task = PandaPregrasp(options=options)
    num_gain_samples = (
        config.num_gain_samples
        if args.num_gain_samples is None
        else args.num_gain_samples
    )
    controller = FeedbackMPPI(
        task,
        num_samples=config.num_samples,
        noise_std=config.noise_scale * task.tau_max,
        temperature=config.temperature,
        mean_adaptation_rate=config.mean_adaptation_rate,
        num_gain_samples=num_gain_samples,
        compute_gains=True,
        plan_horizon=config.plan_horizon,
        spline_type=config.spline_type,
        num_knots=config.num_knots,
        iterations=config.iterations,
    )
    optimize = jax.jit(controller.optimize)

    times = np.asarray(args.fractions) * task.duration
    snapshots = []
    for index, time_sec in enumerate(times):
        print(
            f"[{index + 1}/{len(times)}] linearizing at "
            f"t={time_sec:.3f} s"
        )
        snapshots.append(
            _snapshot(
                task,
                controller,
                optimize,
                float(time_sec),
                args.seed + index,
                args.updates,
            )
        )

    metadata = {
        "schema_version": 1,
        "task": "PandaPregrasp",
        "state_order": ["q", "v"],
        "feedback_mppi_law": "delta_u = K_feedback_mppi @ delta_x",
        "crocoddyl_law": "delta_u = -K_crocoddyl @ delta_x",
        "dt": float(task.dt),
        "horizon_steps": int(controller.ctrl_steps),
        "plan_horizon": float(controller.plan_horizon),
        "reach_duration": float(task.duration),
        "num_samples": int(config.num_samples),
        "num_gain_samples": int(num_gain_samples),
        "temperature": float(config.temperature),
        "noise_scale": float(config.noise_scale),
        "mean_adaptation_rate": float(config.mean_adaptation_rate),
        "spline_type": config.spline_type,
        "num_knots": int(config.num_knots),
        "updates": int(args.updates),
        "note": (
            "A/B are exact first derivatives of one MJX step. Q/R/N/Qf "
            "are exact Hessians of the Hydrax discretized costs around the "
            "updated MPPI mean trajectory."
        ),
    }

    keys = (
        "A",
        "B",
        "Q",
        "R",
        "N",
        "Qf",
        "x_nominal",
        "u_nominal",
        "K_feedback_mppi",
    )
    payload = {key: np.stack([snapshot[key] for snapshot in snapshots]) for key in keys}
    payload.update(
        {
            "times": np.asarray([snapshot["time"] for snapshot in snapshots]),
            "gain_ess": np.asarray(
                [snapshot["gain_ess"] for snapshot in snapshots]
            ),
            "gain_nominal_weight": np.asarray(
                [snapshot["gain_nominal_weight"] for snapshot in snapshots]
            ),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
