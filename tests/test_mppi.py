import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mujoco import mjx

from hydrax.algs.mppi import MPPI
from hydrax.tasks.pendulum import Pendulum


def test_open_loop() -> None:
    """Use MPPI for open-loop pendulum swingup."""
    # Task and optimizer setup
    task = Pendulum()
    opt = MPPI(
        task,
        num_samples=32,
        noise_level=0.1,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    jit_opt = jax.jit(opt.optimize)

    # Initialize the system state and policy parameters
    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        # Do an optimization step
        params, _ = jit_opt(state, params)

    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)

    # Roll out the solution, check that it's good enough
    states, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )
    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0

    if __name__ == "__main__":
        # Plot the solution
        _, ax = plt.subplots(3, 1, sharex=True)
        times = jnp.arange(opt.ctrl_steps) * task.dt

        ax[0].plot(times, states.qpos[0, :, 0])
        ax[0].set_ylabel(r"$\theta$")

        ax[1].plot(times, states.qvel[0, :, 0])
        ax[1].set_ylabel(r"$\dot{\theta}$")

        ax[2].step(times, final_rollout.controls[0], where="post")
        ax[2].axhline(-1.0, color="black", linestyle="--")
        ax[2].axhline(1.0, color="black", linestyle="--")
        ax[2].set_ylabel("u")
        ax[2].set_xlabel("Time (s)")

        time_samples = jnp.linspace(0, times[-1], 100)
        controls = jax.vmap(opt.get_action, in_axes=(None, 0))(
            params, time_samples
        )
        ax[2].plot(time_samples, controls, color="gray", alpha=0.5)

        plt.show()


def test_feedback_policy() -> None:
    """Make sure Feedback-MPPI returns a valid local linear policy."""
    task = Pendulum()
    opt = MPPI(
        task,
        num_samples=32,
        noise_level=0.1,
        temperature=0.05,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )
    jit_opt = jax.jit(opt.optimize_with_feedback)

    state = mjx.make_data(task.model)
    params = opt.init_params()
    params, _, feedback = jit_opt(state, params)

    assert feedback.feedforward.shape == (task.model.nu,)
    assert feedback.feedback_gain.shape == (
        task.model.nu,
        task.feedback_state_dim,
    )
    assert feedback.state_reference.shape == (task.feedback_state_dim,)
    assert jnp.all(jnp.isfinite(feedback.feedback_gain))
    assert jnp.allclose(
        feedback.feedforward, opt.get_action(params, state.time)
    )
    assert jnp.allclose(feedback.state_reference, task.feedback_state(state))


def _pendulum_state(task: Pendulum, qpos: float, qvel: float) -> mjx.Data:
    """Create a pendulum state and refresh derived kinematics."""
    state = task.make_data().replace(
        qpos=jnp.array([qpos], dtype=jnp.float32),
        qvel=jnp.array([qvel], dtype=jnp.float32),
    )
    state = mjx.forward(task.model, state)
    return state


def test_feedback_gain_matches_pendulum_finite_difference() -> None:
    """Feedback gain should match a local finite-difference action Jacobian."""
    task = Pendulum()
    state = _pendulum_state(task, qpos=float(jnp.pi + 0.05), qvel=0.02)
    horizon = 10
    opt = MPPI(
        task,
        num_samples=128,
        noise_level=0.1,
        temperature=0.05,
        plan_horizon=horizon * task.dt,
        spline_type="zero",
        num_knots=horizon,
        iterations=1,
        step_size=1.0,
    )
    params0 = opt.init_params(
        initial_knots=jnp.zeros(
            (opt.num_knots, task.model.nu), dtype=jnp.float32
        )
    )

    def solve_with_feedback(state: mjx.Data):
        return opt.optimize_with_feedback(state, params0)

    def solve_action(state: mjx.Data):
        params, _ = opt.optimize(state, params0)
        return opt.get_action(params, state.time)

    jit_feedback = jax.jit(solve_with_feedback)
    jit_action = jax.jit(solve_action)

    _, _, feedback = jit_feedback(state)
    u0 = np.asarray(jax.device_get(jit_action(state)))

    ff = np.asarray(jax.device_get(feedback.feedforward))
    K = np.asarray(jax.device_get(feedback.feedback_gain))
    x_ref = np.asarray(jax.device_get(feedback.state_reference))
    x0 = np.asarray(jax.device_get(task.feedback_state(state)))

    eps = 1e-3
    jac = np.zeros_like(K)
    for i in range(task.feedback_state_dim):
        delta = np.zeros(task.feedback_state_dim, dtype=np.float32)
        delta[i] = eps
        state_plus = task.set_feedback_state(state, jnp.array(x0 + delta))
        state_minus = task.set_feedback_state(state, jnp.array(x0 - delta))
        state_plus = mjx.forward(task.model, state_plus)
        state_minus = mjx.forward(task.model, state_minus)
        u_plus = np.asarray(jax.device_get(jit_action(state_plus)))
        u_minus = np.asarray(jax.device_get(jit_action(state_minus)))
        jac[:, i] = (u_plus - u_minus) / (2.0 * eps)

    assert np.allclose(x_ref, x0, atol=1e-6)
    assert np.all(np.isfinite(K))
    assert np.allclose(ff, u0, atol=1e-6)
    assert np.linalg.norm(K + jac) / np.linalg.norm(jac) < 0.25


if __name__ == "__main__":
    test_open_loop()
    test_feedback_policy()
    test_feedback_gain_matches_pendulum_finite_difference()
