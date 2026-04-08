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


def _solve_discrete_lqr_gain(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    max_iters: int = 512,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the discrete algebraic Riccati equation iteratively."""
    p = q.copy()

    for _ in range(max_iters):
        h = r + b.T @ p @ b
        k = np.linalg.solve(h, b.T @ p @ a)
        p_next = q + a.T @ p @ (a - b @ k)
        if np.max(np.abs(p_next - p)) < tol:
            p = p_next
            break
        p = p_next

    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return k.astype(np.float32), p.astype(np.float32)


def _nominal_lqr_controls(
    a: np.ndarray,
    b: np.ndarray,
    gain: np.ndarray,
    x0: np.ndarray,
    x_ref: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Roll out a stationary LQR law to get a nominal control plan."""
    x = x0.astype(np.float32).copy()
    controls = []

    for _ in range(horizon):
        u = -(gain @ (x - x_ref)).astype(np.float32)
        controls.append(u)
        x = a @ x + b @ u

    return np.stack(controls)


def _linear_rollout_costs_and_grads(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    q_terminal: np.ndarray,
    x_ref: np.ndarray,
    x0: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sampled rollout costs and exact initial-state gradients."""
    num_samples, horizon, _ = controls.shape
    nx = a.shape[0]
    costs = np.zeros(num_samples, dtype=np.float32)
    grads = np.zeros((num_samples, nx), dtype=np.float32)

    for sample_idx in range(num_samples):
        x = x0.astype(np.float32).copy()
        states = [x]

        for t in range(horizon):
            e = x - x_ref
            u = controls[sample_idx, t]
            costs[sample_idx] += float(e @ q @ e + u @ r @ u)
            x = a @ x + b @ u
            states.append(x)

        e_terminal = states[-1] - x_ref
        costs[sample_idx] += float(e_terminal @ q_terminal @ e_terminal)

        lam = 2.0 * q_terminal @ e_terminal
        for t in range(horizon - 1, -1, -1):
            e = states[t] - x_ref
            lam = 2.0 * q @ e + a.T @ lam
        grads[sample_idx] = lam

    return costs, grads


def _weighted_action_from_samples(
    costs: np.ndarray,
    nominal_action: np.ndarray,
    action_deltas: np.ndarray,
    temperature: float,
    step_size: float,
) -> np.ndarray:
    """Evaluate the sampled MPPI action map for one state."""
    shifted = costs - np.min(costs)
    weights = np.exp(-shifted / temperature)
    weights /= np.sum(weights)
    return nominal_action + step_size * np.sum(
        weights[:, None] * action_deltas, axis=0
    )


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
    assert np.linalg.norm(K + jac) / np.linalg.norm(jac) < 0.05


def test_feedback_gain_matches_linear_sampled_action_jacobian() -> None:
    """The sbmpc-style gain formula should match the sampled action Jacobian."""
    dt = 0.05
    a = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
    b = np.array([[0.0], [dt]], dtype=np.float32)
    q = np.eye(2, dtype=np.float32)
    r = np.eye(1, dtype=np.float32)
    x_ref = np.array([0.5, 0.0], dtype=np.float32)
    x0 = np.array([0.0, 0.0], dtype=np.float32)
    horizon = 25
    num_samples = 4096
    noise_level = 0.5

    lqr_gain, terminal_cost = _solve_discrete_lqr_gain(a, b, q, r)
    nominal_controls = _nominal_lqr_controls(
        a, b, lqr_gain, x0, x_ref, horizon
    )

    rng = np.random.default_rng(0)
    samples_delta = rng.normal(
        scale=noise_level, size=(num_samples, horizon, 1)
    ).astype(np.float32)
    sampled_controls = nominal_controls[None, :, :] + samples_delta
    costs, cost_grads = _linear_rollout_costs_and_grads(
        a,
        b,
        q,
        r,
        terminal_cost,
        x_ref,
        x0,
        sampled_controls,
    )

    opt = MPPI(
        Pendulum(),
        num_samples=32,
        noise_level=0.1,
        temperature=0.5,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=4,
        step_size=1.0,
    )
    gain = np.asarray(
        jax.device_get(
            opt._feedback_gain_from_samples(
                jnp.array(costs),
                jnp.array(samples_delta[:, 0, :]),
                jnp.array(cost_grads),
            )
        )
    )

    def weighted_action(state: np.ndarray) -> np.ndarray:
        rollout_costs, _ = _linear_rollout_costs_and_grads(
            a,
            b,
            q,
            r,
            terminal_cost,
            x_ref,
            state,
            sampled_controls,
        )
        return _weighted_action_from_samples(
            rollout_costs,
            nominal_controls[0],
            samples_delta[:, 0, :],
            temperature=opt.temperature,
            step_size=opt.step_size,
        )

    eps = 1e-4
    jac = np.zeros_like(gain)
    for i in range(x0.shape[0]):
        dx = np.zeros_like(x0)
        dx[i] = eps
        u_plus = weighted_action(x0 + dx)
        u_minus = weighted_action(x0 - dx)
        jac[:, i] = (u_plus - u_minus) / (2.0 * eps)

    assert np.all(np.isfinite(gain))
    assert np.linalg.norm(gain + jac, ord=np.inf) / np.linalg.norm(
        jac, ord=np.inf
    ) < 0.03


if __name__ == "__main__":
    test_open_loop()
    test_feedback_policy()
    test_feedback_gain_matches_pendulum_finite_difference()
    test_feedback_gain_matches_linear_sampled_action_jacobian()
