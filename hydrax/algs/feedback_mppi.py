from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.alg_base import (
    SamplingBasedController,
    SamplingParams,
    Trajectory,
)
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class FeedbackMPPIParams(SamplingParams):
    """Policy parameters for Feedback-MPPI.

    Same as SamplingParams (tk, mean, rng), plus:

    Attributes:
        gains: K = du*/dx₀ from the last optimize call, shape (nu, nq + nv).
               Zero before the first solve, and always zero when the
               controller was built with ``compute_gains=False``.
        gain_ess: Effective sample size ESS = 1/Σᵢwᵢ² of the softmax
                  weights over the gain batch in the last solve: how many
                  rollouts K is effectively averaging over (≈ batch size
                  when the weights are spread, ≈ 1 when one sample takes
                  all the weight). The V-A3 readout of softmax degeneracy
                  — F-MPPI gains are only trustworthy when several samples
                  share the weight. Zero until gains are computed.
        gain_nominal_weight: Softmax weight of the zero-noise nominal
                             (sample 0) within the gain batch. The
                             measured sbmpc failure mode was the nominal
                             winning the softmax (weight ≈ 1 ⇒ K ≈ 0,
                             spiking when a fluke sample beats it), so
                             this is its direct fingerprint (V-A3 bounds
                             how often it exceeds 0.5). Zero until gains
                             are computed.
    """

    gains: jax.Array
    gain_ess: jax.Array
    gain_nominal_weight: jax.Array


class FeedbackMPPI(SamplingBasedController):
    """Feedback-MPPI: MPPI that also returns state-feedback gains K = du*/dx₀.

    The controller of the Feedback-MPPI port
    (doc/feedback_mppi_panda_port_plan.md). The policy update is exactly
    MPPI's exponentially weighted average; on top of it this controller
    differs in three ways:

    - **Per-actuator noise std** (``noise_std``, shape (nu,)). Plain MPPI's
      scalar ``noise_level`` assumes a homogeneous action space — which the
      existing tasks have (e.g. the G1 humanoid uses position-servo
      actuators, so every control is a joint-angle target and the servo kp
      does the per-joint force scaling). A torque-controlled arm has no such
      normalization: the Panda spans 87 Nm shoulders to 12 Nm wrists, and a
      scalar noise either under-drives the shoulders or thrashes the wrists
      (2026-07-03 Phase 1 sweeps: scalar noise diverges or tracks at 0.20 m;
      noise proportional to the torque limits reaches ~0.01 m).

    - **Zero-noise nominal sample** (sample 0 = the warm-started mean), so
      the current plan is always scored. Measured on the pregrasp reach:
      2x better terminal error with it. It is also structurally required by
      the gain computation: the gains are a softmax-weighted combination of
      deviations from exactly this nominal.

    - **Feedback gains** (``compute_gains=True``): each solve also returns
      K = du*/dx₀, the sensitivity of the first applied control to the
      initial state, in ``params.gains`` — the Feedback-MPPI gain, for a
      1 kHz low level applying ``tau = u* + K (x − x₀)`` between 25 Hz
      planner updates, where x₀ is the state the plan was solved from
      (NOT the tracking reference: the planner already responded to the
      tracking error when it solved from x₀, so K multiplies only the
      drift since planning — the linearized re-solve). In feedback mode
      this law fully replaces the fixed joint impedance (user decision
      2026-07-06): K is the local optimal feedback and must be deployable
      straight from the solver. The default is False: the feedforward
      deployment mode uses the sampling improvements only and pays
      nothing for the gain path (the mode switch lives in the pairing
      glue — the example script and the ROS planner adapter).
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_std: jax.Array,
        temperature: float,
        num_gain_samples: int = 128,
        compute_gains: bool = False,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "cubic",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_std: Per-actuator std of the Gaussian sampling noise,
                       shape (nu,).
            temperature: The temperature parameter λ. Higher values take a
                         more even average over the samples.
            num_gain_samples: Rollouts used for the gain computation: the
                              zero-noise nominal + the lowest-cost samples.
            compute_gains: Whether optimize also computes K = du*/dx₀.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different
                           randomizations. Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Must be linear in the knot values ("zero", "linear"
                         or "cubic" — not "akima"): the gain formula relies
                         on it.
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_std = jnp.asarray(noise_std, dtype=jnp.float32)
        assert self.noise_std.shape == (task.model.nu,)

        if not 1 <= num_gain_samples <= num_samples:
            raise ValueError(
                f"num_gain_samples ({num_gain_samples}) must be in "
                f"[1, num_samples] ([1, {num_samples}])"
            )
        if compute_gains and self.num_randomizations > 1:
            # The gains differentiate one model's rollout cost; du*/dx₀ of
            # a risk-combined cost over randomized models is not defined
            # by the F-MPPI formula.
            raise ValueError(
                "feedback gains require a single (non-randomized) model"
            )
        self.num_samples = num_samples
        self.temperature = temperature
        self.num_gain_samples = num_gain_samples
        self.compute_gains = compute_gains

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> FeedbackMPPIParams:
        """Initialize the policy parameters, with zero gains."""
        _params = super().init_params(initial_knots, seed)
        nx = self.task.model.nq + self.task.model.nv
        return FeedbackMPPIParams(
            tk=_params.tk,
            mean=_params.mean,
            rng=_params.rng,
            gains=jnp.zeros((self.task.model.nu, nx)),
            gain_ess=jnp.zeros(()),
            gain_nominal_weight=jnp.zeros(()),
        )

    def sample_knots(
        self, params: FeedbackMPPIParams
    ) -> Tuple[jax.Array, FeedbackMPPIParams]:
        """Sample control knots; sample 0 is the zero-noise nominal."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_std * noise
        controls = controls.at[0].set(params.mean)
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: FeedbackMPPIParams, rollouts: Trajectory
    ) -> FeedbackMPPIParams:
        """Update the mean with an exponentially weighted average (as MPPI)."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)

    def optimize(
        self, state: mjx.Data, params: FeedbackMPPIParams
    ) -> Tuple[FeedbackMPPIParams, Trajectory]:
        """Optimize as usual, then compute the gains from the same rollouts."""
        params, rollouts = super().optimize(state, params)
        if self.compute_gains:
            gains, ess, nominal_weight = self._compute_gains(state, rollouts)
            params = params.replace(
                gains=gains,
                gain_ess=ess,
                gain_nominal_weight=nominal_weight,
            )
        return params, rollouts

    def _compute_gains(
        self, state: mjx.Data, rollouts: Trajectory
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """K = du*/dx₀: sensitivity of the first applied control to the state.

        Returns (K, ESS, nominal weight); the last two are the V-A3
        health readouts of exactly the softmax weights the gains used
        (see FeedbackMPPIParams).

        The F-MPPI gain, mirroring the sbmpc reference implementation
        (``gains.py::MPPIGain.gains_computation`` + the gain path of
        ``solvers.py``, FD-proven there; re-proven here by the V-A2 gate).
        The updated control is u*(t₀) = Σ_b w_b u_b(t₀) with
        w = softmax(-J/temperature) — exactly, because the spline is linear
        in the knot values — and only the weights depend on x₀, so

            K = Σ_b (dw_b/dx₀) u_b(t₀),
            dw_b/dx₀ = -w_b (dJ_b/dx₀ − Σ_j w_j dJ_j/dx₀) / temperature.

        As in sbmpc, the cost gradients are computed only for a gain batch
        of the zero-noise nominal + the (num_gain_samples − 1) lowest-cost
        samples, with the weights renormalized over that batch (exact when
        num_gain_samples == num_samples — the V-A2 setting), and sbmpc's
        non-finite guards (cost saturation, nan_to_num) are kept: one
        diverged rollout must not poison the deployed K. With
        iterations > 1 the gains cover the final iteration's update.
        """
        costs = jnp.sum(rollouts.costs, axis=1)  # (num_samples,)
        # sbmpc _saturate_costs: a non-finite cost gets ~zero softmax weight
        costs = jnp.where(jnp.isfinite(costs), costs, 1e6)

        # sbmpc select_nominal_and_lowest_cost_indices: keep sample 0 (the
        # nominal) and fill the batch with the lowest-cost other samples.
        _, lowest = jax.lax.top_k(-costs[1:], self.num_gain_samples - 1)
        batch = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32), lowest.astype(jnp.int32) + 1]
        )

        # sbmpc samples_delta[:, 0, :]: each sample's control at the
        # application time t₀ = state.time, relative to the nominal's
        # (rollout controls start at tq[0] = tk[0] = t₀; the constant shift
        # cancels in the sum since Σ_b dw_b = 0).
        samples_delta = (
            rollouts.controls[batch, 0, :] - rollouts.controls[0, 0, :]
        )
        samples_delta = jnp.nan_to_num(samples_delta)

        # sbmpc rollout_gradients_to_state: dJ/dx₀ for the gain batch only
        gradients = jax.vmap(
            lambda controls: self._rollout_cost_gradient(state, controls)
        )(rollouts.controls[batch])  # (num_gain_samples, nq + nv)
        gradients = jnp.nan_to_num(gradients)

        # sbmpc gains_computation, with lam = 1/temperature
        weights = jax.nn.softmax(-costs[batch] / self.temperature)
        weights_grad_shift = jnp.sum(weights[:, None] * gradients, axis=0)
        weights_grad = (
            -weights[:, None] * (gradients - weights_grad_shift)
            / self.temperature
        )
        gains = jnp.einsum("bi,bo->oi", weights_grad, samples_delta)
        ess = 1.0 / jnp.sum(jnp.square(weights))
        return jnp.nan_to_num(gains), ess, weights[0]

    def _rollout_cost_gradient(
        self, state: mjx.Data, controls: jax.Array
    ) -> jax.Array:
        """dJ/dx₀ of one scored rollout, via forward-mode AD.

        Mirrors sbmpc's ``rollout_single_state_gradient``: replay the
        eval_rollouts physics/cost sequence as a scalar function of
        (qpos, qvel) — mjx.step recomputes every derived quantity from
        them, so replacing qpos/qvel is a complete state perturbation —
        and evaluate its jvp along the nq + nv basis directions.
        Forward-mode, because reverse-mode AD does not work through MJX's
        internal solver while loops.
        """

        def rollout_cost(qpos: jax.Array, qvel: jax.Array) -> jax.Array:
            def _scan_fn(x: mjx.Data, u: jax.Array):
                x = x.replace(ctrl=u)
                x = mjx.step(self.model, x)
                return x, self.dt * self.task.running_cost(x, u)

            x0 = state.replace(qpos=qpos, qvel=qvel)
            final_state, step_costs = jax.lax.scan(_scan_fn, x0, controls)
            return jnp.sum(step_costs) + self.task.terminal_cost(final_state)

        nq = self.task.model.nq
        basis = jnp.eye(nq + self.task.model.nv, dtype=state.qpos.dtype)
        return jax.vmap(
            lambda tangent: jax.jvp(
                rollout_cost,
                (state.qpos, state.qvel),
                (tangent[:nq], tangent[nq:]),
            )[1]
        )(basis)
