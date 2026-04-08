from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.alg_base import (
    LinearFeedbackPolicy,
    SamplingBasedController,
    SamplingParams,
    Trajectory,
)
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class MPPIParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """


@dataclass
class MPPIStepData:
    """Information from one MPPI optimization iteration."""

    rollouts: Trajectory
    nominal_knots: jax.Array
    dr_rng: jax.Array


class MPPI(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        feedback_num_samples: int | None = None,
        feedback_diff_eps: float = 1e-3,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        step_size: float = 1.0,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            feedback_num_samples: Number of sampled rollouts to use for the
                                  feedback-gain estimate. Defaults to all
                                  sampled rollouts. Lower values reduce the
                                  cost of gain computation.
            feedback_diff_eps: Central-difference step used to estimate rollout
                               cost sensitivities for the feedback gains.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
            step_size: Blending factor for mean update (0, 1]. At 1.0, the mean
                       is fully replaced (standard MPPI). Lower values blend
                       with the previous mean for smoother convergence.
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
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.step_size = step_size
        self.feedback_num_samples = (
            num_samples
            if feedback_num_samples is None
            else feedback_num_samples
        )
        self.feedback_diff_eps = feedback_diff_eps
        if self.feedback_num_samples < 1:
            raise ValueError("feedback_num_samples must be at least 1!")
        if self.feedback_diff_eps <= 0.0:
            raise ValueError("feedback_diff_eps must be positive!")

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        return MPPIParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def sample_knots(self, params: MPPIParams) -> Tuple[jax.Array, MPPIParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        return controls, params.replace(rng=rng)

    def optimize(
        self, state: mjx.Data, params: MPPIParams
    ) -> Tuple[MPPIParams, Trajectory]:
        """Run MPPI and return the updated parameters and final rollouts."""
        params, step_data = self._optimize_rollouts(state, params)
        return params, step_data.rollouts

    def optimize_with_feedback(
        self, state: mjx.Data, params: MPPIParams
    ) -> Tuple[MPPIParams, Trajectory, LinearFeedbackPolicy]:
        """Run MPPI and compute a local linear feedback policy.

        The returned feedback gain follows the convention

            u(x) ≈ feedforward + K @ (state_reference - x)

        so it can be fed directly to a linear feedback controller.
        """
        params, final_step = self._optimize_rollouts(state, params)
        feedback = self._feedback_policy(
            state,
            params,
            final_step.rollouts,
            final_step.nominal_knots,
            final_step.dr_rng,
        )
        return params, final_step.rollouts, feedback

    def _optimize_rollouts(
        self, state: mjx.Data, params: MPPIParams
    ) -> Tuple[MPPIParams, MPPIStepData]:
        """Run MPPI iterations and return the final sampled rollouts."""
        params = self._warmstart_params(state, params)

        def _optimize_scan_body(params: MPPIParams, _: jax.Array):
            nominal_knots = params.mean

            knots, params = self.sample_knots(params)
            knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

            rng, dr_rng = jax.random.split(params.rng)
            rollouts = self.rollout_with_randomizations(
                state, params.tk, knots, dr_rng
            )
            params = params.replace(rng=rng)
            params = self.update_params(params, rollouts)

            step_data = MPPIStepData(
                rollouts=rollouts, nominal_knots=nominal_knots, dr_rng=dr_rng
            )
            return params, step_data

        params, step_data = jax.lax.scan(
            f=_optimize_scan_body, init=params, xs=jnp.arange(self.iterations)
        )
        final_step = jax.tree.map(lambda x: x[-1], step_data)
        return params, final_step

    def update_params(
        self, params: MPPIParams, rollouts: Trajectory
    ) -> MPPIParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        weights = self._importance_weights(costs)
        weighted_mean = jnp.sum(
            weights[:, None, None] * rollouts.knots, axis=0
        )
        mean = (
            1.0 - self.step_size
        ) * params.mean + self.step_size * weighted_mean
        return params.replace(mean=mean)

    def _warmstart_params(
        self, state: mjx.Data, params: MPPIParams
    ) -> MPPIParams:
        """Advance the spline forward in time before replanning."""
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(
            new_tk, params.tk, params.mean[None, ...]
        )[0]
        return params.replace(tk=new_tk, mean=new_mean)

    def _feedback_policy(
        self,
        state: mjx.Data,
        params: MPPIParams,
        rollouts: Trajectory,
        nominal_knots: jax.Array,
        dr_rng: jax.Array,
    ) -> LinearFeedbackPolicy:
        """Compute a local linear feedback policy.

        This follows the same sampled-perturbation structure as sbmpc:
        we differentiate each sampled rollout cost with respect to the initial
        state, differentiate the MPPI importance weights, then contract those
        weight sensitivities with the first-step action perturbations.
        """
        costs = self._sanitize_costs(jnp.sum(rollouts.costs, axis=1))
        sample_inds = self._feedback_sample_inds(costs)
        costs = costs[sample_inds]
        knots = rollouts.knots[sample_inds]
        controls = rollouts.controls[sample_inds]

        state_reference = self.task.feedback_state(state)
        cost_grads = self._rollout_cost_gradients(
            state_reference, state, params.tk, knots, dr_rng
        )

        nominal_action = self.interp_func(
            state.time, params.tk, nominal_knots[None, ...]
        )[0]
        action_deltas = controls[:, 0, :] - nominal_action[None, :]
        feedback_gain = self._feedback_gain_from_samples(
            costs, action_deltas, cost_grads
        )

        return LinearFeedbackPolicy(
            feedforward=self.get_action(params, state.time),
            feedback_gain=feedback_gain,
            state_reference=state_reference,
        )

    def _feedback_sample_inds(self, costs: jax.Array) -> jax.Array:
        """Select which rollouts to use when estimating the feedback gain."""
        num_feedback = min(self.feedback_num_samples, self.num_samples)
        if num_feedback == self.num_samples:
            return jnp.arange(self.num_samples)
        return jnp.argsort(costs)[:num_feedback]

    def _importance_weights(self, costs: jax.Array) -> jax.Array:
        """Compute MPPI importance weights after sanitizing the costs."""
        costs = self._sanitize_costs(costs)
        return jax.nn.softmax(-costs / self.temperature, axis=0)

    def _sanitize_costs(self, costs: jax.Array) -> jax.Array:
        """Clamp invalid rollout costs so the MPPI weights stay well-defined."""
        return jnp.nan_to_num(costs, nan=1e6, posinf=1e6, neginf=-1e6)

    def _rollout_cost_gradients(
        self,
        state_reference: jax.Array,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        dr_rng: jax.Array,
    ) -> jax.Array:
        """Estimate sampled rollout-cost gradients with central differences."""
        eps = self.feedback_diff_eps
        eye = jnp.eye(self.task.feedback_state_dim, dtype=state_reference.dtype)
        perturbations = jnp.concatenate([eye, -eye], axis=0)

        def _perturbed_costs(delta: jax.Array) -> jax.Array:
            rollout_state = self.task.set_feedback_state(
                state, state_reference + eps * delta
            )
            rollout_state = mjx.forward(self.task.model, rollout_state)
            return self._rollout_total_costs(rollout_state, tk, knots, dr_rng)

        shifted_costs = jax.vmap(_perturbed_costs)(perturbations)
        plus_costs, minus_costs = jnp.split(shifted_costs, 2, axis=0)
        return ((plus_costs - minus_costs) / (2.0 * eps)).T

    def _rollout_total_costs(
        self,
        rollout_state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        dr_rng: jax.Array,
    ) -> jax.Array:
        """Compute total rollout costs for a fixed initial state."""
        rollouts = self.rollout_with_randomizations(
            rollout_state, tk, knots, dr_rng
        )
        return self._sanitize_costs(jnp.sum(rollouts.costs, axis=1))

    def _feedback_gain_from_samples(
        self,
        costs: jax.Array,
        action_deltas: jax.Array,
        cost_grads: jax.Array,
    ) -> jax.Array:
        """Aggregate sampled cost gradients into an sbmpc-style gain matrix."""
        weights = self._importance_weights(costs)
        grad_shift = jnp.sum(weights[:, None] * cost_grads, axis=0)
        weight_grads = -(1.0 / self.temperature) * weights[:, None] * (
            cost_grads - grad_shift
        )
        return -self.step_size * jnp.einsum(
            "si,sj->ij", action_deltas, weight_grads
        )
