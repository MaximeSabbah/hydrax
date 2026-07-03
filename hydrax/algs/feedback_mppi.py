from typing import Tuple

import jax
import jax.numpy as jnp

from hydrax.algs.mppi import MPPI, MPPIParams
from hydrax.task_base import Task


class FeedbackMPPI(MPPI):
    """MPPI with per-actuator noise and a zero-noise nominal sample.

    This is the controller of the Feedback-MPPI port
    (doc/feedback_mppi_panda_port_plan.md), extending plain MPPI with two
    sampling changes:

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
      the feedback-gain computation this class exists for: the gains
      K = du*/dx (added in Phase 2 of the port plan) are a softmax-weighted
      combination of deviations from exactly this nominal.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_std: jax.Array,
        temperature: float,
        **kwargs,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_std: Per-actuator std of the Gaussian sampling noise,
                       shape (nu,).
            temperature: The temperature parameter λ. Higher values take a
                         more even average over the samples.
            **kwargs: Remaining SamplingBasedController options
                      (plan_horizon, spline_type, num_knots, iterations, ...).
        """
        super().__init__(
            task,
            num_samples=num_samples,
            noise_level=1.0,  # unused: noise_std takes its place
            temperature=temperature,
            **kwargs,
        )
        self.noise_std = jnp.asarray(noise_std, dtype=jnp.float32)
        assert self.noise_std.shape == (task.model.nu,)

    def sample_knots(self, params: MPPIParams) -> Tuple[jax.Array, MPPIParams]:
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
