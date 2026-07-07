from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


@dataclass
class PandaPregraspOptions:
    """Configuration options for the PandaPregrasp task."""

    # --- Cost weights ---

    # Joint configuration (qpos) tracking
    configuration_cost_weight: float = 1.0

    # Joint velocity (qvel) tracking
    velocity_cost_weight: float = 0.1

    # Control tracking around the feedforward torque plan, scaled by the
    # torque limits so the error is dimensionless
    control_cost_weight: float = 0.01

    # --- Task geometry ---

    # Pregrasp position: 5 cm above the pick-and-place scene's object
    # (object top at z = 0.05)
    goal_pos: Tuple[float, float, float] = (0.5, 0.0, 0.10)

    # Pregrasp orientation (rotation matrix rows): gripper pointing down,
    # x-axis along world x
    goal_rot: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )

    # Joint configuration the reach starts from (the scene home keyframe)
    start_q: Tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

    # --- Robot limits (Franka FER) ---

    tau_max: Tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    vel_max: Tuple[float, ...] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)

    # --- Reference plan ---

    # Nominal reach duration (s); stretched if the peak plan velocity would
    # exceed max_velocity_fraction of the velocity limits
    duration_sec: float = 7.5

    # Cap on the plan's peak joint velocity, as a fraction of the limits
    max_velocity_fraction: float = 0.20

    # --- Deployment low level (not used by the task's costs/dynamics) ---

    # Fixed joint-impedance gains of the feedforward-mode 1 kHz law (LFC
    # with constant gains). Real LFC configuration values; single source
    # for the Tier A example loop and the ROS planner adapter.
    kp_fixed: Tuple[float, ...] = (1000.0, 1000.0, 1000.0, 1000.0, 20.0, 10.0, 5.0)
    kd_fixed: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 1.0)

    # --- Domain randomization ranges ---

    # Body mass: multiplicative scale drawn from [1-scale, 1+scale]
    body_mass_scale: float = 0.1

    # Center-of-mass position: additive noise drawn from [-offset, +offset] (m)
    body_ipos_offset: float = 0.005

    # Joint damping: uniform range (N·m·s/rad)
    dof_damping_range: Tuple[float, float] = (0.5, 2.0)

    # Joint friction loss: uniform range (N·m)
    dof_frictionloss_range: Tuple[float, float] = (0.0, 1.0)

    # Torque calibration: multiplicative scale drawn from [1-scale, 1+scale]
    actuator_gain_scale: float = 0.05


@dataclass
class PregraspControllerConfig:
    """Solver configuration for the pregrasp reach.

    Consumed by whoever pairs the task with its FeedbackMPPI controller
    (the example script, the ROS planner adapter). This dataclass is the
    typed schema for the ``solver:`` section of the single tuning surface,
    ``configs/pregrasp.yaml`` (loaded via
    ``hydrax.configs.load_pregrasp_config``); the defaults apply for keys
    the yaml omits.
    """

    num_samples: int = 1024

    # Per-joint sampling noise std = noise_scale * tau_max
    noise_scale: float = 0.03

    temperature: float = 0.01

    # Fraction of the softmax mean update applied per iteration (1.0 = the
    # plain MPPI update). Below 1 damps the solve-to-solve hover of tau_ff
    # around a reached goal (the hold jitter) at the cost of slower
    # convergence per iteration.
    mean_adaptation_rate: float = 1.0

    plan_horizon: float = 0.4
    spline_type: str = "cubic"
    num_knots: int = 4
    iterations: int = 1

    # Feedback-gain batch: the zero-noise nominal + the lowest-cost samples
    # used for K = du*/dx0. Only consumed when the pairing glue enables the
    # gain computation (feedback mode).
    num_gain_samples: int = 128


class PandaPregrasp(Task):
    """The Panda tracks a minimum-jerk joint plan to a pregrasp pose.

    The model (models/panda/pregrasp.xml) is the 7-DoF torque-controlled
    Panda: direct torque motors at the Franka limits, contacts disabled,
    timestep = the 25 Hz control period, so rollouts take one physics step
    per control step.

    All references are generated at construction, self-contained:
      1. damped-least-squares IK gives the goal configuration for the
         pregrasp pose in the options,
      2. a quintic (minimum-jerk) joint plan runs from the start
         configuration to the goal,
      3. feedforward torques along the plan come from MuJoCo inverse
         dynamics.

    The reference is indexed by ``state.time`` and holds the goal (at zero
    velocity, with gravity feedforward) past the end of the plan.
    """

    def __init__(
        self,
        impl: str = "jax",
        options: PandaPregraspOptions | None = None,
    ) -> None:
        """Load the MuJoCo model and build the reference plan.

        Args:
            impl: Backend to use for simulation rollouts ("jax" or "warp").
            options: Task options controlling cost weights, the task
                     geometry, the reference plan, and domain randomization
                     ranges.
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/panda/pregrasp.xml"
        )
        super().__init__(mj_model, trace_sites=["gripper"], impl=impl)

        if options is None:
            options = PandaPregraspOptions()
        self.options = options

        # Reference plan: IK goal, then a quintic joint plan sampled at the
        # control period, then inverse-dynamics feedforward torques.
        self.start_q = np.asarray(options.start_q, dtype=np.float64)
        vel_max = np.asarray(options.vel_max, dtype=np.float64)
        self.goal_q = self._solve_ik(
            np.asarray(options.goal_pos, dtype=np.float64),
            np.asarray(options.goal_rot, dtype=np.float64),
            self.start_q,
        )

        # Stretch the duration if the quintic peak velocity (1.875*dq/T)
        # would exceed the requested fraction of the velocity limits.
        dq = np.abs(self.goal_q - self.start_q)
        t_vel = 1.875 * np.max(dq / (options.max_velocity_fraction * vel_max))
        self.duration = max(options.duration_sec, float(t_vel))

        q_plan, v_plan, a_plan = self._min_jerk_plan(
            self.start_q, self.goal_q, self.duration, self.dt
        )
        tau_plan = self._inverse_dynamics(q_plan, v_plan, a_plan)

        # Convert reference data to jax arrays
        self.reference_fps = 1.0 / self.dt
        self.reference_qpos = jnp.array(q_plan, dtype=jnp.float32)
        self.reference_qvel = jnp.array(v_plan, dtype=jnp.float32)
        self.reference_ctrl = jnp.array(tau_plan, dtype=jnp.float32)
        self.tau_max = jnp.array(options.tau_max, dtype=jnp.float32)

        # Weigh different cost terms, then normalize so all terms add to 1.
        total_weights = (
            options.configuration_cost_weight
            + options.velocity_cost_weight
            + options.control_cost_weight
        )
        self.configuration_cost_weight = (
            options.configuration_cost_weight / total_weights
        )
        self.velocity_cost_weight = (
            options.velocity_cost_weight / total_weights
        )
        self.control_cost_weight = (
            options.control_cost_weight / total_weights
        )

    def _solve_ik(
        self,
        pos: np.ndarray,
        rot: np.ndarray,
        q0: np.ndarray,
        site: str = "gripper",
        iters: int = 200,
        damping: float = 1e-3,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """Damped-least-squares IK for a site pose (position + orientation)."""
        model = self.mj_model
        data = mujoco.MjData(model)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        q = q0.copy()
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for _ in range(iters):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            e_pos = pos - data.site_xpos[sid]
            R = data.site_xmat[sid].reshape(3, 3)
            e_rot = 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))
            err = np.concatenate([e_pos, e_rot])
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J = np.vstack([jacp, jacr])
            dq = np.linalg.solve(
                J.T @ J + damping * np.eye(model.nv), J.T @ err
            )
            q = np.clip(q + dq, model.jnt_range[:, 0], model.jnt_range[:, 1])
        return q

    @staticmethod
    def _min_jerk_plan(
        q0: np.ndarray,
        qf: np.ndarray,
        duration: float,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quintic (minimum-jerk) joint plan from q0 to qf: (q, v, a) at dt."""
        n = int(round(duration / dt)) + 1
        tau = np.linspace(0.0, 1.0, n)[:, None]
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
        dq = (qf - q0)[None, :]
        return q0 + s * dq, ds * dq, dds * dq

    def _inverse_dynamics(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        """Feedforward torque along the plan via mj_inverse (contact-free)."""
        tau_max = np.asarray(self.options.tau_max, dtype=np.float64)
        data = mujoco.MjData(self.mj_model)
        tau = np.zeros_like(q)
        for k in range(q.shape[0]):
            data.qpos[:] = q[k]
            data.qvel[:] = v[k]
            data.qacc[:] = a[k]
            mujoco.mj_inverse(self.mj_model, data)
            tau[k] = data.qfrc_inverse
        return np.clip(tau, -tau_max, tau_max)

    def _get_reference_configuration(self, t: jax.Array) -> jax.Array:
        """Get the reference position (q) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference_qpos.shape[0] - 1)
        return self.reference_qpos[i, :]

    def _get_reference_velocity(self, t: jax.Array) -> jax.Array:
        """Get the reference velocity (v) at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference_qvel.shape[0] - 1)
        return self.reference_qvel[i, :]

    def _get_reference_control(self, t: jax.Array) -> jax.Array:
        """Get the feedforward torque at time t."""
        i = jnp.int32(t * self.reference_fps)
        i = jnp.clip(i, 0, self.reference_ctrl.shape[0] - 1)
        return self.reference_ctrl[i, :]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        # Joint angle tracking error
        q_ref = self._get_reference_configuration(state.time)
        q_err = state.qpos - q_ref  # size (nq,)

        # Joint velocity tracking error
        v_ref = self._get_reference_velocity(state.time)
        v_err = state.qvel - v_ref  # size (nv,)

        # Control error around the feedforward plan, per-joint normalized so
        # the strong shoulder joints (87 Nm) and the wrist (12 Nm) compare
        u_ref = self._get_reference_control(state.time)
        u_err = (control - u_ref) / self.tau_max  # size (nu,)

        # Tracking costs J = 1 - exp(-|error|^2) for each error term. This
        # puts each error term between 0 and 1. (Phase 3 diagnostic, kept
        # for the record: plain quadratic terms give the same tracking to
        # within run noise — the two are identical at small errors — so
        # the saturated form is kept for its bounded scale.)
        q_squared_error = jnp.sum(jnp.square(q_err))
        q_cost = 1.0 - jnp.exp(-q_squared_error)

        v_squared_error = jnp.sum(jnp.square(v_err))
        v_cost = 1.0 - jnp.exp(-v_squared_error)

        u_squared_error = jnp.sum(jnp.square(u_err))
        u_cost = 1.0 - jnp.exp(-u_squared_error)

        # Weighted sum of the different error terms. Weights are normalized
        # so the total cost is between 0 and 1.
        return (
            self.configuration_cost_weight * q_cost
            + self.velocity_cost_weight * v_cost
            + self.control_cost_weight * u_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # We'll use the same cost as the running costs, evaluated at the
        # feedforward control (zero control error). Multiplying by the time
        # step dt ensures that the terminal cost is weighed equally with the
        # running costs; it should not be interpreted as a cost-to-go.
        u_ref = self._get_reference_control(state.time)
        return self.running_cost(state, u_ref) * self.dt

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize physical modeling parameters.

        Contact parameters are not randomized: the pregrasp model has
        contacts disabled. Actuator gain randomization scales the direct
        torque motors, modeling torque calibration error.
        """
        opts = self.options
        rng, mass_rng, ipos_rng, damping_rng, fric_rng, gain_rng = (
            jax.random.split(rng, 6)
        )

        # Body masses: multiplicative noise ±body_mass_scale
        n_bodies = self.model.body_mass.shape[0]
        mass_scale = jax.random.uniform(
            mass_rng,
            (n_bodies,),
            minval=1.0 - opts.body_mass_scale,
            maxval=1.0 + opts.body_mass_scale,
        )
        body_mass = self.model.body_mass * mass_scale

        # Center of mass positions: additive noise ±body_ipos_offset per axis
        body_ipos = self.model.body_ipos + jax.random.uniform(
            ipos_rng,
            self.model.body_ipos.shape,
            minval=-opts.body_ipos_offset,
            maxval=opts.body_ipos_offset,
        )

        # Joint damping (all 7 DOFs are actuated arm joints)
        n_dof = self.model.dof_damping.shape[0]
        dof_damping = jax.random.uniform(
            damping_rng,
            (n_dof,),
            minval=opts.dof_damping_range[0],
            maxval=opts.dof_damping_range[1],
        )

        # Joint friction loss
        dof_frictionloss = jax.random.uniform(
            fric_rng,
            (n_dof,),
            minval=opts.dof_frictionloss_range[0],
            maxval=opts.dof_frictionloss_range[1],
        )

        # Torque calibration: gainprm[:, 0] scales motor force output
        n_act = self.model.actuator_gainprm.shape[0]
        gain_scale = jax.random.uniform(
            gain_rng,
            (n_act,),
            minval=1.0 - opts.actuator_gain_scale,
            maxval=1.0 + opts.actuator_gain_scale,
        )
        actuator_gainprm = self.model.actuator_gainprm.at[:, 0].set(
            self.model.actuator_gainprm[:, 0] * gain_scale
        )

        return {
            "body_mass": body_mass,
            "body_ipos": body_ipos,
            "dof_damping": dof_damping,
            "dof_frictionloss": dof_frictionloss,
            "actuator_gainprm": actuator_gainprm,
        }

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured joint positions and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.001 * jax.random.normal(q_rng, (self.model.nq,))
        v_err = 0.01 * jax.random.normal(v_rng, (self.model.nv,))
        return {"qpos": data.qpos + q_err, "qvel": data.qvel + v_err}
