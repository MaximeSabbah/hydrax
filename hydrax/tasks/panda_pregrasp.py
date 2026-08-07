from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


# Activations selectable by PandaPregraspOptions.cost_activation, applied to
# each squared residual |r|^2. Crocoddyl's ActivationModel, in the two forms
# this task has used.
_ACTIVATIONS = {
    # ActivationModelWeightedQuad: a(r) = 0.5 r^T diag(w) r, Hessian diag(w)
    "quadratic": lambda squared_error: 0.5 * squared_error,
    # Bounds each term to [0, 1); curvature -> 0 as the error grows
    "saturated": lambda squared_error: 1.0 - jnp.exp(-squared_error),
}


@dataclass
class PandaPregraspOptions:
    """Configuration options for the PandaPregrasp task."""

    # --- Cost weights ---
    #
    # ABSOLUTE weights of a quadratic (crocoddyl-style) cost: they are used
    # as given, NOT normalized against each other. Their ratio sets the
    # trade-off and their scale sets the cost's curvature, which is what the
    # sampling temperature is measured against -- so changing them requires
    # revisiting solver.temperature.
    #
    # Proportions follow crocoddyl's canonical reaching OCPs (tracking 10,
    # state regularization 1e-1, control regularization 1e-4): the control
    # term is a regularizer, orders of magnitude below the tracking terms,
    # not a co-equal objective.

    # Which activation wraps each weighted squared residual, mirroring
    # crocoddyl's pluggable ActivationModel (the residual, the weights and
    # the activation are three separate choices there, so they are here too):
    #
    #   "quadratic" -> 0.5 * |r|^2   — crocoddyl's ActivationModelWeightedQuad,
    #                  constant Hessian, the default and what the weights
    #                  below are tuned for.
    #   "saturated" -> 1 - exp(-|r|^2)  — bounds each term to [0, 1]. Kept
    #                  available, but note its curvature vanishes as the
    #                  error grows, which is what made the OCP blind to the
    #                  control (see running_cost). If you select it, retune
    #                  the weights and solver.temperature with it.
    cost_activation: str = "quadratic"

    # Joint configuration (qpos) tracking
    configuration_cost_weight: float = 10.0

    # Joint velocity (qvel) tracking
    velocity_cost_weight: float = 0.1

    # Control regularization around the feedforward torque plan, scaled by
    # the torque limits so the error is dimensionless (equivalently, a
    # crocoddyl weighted-quadratic activation with w_i = 1 / tau_max_i^2)
    control_cost_weight: float = 1e-4

    # --- Task geometry ---

    # Pregrasp position: 7.5 cm above the pick-and-place scene's canonical
    # cube-center safety reference (z = 0.105 m).
    goal_pos: Tuple[float, float, float] = (0.5, 0.0, 0.18)

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

    # --- Planning model constraint solver ---

    # MuJoCo's Newton solver budget for the derived planning model, set by
    # the `model:` section of the tuning yaml.
    #
    # These were MuJoCo's defaults (100/50) while the planning model carried
    # no constraints at all — contacts are disabled, so the solver had
    # nothing to converge and the budget was never paid. Joint frictionloss
    # adds one constraint per joint, and MJX cannot early-exit a solver loop
    # the way C MuJoCo does: it runs every iteration it is given, for every
    # sample of every rollout step. At 100/50 that measured ~125 ms per
    # control step against a 40 ms budget.
    #
    # 5/8 (what the ROS plant scene already uses) converges seven friction
    # constraints to machine precision — 2.8e-17 rad over one 8-step
    # horizon against a fully-converged reference. Below that accuracy
    # falls off a cliff: 3/5 gives 7.1e-3 rad and 2/4 gives 2.9e-2, because
    # the line search, not the Newton count, is what binds.
    mujoco_solver_iterations: int = 5
    mujoco_solver_ls_iterations: int = 8

    # --- Reference plan ---

    # Nominal reach duration (s); stretched if the peak plan velocity would
    # exceed max_velocity_fraction of the velocity limits
    duration_sec: float = 7.5

    # Cap on the plan's peak joint velocity, as a fraction of the limits
    max_velocity_fraction: float = 0.20

    # --- Deployment low level (not used by the task's costs/dynamics) ---

    # Fixed joint-impedance gains of the feedforward-mode 1 kHz law (LFC
    # with constant gains). Real LFC configuration values; single source
    # for the Tier A example loop and the ROS planner adapter. Set by the
    # `feedforward:` section of the tuning yaml.
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

    The model is the 7-DoF planning variant of models/panda/panda.xml,
    derived at load time (see _derive_arm_planning_model): direct torque
    motors at the Franka limits, contacts disabled, timestep = the 25 Hz
    control period, so rollouts take one physics step per control step.

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
        if options is None:
            options = PandaPregraspOptions()
        self.options = options

        mj_model = self._derive_arm_planning_model(
            options.start_q,
            solver_iterations=options.mujoco_solver_iterations,
            solver_ls_iterations=options.mujoco_solver_ls_iterations,
        )
        super().__init__(mj_model, trace_sites=["gripper"], impl=impl)

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

        # Cost weights are used as given. They are deliberately NOT
        # normalized to sum to 1: that normalization made the weights nearly
        # untunable, since w_q was already 0.90 of the total and could only
        # ever reach 1.0 -- a hard ceiling of 1.111x no matter what value the
        # yaml held, whatever the intent (measured 2026-08-07; it explains
        # the earlier finding that configuration_weight x10 moved the gains
        # by only 1.08x). It also hid the absolute cost scale that the
        # sampling temperature has to be matched against.
        self.configuration_cost_weight = options.configuration_cost_weight
        self.velocity_cost_weight = options.velocity_cost_weight
        self.control_cost_weight = options.control_cost_weight

        if options.cost_activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown cost_activation {options.cost_activation!r}; "
                f"expected one of {sorted(_ACTIVATIONS)}"
            )
        self.cost_activation = options.cost_activation

    @staticmethod
    def _derive_arm_planning_model(
        home_q,
        *,
        solver_iterations: int = PandaPregraspOptions.mujoco_solver_iterations,
        solver_ls_iterations: int = PandaPregraspOptions.mujoco_solver_ls_iterations,
    ) -> mujoco.MjModel:
        """The 7-DoF PLANNING variant of models/panda/panda.xml.

        There is one robot description (panda.xml: arm + articulated
        gripper, torque motors, position-servo gripper). The MPPI plans
        only the arm, so its model is derived here at load time: finger
        joints removed (bodies and inertia stay — the hand keeps its true
        mass), gripper actuator and finger coupling dropped, contacts
        disabled, and the timestep set to the 25 Hz control period so
        rollouts take one physics step per control step. ``home_q``
        becomes the model's home keyframe.

        tests/test_panda_model.py pins these invariants and the arm
        parity with the plant; the derivation was asserted structurally
        identical to the previously committed pregrasp.xml before that
        file was removed (2026-07-09 model consolidation).
        """
        spec = mujoco.MjSpec.from_file(ROOT + "/models/panda/panda.xml")
        for joint in list(spec.joints):
            if joint.name in ("finger_joint1", "finger_joint2"):
                spec.delete(joint)
        for actuator in list(spec.actuators):
            if actuator.name == "actuator8":
                spec.delete(actuator)
        for equality in list(spec.equalities):
            spec.delete(equality)
        spec.option.timestep = 0.04
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        # See PandaPregraspOptions.mujoco_solver_iterations for why this is
        # not MuJoCo's 100/50 default and what it costs to lower it further.
        spec.option.iterations = solver_iterations
        spec.option.ls_iterations = solver_ls_iterations
        spec.add_key(name="home", qpos=list(home_q))
        return spec.compile()

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

        # Weighted sum of the activated squared residuals — crocoddyl's
        # CostModelSum over three CostModelResidual terms.
        #
        # The default "quadratic" activation is ActivationModelWeightedQuad
        # (0.5 |r|^2, constant Hessian). It replaced "saturated"
        # (1 - exp(-|r|^2)) as the default because that form's curvature
        # vanishes as the error grows: measured at the state where the real
        # FR3 parks, d2J/du2 was ~1e-4 with three NEGATIVE entries, so the
        # optimizer could not see the control. On hardware that showed up as
        # tau_ff on the gravity-free joints (j1/j3/j5/j7) being noise about
        # zero — 87-148 sign flips per 750 solves, never reaching the
        # 0.33-0.89 N.m breakaway friction — and the arm crept instead of
        # moving. The quadratic form measures 1.7-2.5x breakaway with zero
        # sign flips at that same state (2026-08-07).
        activation = _ACTIVATIONS[self.cost_activation]
        return (
            self.configuration_cost_weight * activation(jnp.sum(jnp.square(q_err)))
            + self.velocity_cost_weight * activation(jnp.sum(jnp.square(v_err)))
            + self.control_cost_weight * activation(jnp.sum(jnp.square(u_err)))
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        # The same cost as the running costs, evaluated at the feedforward
        # control (zero control error), and NOT scaled by dt.
        #
        # This mirrors crocoddyl's integration convention: running nodes are
        # IntegratedActionModelEuler(model, dt) and contribute dt*l(x,u),
        # while the terminal node is IntegratedActionModelEuler(model, 0.0)
        # and contributes l_T(x) unscaled. alg_base already applies that
        # split (it sums dt*running_cost over the rollout and adds
        # terminal_cost once), so the dt here was a second scaling that made
        # the terminal cost weigh the SAME as one running step -- 1/dt = 25x
        # weaker than the convention, leaving nothing that rewards arriving.
        #
        # Measured (2026-08-07, closed-loop offline, 3 seeds, plant at the
        # sbmpc_ros sim settings): removing this factor moves the reach from
        # 66.7 mm to 9.4 mm at t = 11.5 s, and the time to cross 10 mm from
        # ~60 s to ~13 s. Torque and velocity margins are unchanged
        # (max 0.33 of the torque limit, 0.12 of the velocity limit).
        #
        # The control term is dropped, as crocoddyl's terminal CostModelSum
        # drops its control regularization: there is no control at the
        # terminal node to regularize.
        q_err = state.qpos - self._get_reference_configuration(state.time)
        v_err = state.qvel - self._get_reference_velocity(state.time)
        activation = _ACTIVATIONS[self.cost_activation]
        return (
            self.configuration_cost_weight * activation(jnp.sum(jnp.square(q_err)))
            + self.velocity_cost_weight * activation(jnp.sum(jnp.square(v_err)))
        )

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
