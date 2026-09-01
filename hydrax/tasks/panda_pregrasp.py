from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.cost_residuals import CostTerm, build_cost_terms, cost_sum
from hydrax.task_base import Task

# Friction feedforward, live only when with_frictionloss is False. The plant
# (sim and FR3) keeps its identified joint friction, so a frictionless planning
# model under-commands by that torque: the arm stalls mid-reach and then creeps
# for minutes. tau += f_c*tanh(v/eps) adds torque in the DIRECTION OF MOTION,
# which is what cancels it -- the opposite sign doubles friction instead.
FRICTION_FF_FRACTION = 0.8
FRICTION_FF_V_EPS = 0.04


@dataclass
class PandaPregraspOptions:
    """Configuration options for the PandaPregrasp task."""

    # --- Cost ---
    #
    # The cost is crocoddyl's CostModelSum: one weighted residual per entry,
    # picked by name from hydrax.tasks.cost_residuals.RESIDUALS. Which terms
    # are present IS the controller definition, so it is set by the
    # `costs.running` / `costs.terminal` blocks of the tuning yaml rather
    # than by editing running_cost(). See doc/cost_residuals.md.
    #
    # Weights are ABSOLUTE: used as given, NOT normalized against each other.
    # Their ratio sets the trade-off and their scale sets the cost's
    # curvature, which is what the sampling temperature is measured against
    # -- so changing them requires revisiting solver.temperature.
    #
    # The defaults are the configuration that measured 1.6 mm terminal in
    # sim: crocoddyl's canonical reaching proportions (tracking 10, state
    # regularization 1e-1, control regularization 1e-4), tracking a joint
    # plan. The terminal drops the control term, as crocoddyl's terminal
    # CostModelSum does -- there is no control at that node to regularize.
    running_costs: Tuple[CostTerm, ...] = (
        CostTerm("joint_position", (10.0,), "plan"),
        CostTerm("joint_velocity", (0.1,), "plan"),
        CostTerm("control", (1e-4,), "gravity"),
    )
    terminal_costs: Tuple[CostTerm, ...] = (
        CostTerm("joint_position", (10.0,), "plan"),
        CostTerm("joint_velocity", (0.1,), "plan"),
    )

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

    # Control period (s), and the planning model's physics timestep: rollouts
    # take one physics step per control step. The solver's num_steps then sets
    # the prediction horizon, control_period * num_steps. The bridge's
    # publish_rate_hz must equal 1 / control_period or the plan clock drifts.
    control_period: float = 0.04

    # Keep the identified joint frictionloss in the PLANNING model. The model
    # file is never edited; False zeroes it at derivation. Contacts are
    # disabled, so this is the only constraint source: dropping it measured
    # 30.1 -> 10.7 ms per solve, but 3.14 -> 21.66 mm terminal error and an
    # arm still moving at 0.11 rad/s (2026-08-07, 3 seeds).
    with_frictionloss: bool = True

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

    # Prediction horizon in control steps. The horizon in seconds is
    # num_steps * PandaPregraspOptions.control_period; the pairing glue
    # passes that product to the controller as plan_horizon.
    num_steps: int = 8
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

        # Built WITH friction whatever the option, then zeroed here: zeroing in
        # the deriver would destroy the identified values, and the friction
        # feedforward below needs them.
        mj_model = self._derive_arm_planning_model(
            options.start_q,
            solver_iterations=options.mujoco_solver_iterations,
            solver_ls_iterations=options.mujoco_solver_ls_iterations,
            with_frictionloss=True,
            control_period=options.control_period,
        )
        if options.with_frictionloss:
            self.friction_ff_tau = np.zeros(mj_model.nu, dtype=np.float64)
        else:
            self.friction_ff_tau = np.asarray(
                mj_model.dof_frictionloss[: mj_model.nu], dtype=np.float64
            ).copy()
            mj_model.dof_frictionloss[:] = 0.0
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

        # The site the end-effector residuals measure. trace_sites already
        # makes site_xpos available inside the rollout, so the residual is a
        # lookup, not an extra kinematics pass.
        self.gripper_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
        )
        self.goal_pos = jnp.array(options.goal_pos, dtype=jnp.float32)
        self.goal_rot = jnp.array(options.goal_rot, dtype=jnp.float32)

        # Resolve the yaml's term list against this model: lengths checked,
        # scalar weights broadcast, control terms rejected at the terminal.
        # Weights are used as given and are deliberately NOT normalized to
        # sum to 1 -- that normalization made them nearly untunable, since
        # w_q was already 0.90 of the total and could only ever reach 1.0, a
        # hard ceiling of 1.111x no matter what the yaml held (measured
        # 2026-08-07; it explains the earlier finding that
        # configuration_weight x10 moved the gains by only 1.08x). It also
        # hid the absolute cost scale the sampling temperature is matched
        # against.
        #
        # Composition happens here, at construction, so a term the yaml
        # omits is absent from the compiled graph rather than multiplied
        # by zero in it.
        self._running_terms = build_cost_terms(
            self, options.running_costs, "costs.running", allow_control=True
        )
        self._terminal_terms = build_cost_terms(
            self, options.terminal_costs, "costs.terminal", allow_control=False
        )

    @staticmethod
    def _derive_arm_planning_model(
        home_q,
        *,
        solver_iterations: int = PandaPregraspOptions.mujoco_solver_iterations,
        solver_ls_iterations: int = PandaPregraspOptions.mujoco_solver_ls_iterations,
        with_frictionloss: bool = PandaPregraspOptions.with_frictionloss,
        control_period: float = PandaPregraspOptions.control_period,
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
        spec.option.timestep = control_period
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        # See PandaPregraspOptions.mujoco_solver_iterations for why this is
        # not MuJoCo's 100/50 default and what it costs to lower it further.
        spec.option.iterations = solver_iterations
        spec.option.ls_iterations = solver_ls_iterations
        spec.add_key(name="home", qpos=list(home_q))
        # Plan parameters carried on mjx.Data.userdata: [q0 (nq), duration].
        # userdata is a traced field that mjx.step passes through untouched,
        # so the reference plan reaches running_cost as an ARGUMENT rather
        # than a trace-time constant. That is what makes the compiled rollout
        # plan-agnostic: any start configuration reuses the same executable
        # (and the same compilation-cache entry) instead of recompiling.
        # userdata layout, see plan_userdata:
        #   q0 | duration | goal_q | goal_pos | goal_rot (row-major 3x3)
        spec.nuserdata = 2 * len(home_q) + 13
        model = spec.compile()
        if not with_frictionloss:
            model.dof_frictionloss[:] = 0.0
        return model

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


    def friction_feedforward(self, v: np.ndarray) -> np.ndarray:
        """Torque cancelling the plant's joint friction; zeros if it is modelled.

        ``friction_ff_tau`` is zero whenever with_frictionloss is True, so this
        is identically zero for a planner that already models friction.
        """
        return FRICTION_FF_FRACTION * self.friction_ff_tau * np.tanh(
            np.asarray(v, dtype=np.float64).reshape(-1) / FRICTION_FF_V_EPS
        )

    def plan_userdata(
        self,
        q0: np.ndarray,
        duration: float,
        goal_q: np.ndarray | None = None,
        goal_pos: np.ndarray | None = None,
        goal_rot: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pack the task parameters for mjx.Data.userdata.

        Layout:
        ``[q0 (nq) | duration (1) | goal_q (nq) | goal_pos (3) | goal_rot (9)]``
        with ``goal_rot`` row-major.

        userdata is a traced field that mjx.step passes through untouched, so
        everything here is re-parameterizable at runtime WITHOUT recompiling.
        The goal travels with the plan for exactly that reason: it will come
        from vision, and a goal baked into the compiled graph would mean a new
        HLO and a full recompile per goal.

        The goal arguments default to the construction-time goal, so a caller
        that only re-anchors the plan start keeps the current target.
        """
        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
        goal_q = self.goal_q if goal_q is None else goal_q
        goal_pos = self.goal_pos if goal_pos is None else goal_pos
        goal_rot = self.goal_rot if goal_rot is None else goal_rot
        return np.concatenate(
            [
                q0,
                [float(duration)],
                np.asarray(goal_q, dtype=np.float64).reshape(-1),
                np.asarray(goal_pos, dtype=np.float64).reshape(-1),
                np.asarray(goal_rot, dtype=np.float64).reshape(-1),
            ]
        )

    def _plan_params(
        self, state: mjx.Data
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """(q0, duration, goal_q) of the reference plan, read off the state.

        Falls back to the construction-time plan when userdata is unset (all
        zero duration), so the task still works outside the ROS adapter.
        """
        nq = self.model.nq
        q0, duration = state.userdata[:nq], state.userdata[nq]
        goal_q = state.userdata[nq + 1 : 2 * nq + 1]
        unset = duration <= 0.0
        return (
            jnp.where(unset, jnp.asarray(self.start_q, dtype=q0.dtype), q0),
            jnp.where(unset, self.duration, duration),
            jnp.where(unset, jnp.asarray(self.goal_q, dtype=q0.dtype), goal_q),
        )

    def _goal_pos(self, state: mjx.Data) -> jax.Array:
        """The end-effector goal position, read off the state.

        Same fallback rule as _plan_params: an unset userdata (zero duration)
        means the construction-time goal.
        """
        nq = self.model.nq
        goal_pos = state.userdata[2 * nq + 1 : 2 * nq + 4]
        unset = state.userdata[nq] <= 0.0
        return jnp.where(
            unset, jnp.asarray(self.goal_pos, dtype=goal_pos.dtype), goal_pos
        )

    def _goal_rot(self, state: mjx.Data) -> jax.Array:
        """The end-effector goal orientation (3x3), read off the state."""
        nq = self.model.nq
        goal_rot = state.userdata[2 * nq + 4 : 2 * nq + 13].reshape(3, 3)
        unset = state.userdata[nq] <= 0.0
        return jnp.where(
            unset, jnp.asarray(self.goal_rot, dtype=goal_rot.dtype), goal_rot
        )

    def _reference(self, state: mjx.Data) -> Tuple[jax.Array, jax.Array]:
        """Minimum-jerk reference (q, v) at state.time, evaluated in closed form.

        The quintic is analytic in its start and goal, so the plan is fully
        described by (q0, duration, goal_q) on userdata instead of a sampled
        table baked into the compiled graph. Past the plan end it holds the
        goal at zero velocity, matching the previous table lookup's index
        clamp.
        """
        q0, duration, goal_q = self._plan_params(state)
        dq = goal_q - q0
        tau = jnp.clip(state.time / duration, 0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        return q0 + s * dq, ds * dq

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ): the `costs.running` terms."""
        # Which residuals are summed here is a yaml decision, not a code one
        # -- see cost_residuals.RESIDUALS and doc/cost_residuals.md.
        #
        # The default "quadratic" activation is ActivationModelWeightedQuad
        # (0.5 Σ w_i r_i², constant Hessian). It replaced "saturated"
        # (1 - exp(-Σ w_i r_i²)) as the default because that form's curvature
        # vanishes as the error grows: measured at the state where the real
        # FR3 parks, d2J/du2 was ~1e-4 with three NEGATIVE entries, so the
        # optimizer could not see the control. On hardware that showed up as
        # tau_ff on the gravity-free joints (j1/j3/j5/j7) being noise about
        # zero — 87-148 sign flips per 750 solves, never reaching the
        # 0.33-0.89 N.m breakaway friction — and the arm crept instead of
        # moving. The quadratic form measures 1.7-2.5x breakaway with zero
        # sign flips at that same state (2026-08-07).
        return cost_sum(self, self._running_terms, state, control)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T): the `costs.terminal` terms, NOT dt-scaled.

        The absent dt mirrors crocoddyl's integration convention: running
        nodes are IntegratedActionModelEuler(model, dt) and contribute
        dt*l(x,u), while the terminal node is
        IntegratedActionModelEuler(model, 0.0) and contributes l_T(x)
        unscaled. alg_base already applies that split (it sums dt*running_cost
        over the rollout and adds terminal_cost once), so a dt here was a
        second scaling that made the terminal cost weigh the SAME as one
        running step -- 1/dt = 25x weaker than the convention, leaving nothing
        that rewards arriving.

        Measured (2026-08-07, closed-loop offline, 3 seeds, plant at the
        sbmpc_ros sim settings): removing that factor moved the reach from
        66.7 mm to 9.4 mm at t = 11.5 s, and the time to cross 10 mm from
        ~60 s to ~13 s. Torque and velocity margins were unchanged (max 0.33
        of the torque limit, 0.12 of the velocity limit).

        Control-dependent residuals are rejected from this block at
        construction, as crocoddyl's terminal CostModelSum drops its control
        regularization: there is no control at this node to regularize.
        """
        return cost_sum(self, self._terminal_terms, state, None)

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
