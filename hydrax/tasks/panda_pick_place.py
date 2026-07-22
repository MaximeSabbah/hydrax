from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import jax.numpy as jnp
import mujoco
import numpy as np

from hydrax.task_base import Task
from hydrax.tasks.panda_pregrasp import PandaPregrasp


class Phase(IntEnum):
    """The pick-and-place sequence (sbmpc panda_pick_and_place structure)."""

    PREGRASP = 0  # over the cube
    DESCEND = 1  # gripper straddles the cube
    CLOSE = 2  # dwell: gripper closes
    LIFT = 3  # cube up to carry height
    TRANSPORT = 4  # carry to over the target
    PLACE = 5  # down onto the target
    OPEN = 6  # dwell: gripper opens
    RETREAT = 7  # clear of the placed cube
    DONE = 8  # hold


# Gripper commanded closed from CLOSE entry until OPEN entry
_GRIPPER_CLOSED_PHASES = (Phase.CLOSE, Phase.LIFT, Phase.TRANSPORT, Phase.PLACE)

# Dwell phases hold the previous goal while the gripper actuates; their
# boundary is time-only (the dwell duration is built into the timeline).
_DWELL_PHASES = (Phase.CLOSE, Phase.OPEN)

# Fixed task-success contract. These are deliberately code-level invariants,
# not controller/OCP tuning parameters: a motion phase may finish only while
# the measured end effector is at its goal and remains sufficiently still.
# The action-pose bound applies before both gripper commands (CLOSE and OPEN).
_ARRIVAL_EE_POSITION_TOLERANCE_M = 0.030
_ARRIVAL_EE_ORIENTATION_TOLERANCE_RAD = 0.10
_ACTION_EE_POSITION_TOLERANCE_M = 0.010
_ACTION_EE_ORIENTATION_TOLERANCE_RAD = 0.075
_EE_LINEAR_SPEED_TOLERANCE_M_S = 0.060
_EE_ANGULAR_SPEED_TOLERANCE_RAD_S = 0.20
_EE_STEADY_REQUIRED_SAMPLES = 5


@dataclass(frozen=True, slots=True)
class PickPlacePhaseDiagnostics:
    """JSON-safe snapshot of the active phase-transition gate.

    Joint indices are zero-based. ``transition_status`` describes the
    current phase boundary, while ``precision_hold`` marks a stationary
    grasp/place precision window for diagnostics. It does not select the
    controller gain law.
    """

    phase: str
    next_phase: str | None
    gate_type: str
    plan_time_sec: float
    phase_elapsed_sec: float
    boundary_time_sec: float | None
    time_to_boundary_sec: float | None
    at_boundary: bool
    transition_status: str
    transition_blocked: bool
    q_error_signed_by_joint_rad: tuple[float, ...]
    q_error_max_rad: float
    q_error_joint_index: int
    velocity_by_joint_rad_s: tuple[float, ...]
    velocity_abs_max_rad_s: float
    velocity_joint_index: int
    ee_position_m: tuple[float, ...]
    ee_goal_position_m: tuple[float, ...]
    ee_position_error_signed_m: tuple[float, ...]
    ee_position_error_norm_m: float
    ee_position_tolerance_m: float | None
    ee_position_ok: bool | None
    ee_orientation_error_rad: float
    ee_orientation_tolerance_rad: float | None
    ee_orientation_ok: bool | None
    ee_linear_velocity_m_s: tuple[float, ...]
    ee_linear_speed_m_s: float
    ee_linear_speed_tolerance_m_s: float | None
    ee_linear_speed_ok: bool | None
    ee_angular_velocity_rad_s: tuple[float, ...]
    ee_angular_speed_rad_s: float
    ee_angular_speed_tolerance_rad_s: float | None
    ee_angular_speed_ok: bool | None
    transition_blockers: tuple[str, ...]
    consecutive_eligible_cycles: int | None
    consecutive_required_cycles: int | None
    consecutive_ok: bool | None
    precision_hold: bool
    gripper_command: str


@dataclass
class PandaPickPlaceOptions:
    """Configuration options for the PandaPickPlace task."""

    # --- Cost weights (the certified pregrasp tracking cost) ---

    configuration_cost_weight: float = 1.0
    velocity_cost_weight: float = 0.1
    control_cost_weight: float = 0.01

    # --- Task geometry ---

    # Cube and placement poses on the real setup (deployment decision
    # 2026-07-09: fixed config values, no perception). The 0.105 m z value
    # is the reviewed table-clearance safety reference. The MuJoCo plant
    # keeps its dynamic cube at that height with body gravity compensation;
    # it deliberately does not model or certify table collisions.
    cube_pos: Tuple[float, float, float] = (0.5, 0.0, 0.105)
    target_pos: Tuple[float, float, float] = (0.65, 0.0, 0.105)

    # EE goal offsets from the cube/target centers, per phase. The grasp
    # offset is zero: the gripper site sits between the finger pads, so
    # site = cube center straddles the cube.
    pregrasp_offset: Tuple[float, float, float] = (0.0, 0.0, 0.075)
    grasp_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    carry_offset: Tuple[float, float, float] = (0.0, 0.0, 0.15)
    # Release with the cube essentially seated: the cube migrates up to
    # ~1 cm within the grip during the run (P-A3 measured), so a higher
    # release drops it and scatters the placement by the bounce
    place_offset: Tuple[float, float, float] = (0.0, 0.0, 0.002)
    retreat_offset: Tuple[float, float, float] = (0.0, 0.0, 0.15)

    # EE orientation for every phase: gripper pointing down
    goal_rot: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )

    # Joint configuration the sequence starts from (the scene home keyframe)
    start_q: Tuple[float, ...] = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)

    # --- Robot limits (Franka FER) ---

    tau_max: Tuple[float, ...] = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    vel_max: Tuple[float, ...] = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)

    # --- Reference timeline ---

    # Cap on each segment's peak joint velocity, as a fraction of the limits
    max_velocity_fraction: float = 0.20

    # Floor on each motion segment's duration: short precision moves
    # (descend, place) stretch to this instead of running at the cap
    min_segment_sec: float = 1.5

    # Duration of the CLOSE/OPEN dwells (sim value; the ROS bridge
    # additionally waits for the gripper action result)
    dwell_sec: float = 1.0

    # The dwells are settle-then-actuate: the reference remains stationary
    # for the first dwell_settle_sec before the gripper actuates. The
    # controller mode and gain calculation remain unchanged throughout.
    # During OPEN's settle the gripper stays closed (settle at the place
    # pose while holding, then release).
    dwell_settle_sec: float = 0.5

    # --- Deployment low level (not used by the task's costs/dynamics) ---

    kp_fixed: Tuple[float, ...] = (1000.0, 1000.0, 1000.0, 1000.0, 20.0, 10.0, 5.0)
    kd_fixed: Tuple[float, ...] = (5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 1.0)

    # --- Domain randomization ranges (as pregrasp) ---

    body_mass_scale: float = 0.1
    body_ipos_offset: float = 0.005
    dof_damping_range: Tuple[float, float] = (0.5, 2.0)
    dof_frictionloss_range: Tuple[float, float] = (0.0, 1.0)
    actuator_gain_scale: float = 0.05


@dataclass
class PickPlaceControllerConfig:
    """Solver configuration for the pick-and-place sequence.

    Consumed by whoever pairs the task with its FeedbackMPPI controller
    (the example script, the ROS planner adapter). This dataclass is the
    typed schema for the ``solver:`` section of the single tuning surface,
    ``configs/pick_place.yaml`` (loaded via
    ``hydrax.configs.load_pick_place_config``); the defaults apply for
    keys the yaml omits. Mirrors ``PregraspControllerConfig`` — the
    pick-and-place runs the same FeedbackMPPI controller, and the
    committed yaml starts from the frozen pregrasp values.
    """

    num_samples: int = 1024

    # Per-joint sampling noise std = noise_scale * tau_max
    noise_scale: float = 0.03

    temperature: float = 0.01

    # Fraction of the softmax mean update applied per iteration (1.0 = the
    # plain MPPI update).
    mean_adaptation_rate: float = 1.0

    plan_horizon: float = 0.4
    spline_type: str = "cubic"
    num_knots: int = 4
    iterations: int = 1

    # Feedback-gain batch: the zero-noise nominal + the lowest-cost samples
    # used for K = du*/dx0. Only consumed when the pairing glue enables the
    # gain computation (feedback mode).
    num_gain_samples: int = 128


class PandaPickPlace(PandaPregrasp):
    """The Panda tracks a phase-chained joint plan through a pick-and-place.

    Everything the controller sees is the certified pregrasp machinery,
    inherited unchanged: the same planning model (derived from
    models/panda/panda.xml), the same tracking cost, the same reference
    lookup by ``state.time``. What changes is the reference itself: ONE
    timeline chaining min-jerk segments between the phase goals (IK'd from
    the cube/target poses in the options) with constant dwell segments
    where the gripper actuates.

    Progress through the sequence is event-gated by the PLAN CLOCK, not by
    rewriting references: the reference arrays are compile-time constants
    of the jitted cost (swapping them at runtime would silently do
    nothing), so ``PickPlacePhaseMachine`` advances ``state.time`` and
    refuses to cross a segment boundary until the arm has converged on the
    segment goal. Each segment therefore starts from a converged
    (measured-to-tolerance) posture, and no phase change ever recompiles.
    """

    def __init__(
        self,
        impl: str = "jax",
        options: PandaPickPlaceOptions | None = None,
    ) -> None:
        """Load the MuJoCo model and build the phase-chained reference."""
        if options is None:
            options = PandaPickPlaceOptions()
        self.options = options

        mj_model = self._derive_arm_planning_model(options.start_q)
        # PandaPregrasp.__init__ builds the single pregrasp reach; this
        # task builds its own timeline, so it initializes the Task base
        # directly and reuses the parent's machinery (IK, quintic,
        # inverse dynamics, cost) as methods.
        Task.__init__(self, mj_model, trace_sites=["gripper"], impl=impl)

        self.start_q = np.asarray(options.start_q, dtype=np.float64)
        cube = np.asarray(options.cube_pos, dtype=np.float64)
        target = np.asarray(options.target_pos, dtype=np.float64)
        goal_rot = np.asarray(options.goal_rot, dtype=np.float64)
        vel_max = np.asarray(options.vel_max, dtype=np.float64)

        phase_goal_pos = {
            Phase.PREGRASP: cube + options.pregrasp_offset,
            Phase.DESCEND: cube + options.grasp_offset,
            Phase.CLOSE: cube + options.grasp_offset,
            Phase.LIFT: cube + options.carry_offset,
            Phase.TRANSPORT: target + options.carry_offset,
            Phase.PLACE: target + options.place_offset,
            Phase.OPEN: target + options.place_offset,
            Phase.RETREAT: target + options.retreat_offset,
        }

        # IK chain: each motion goal solved from the previous phase's goal
        # for configuration continuity; dwells reuse the previous goal.
        self.phase_goal_q = np.zeros((len(phase_goal_pos), 7))
        q_prev = self.start_q
        for phase in list(phase_goal_pos):
            if phase in _DWELL_PHASES:
                self.phase_goal_q[phase] = q_prev
            else:
                self.phase_goal_q[phase] = self._solve_ik(
                    phase_goal_pos[phase], goal_rot, q_prev
                )
                q_prev = self.phase_goal_q[phase]

        # Cache the actually achievable FK pose of every IK goal once. Runtime
        # gates compare the measured EE against these arrays, so the ROS
        # adapter and direct MuJoCo example share exactly the same task-space
        # contract without performing goal FK in the control loop.
        pose_data = mujoco.MjData(mj_model)
        site_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
        )
        self.phase_goal_ee_positions = np.zeros((len(phase_goal_pos), 3))
        self.phase_goal_ee_rotations = np.zeros((len(phase_goal_pos), 3, 3))
        for phase in list(phase_goal_pos):
            pose_data.qpos[:] = self.phase_goal_q[phase]
            pose_data.qvel[:] = 0.0
            mujoco.mj_forward(mj_model, pose_data)
            self.phase_goal_ee_positions[phase] = pose_data.site_xpos[site_id]
            self.phase_goal_ee_rotations[phase] = pose_data.site_xmat[
                site_id
            ].reshape(3, 3)

        # One timeline: min-jerk segments between goals (duration from the
        # velocity cap, floored for gentle precision moves), constant
        # segments for the dwells. Segments share their boundary sample.
        q_parts, v_parts, a_parts, end_times = [], [], [], []
        q_prev, t_end = self.start_q, 0.0
        for phase in list(phase_goal_pos):
            goal_q = self.phase_goal_q[phase]
            if phase in _DWELL_PHASES:
                n = int(round(options.dwell_sec / self.dt))
                q_seg = np.tile(goal_q, (n, 1))
                v_seg = np.zeros_like(q_seg)
                a_seg = np.zeros_like(q_seg)
                t_end += n * self.dt
            else:
                dq = np.abs(goal_q - q_prev)
                t_vel = 1.875 * np.max(
                    dq / (options.max_velocity_fraction * vel_max)
                )
                duration = max(options.min_segment_sec, float(t_vel))
                # ceil to the dt grid: rounding down would push the peak
                # velocity over the cap
                duration = float(np.ceil(duration / self.dt - 1e-9) * self.dt)
                q_seg, v_seg, a_seg = self._min_jerk_plan(
                    q_prev, goal_q, duration, self.dt
                )
                # the quintic includes its start sample = previous end
                q_seg, v_seg, a_seg = q_seg[1:], v_seg[1:], a_seg[1:]
                t_end += duration
            q_parts.append(q_seg)
            v_parts.append(v_seg)
            a_parts.append(a_seg)
            end_times.append(t_end)
            q_prev = goal_q

        q_plan = np.vstack([self.start_q[None, :], *q_parts])
        v_plan = np.vstack([np.zeros((1, 7)), *v_parts])
        a_plan = np.vstack([np.zeros((1, 7)), *a_parts])
        tau_plan = self._inverse_dynamics(q_plan, v_plan, a_plan)

        self.segment_end_times = np.asarray(end_times)
        self.duration = float(t_end)
        self.reference_fps = 1.0 / self.dt
        self.reference_qpos = jnp.array(q_plan, dtype=jnp.float32)
        self.reference_qvel = jnp.array(v_plan, dtype=jnp.float32)
        self.reference_ctrl = jnp.array(tau_plan, dtype=jnp.float32)
        self.tau_max = jnp.array(options.tau_max, dtype=jnp.float32)

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


class PickPlacePhaseMachine:
    """The event-gated plan clock driving a PandaPickPlace timeline.

    Owned by the runtime loop (the Tier A example now, the ROS planner
    adapter in P3), stepped once per 25 Hz cycle with the measured state.
    Inside a segment the clock advances normally (the solver tracks the
    moving reference, as in the pregrasp). At a segment boundary it holds
    the reference on the segment goal until the arm has converged there
    (dwell boundaries are time-only: their duration is the dwell itself),
    then crosses into the next phase. The gripper command follows the
    phase: closed from CLOSE entry until OPEN entry.
    """

    def __init__(self, task: PandaPickPlace) -> None:
        self._end_times = task.segment_end_times
        self._goals = task.phase_goal_q
        self._goal_ee_positions = task.phase_goal_ee_positions
        self._goal_ee_rotations = task.phase_goal_ee_rotations
        self._settle = float(task.options.dwell_settle_sec)
        self._dt = float(task.dt)
        self._required_cycles = _EE_STEADY_REQUIRED_SAMPLES
        # kept a python float: an np.float64 leaking into state.time
        # changes the traced dtype and silently recompiles the solver
        self.plan_time = 0.0
        # Set during the final wait at a dwell-entry boundary, once the arm
        # is inside the ordinary arrival tolerance but has not yet passed
        # the tighter grasp/place gate. This drives diagnostics only; it
        # never requests a different gain law.
        self._stalled_at_dwell_entry = False
        self._eligible_cycles = 0

    @property
    def phase(self) -> Phase:
        i = int(np.searchsorted(self._end_times, self.plan_time, side="left"))
        return Phase(min(i, int(Phase.DONE)))

    @property
    def gripper_closed(self) -> bool:
        phase = self.phase
        if phase == Phase.CLOSE:  # actuates after the stationary settle
            return self._time_into_phase() >= self._settle
        if phase == Phase.OPEN:  # holds through the settle, then releases
            return self._time_into_phase() < self._settle
        return phase in _GRIPPER_CLOSED_PHASES

    @property
    def precision_hold(self) -> bool:
        """Mark stationary grasp/place precision windows for diagnostics.

        This is true during dwells and the final waits at their entry
        boundaries. The selected controller law remains unchanged.
        """
        return self.phase in _DWELL_PHASES or self._stalled_at_dwell_entry

    def _time_into_phase(self) -> float:
        phase = self.phase
        start = float(self._end_times[phase - 1]) if phase > 0 else 0.0
        return self.plan_time - start

    def diagnostics_snapshot(
        self,
        q: np.ndarray,
        v: np.ndarray,
        ee_position: np.ndarray,
        ee_rotation: np.ndarray,
        ee_linear_velocity: np.ndarray,
        ee_angular_velocity: np.ndarray,
    ) -> PickPlacePhaseDiagnostics:
        """Report the gate inputs; joint-space values are diagnostic only."""
        phase = self.phase
        next_phase = None if phase == Phase.DONE else Phase(phase + 1).name
        goal_phase = min(phase, Phase.RETREAT)
        q_err, q_joint, v_err, v_joint = self._state_errors(q, v, goal_phase)
        ee_signed, ee_err, orientation_err = self._pose_errors(
            ee_position, ee_rotation, goal_phase
        )
        linear_velocity = np.asarray(
            ee_linear_velocity, dtype=np.float64
        ).reshape(3)
        angular_velocity = np.asarray(
            ee_angular_velocity, dtype=np.float64
        ).reshape(3)
        linear_speed = float(np.linalg.norm(linear_velocity))
        angular_speed = float(np.linalg.norm(angular_velocity))
        q_error_signed_by_joint = tuple(
            float(value)
            for value in (
                np.asarray(q, dtype=np.float64).reshape(-1)
                - self._goals[goal_phase]
            )
        )
        velocity_by_joint = tuple(
            float(value)
            for value in np.asarray(v, dtype=np.float64).reshape(-1)
        )

        blockers: tuple[str, ...] = ()
        eligible_cycles: int | None = None
        required_cycles: int | None = None
        consecutive_ok: bool | None = None
        ee_position_tol: float | None = None
        ee_orientation_tol: float | None = None
        ee_linear_speed_tol: float | None = None
        ee_angular_speed_tol: float | None = None
        ee_position_ok: bool | None = None
        ee_orientation_ok: bool | None = None
        ee_linear_speed_ok: bool | None = None
        ee_angular_speed_ok: bool | None = None
        if phase == Phase.DONE:
            gate_type = "done"
            boundary = None
            time_to_boundary = None
            at_boundary = False
            status = "done"
        else:
            boundary = float(self._end_times[phase])
            time_to_boundary = max(0.0, boundary - self.plan_time)
            at_boundary = self.plan_time >= boundary
            if phase in _DWELL_PHASES:
                gate_type = "time_only"
                status = "time_only"
            else:
                gate_type = "task_space"
                ee_position_tol, ee_orientation_tol = (
                    self._transition_pose_tolerances(phase)
                )
                ee_linear_speed_tol = _EE_LINEAR_SPEED_TOLERANCE_M_S
                ee_angular_speed_tol = _EE_ANGULAR_SPEED_TOLERANCE_RAD_S
                ee_position_ok = ee_err <= ee_position_tol
                ee_orientation_ok = orientation_err <= ee_orientation_tol
                ee_linear_speed_ok = linear_speed <= ee_linear_speed_tol
                ee_angular_speed_ok = angular_speed <= ee_angular_speed_tol
                eligible_cycles = int(self._eligible_cycles)
                required_cycles = self._required_cycles
                consecutive_ok = eligible_cycles >= required_cycles
                blockers = self._transition_blockers(
                    ee_position,
                    ee_rotation,
                    linear_velocity,
                    angular_velocity,
                    phase,
                )
                if not at_boundary:
                    status = "tracking"
                    blockers = ()
                elif blockers:
                    status = "blocked_" + "_and_".join(blockers)
                elif not consecutive_ok:
                    status = "settling"
                else:
                    status = "ready"

        transition_blocked = bool(
            at_boundary and status not in ("ready", "time_only")
        )
        return PickPlacePhaseDiagnostics(
            phase=phase.name,
            next_phase=next_phase,
            gate_type=gate_type,
            plan_time_sec=float(self.plan_time),
            phase_elapsed_sec=float(self._time_into_phase()),
            boundary_time_sec=boundary,
            time_to_boundary_sec=time_to_boundary,
            at_boundary=bool(at_boundary),
            transition_status=status,
            transition_blocked=transition_blocked,
            q_error_signed_by_joint_rad=q_error_signed_by_joint,
            q_error_max_rad=q_err,
            q_error_joint_index=q_joint,
            velocity_by_joint_rad_s=velocity_by_joint,
            velocity_abs_max_rad_s=v_err,
            velocity_joint_index=v_joint,
            ee_position_m=tuple(
                float(value)
                for value in np.asarray(ee_position, dtype=np.float64).reshape(3)
            ),
            ee_goal_position_m=tuple(
                float(value) for value in self._goal_ee_positions[goal_phase]
            ),
            ee_position_error_signed_m=tuple(float(value) for value in ee_signed),
            ee_position_error_norm_m=ee_err,
            ee_position_tolerance_m=ee_position_tol,
            ee_position_ok=ee_position_ok,
            ee_orientation_error_rad=orientation_err,
            ee_orientation_tolerance_rad=ee_orientation_tol,
            ee_orientation_ok=ee_orientation_ok,
            ee_linear_velocity_m_s=tuple(
                float(value) for value in linear_velocity
            ),
            ee_linear_speed_m_s=linear_speed,
            ee_linear_speed_tolerance_m_s=ee_linear_speed_tol,
            ee_linear_speed_ok=ee_linear_speed_ok,
            ee_angular_velocity_rad_s=tuple(
                float(value) for value in angular_velocity
            ),
            ee_angular_speed_rad_s=angular_speed,
            ee_angular_speed_tolerance_rad_s=ee_angular_speed_tol,
            ee_angular_speed_ok=ee_angular_speed_ok,
            transition_blockers=blockers,
            consecutive_eligible_cycles=eligible_cycles,
            consecutive_required_cycles=required_cycles,
            consecutive_ok=consecutive_ok,
            precision_hold=bool(self.precision_hold),
            gripper_command="close" if self.gripper_closed else "open",
        )

    def update(
        self,
        q: np.ndarray,
        v: np.ndarray,
        ee_position: np.ndarray,
        ee_rotation: np.ndarray,
        ee_linear_velocity: np.ndarray,
        ee_angular_velocity: np.ndarray,
    ) -> float:
        """Advance the clock from the EE pose/stillness contract."""
        # Joint state is intentionally not part of phase acceptance. It is
        # passed alongside the EE sample so callers use one coherent measured
        # state for update and diagnostics, and is logged by the latter.
        del q, v
        phase = self.phase
        if phase == Phase.DONE:
            self.plan_time += self._dt
            self._eligible_cycles = 0
            return self.plan_time

        boundary = self._end_times[phase]
        t_next = self.plan_time + self._dt
        if t_next < boundary:
            self.plan_time = t_next
            self._eligible_cycles = 0
            self._stalled_at_dwell_entry = False
        elif phase in _DWELL_PHASES:
            self.plan_time = t_next
            self._eligible_cycles = 0
            self._stalled_at_dwell_entry = False
        else:
            blockers = self._transition_blockers(
                ee_position,
                ee_rotation,
                ee_linear_velocity,
                ee_angular_velocity,
                phase,
            )
            self._eligible_cycles = 0 if blockers else self._eligible_cycles + 1
            if self._eligible_cycles >= self._required_cycles:
                self.plan_time = t_next
                self._eligible_cycles = 0
                self._stalled_at_dwell_entry = False
            else:
                self.plan_time = float(boundary)
                _, ee_err, orientation_err = self._pose_errors(
                    ee_position, ee_rotation, phase
                )
                self._stalled_at_dwell_entry = bool(
                    Phase(phase + 1) in _DWELL_PHASES
                    and ee_err <= _ARRIVAL_EE_POSITION_TOLERANCE_M
                    and orientation_err
                    <= _ARRIVAL_EE_ORIENTATION_TOLERANCE_RAD
                )
        return self.plan_time

    def _transition_blockers(
        self,
        ee_position: np.ndarray,
        ee_rotation: np.ndarray,
        ee_linear_velocity: np.ndarray,
        ee_angular_velocity: np.ndarray,
        phase: Phase,
    ) -> tuple[str, ...]:
        _, ee_err, orientation_err = self._pose_errors(
            ee_position, ee_rotation, phase
        )
        linear_speed = float(np.linalg.norm(ee_linear_velocity))
        angular_speed = float(np.linalg.norm(ee_angular_velocity))
        ee_position_tol, ee_orientation_tol = self._transition_pose_tolerances(
            phase
        )
        blockers = []
        if not np.isfinite(ee_err) or ee_err > ee_position_tol:
            blockers.append("ee_position")
        if not np.isfinite(orientation_err) or orientation_err > ee_orientation_tol:
            blockers.append("ee_orientation")
        if (
            not np.isfinite(linear_speed)
            or linear_speed > _EE_LINEAR_SPEED_TOLERANCE_M_S
        ):
            blockers.append("ee_linear_speed")
        if (
            not np.isfinite(angular_speed)
            or angular_speed > _EE_ANGULAR_SPEED_TOLERANCE_RAD_S
        ):
            blockers.append("ee_angular_speed")
        return tuple(blockers)

    def _transition_pose_tolerances(self, phase: Phase) -> tuple[float, float]:
        """Return position/orientation tolerances for a motion boundary."""
        next_phase = Phase(phase + 1)
        if next_phase in _DWELL_PHASES:
            return (
                _ACTION_EE_POSITION_TOLERANCE_M,
                _ACTION_EE_ORIENTATION_TOLERANCE_RAD,
            )
        return (
            _ARRIVAL_EE_POSITION_TOLERANCE_M,
            _ARRIVAL_EE_ORIENTATION_TOLERANCE_RAD,
        )

    def _pose_errors(
        self,
        ee_position: np.ndarray,
        ee_rotation: np.ndarray,
        goal_phase: Phase,
    ) -> tuple[np.ndarray, float, float]:
        position = np.asarray(ee_position, dtype=np.float64).reshape(3)
        rotation = np.asarray(ee_rotation, dtype=np.float64).reshape(3, 3)
        signed = position - self._goal_ee_positions[goal_phase]
        relative = self._goal_ee_rotations[goal_phase].T @ rotation
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        return signed, float(np.linalg.norm(signed)), float(np.arccos(cosine))

    def _state_errors(
        self, q: np.ndarray, v: np.ndarray, goal_phase: Phase
    ) -> tuple[float, int, float, int]:
        """Return maximum errors and their zero-based joint indices."""
        q_error = np.abs(
            np.asarray(q, dtype=np.float64).reshape(-1)
            - self._goals[goal_phase]
        )
        velocity = np.abs(np.asarray(v, dtype=np.float64).reshape(-1))
        q_joint = int(np.argmax(q_error))
        v_joint = int(np.argmax(velocity))
        return (
            float(q_error[q_joint]),
            q_joint,
            float(velocity[v_joint]),
            v_joint,
        )
