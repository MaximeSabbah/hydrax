from enum import IntEnum
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Phase(IntEnum):
    """Minimal full pick-and-place sequence."""

    PREGRASP = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    TRANSPORT = 4
    PLACE = 5
    OPEN = 6
    RETREAT = 7
    DONE = 8


_DESIRED_X = jnp.array([1.0, 0.0, 0.0])
_DESIRED_Z = jnp.array([0.0, 0.0, -1.0])


class PickAndPlace(Task):
    """Pick up the cube, move it to the target, release, and retreat."""

    PREGRASP_CLEARANCE = 0.05
    CARRY_CLEARANCE = 0.12
    PLACE_CLEARANCE = 0.005
    RETREAT_CLEARANCE = 0.05
    MIN_CLEARANCE = 0.02

    PREGRASP_POS_TOL = 0.02
    DESCEND_POS_TOL = 0.01
    CARRY_OBJ_TOL = 0.025
    PLACE_OBJ_TOL = 0.02
    RETREAT_POS_TOL = 0.03
    SUCCESS_ORI_TOL = 0.08

    PREGRASP_HOLD_STEPS = 4
    DESCEND_HOLD_STEPS = 1
    CLOSE_CONTACT_HOLD_STEPS = 4
    LIFT_HOLD_STEPS = 2
    TRANSPORT_HOLD_STEPS = 2
    PLACE_HOLD_STEPS = 2
    OPEN_HOLD_STEPS = 2
    RETREAT_HOLD_STEPS = 2

    CLOSED_FINGER_WIDTH = 0.022
    OPEN_FINGER_WIDTH = 0.035

    def __init__(self, impl: str = "jax") -> None:
        """Load the Panda model and configure the placement sequence."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/panda/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["gripper"], impl=impl)

        self.ee_pos_adr = mj_model.sensor_adr[mj_model.sensor("ee_pos").id]
        self.ee_xaxis_adr = mj_model.sensor_adr[mj_model.sensor("ee_xaxis").id]
        self.ee_zaxis_adr = mj_model.sensor_adr[mj_model.sensor("ee_zaxis").id]
        self.obj_pos_adr = mj_model.sensor_adr[mj_model.sensor("obj_pos").id]
        self.target_pos_adr = mj_model.sensor_adr[
            mj_model.sensor("target_pos").id
        ]

        self.object_body_idx = mj_model.body("object").id
        self.object_geom_idx = mj_model.geom("object_geom").id
        self.gripper_site_idx = mj_model.site("gripper").id
        self.left_pad_geom_idx = mj_model.geom("left_finger_pad").id
        self.right_pad_geom_idx = mj_model.geom("right_finger_pad").id

        self.gripper_idx = 7
        self.gripper_open = 0.04
        self.gripper_close = 0.0

        self.object_half_height = float(
            mj_model.geom_size[self.object_geom_idx, 2]
        )
        self.pregrasp_offset = jnp.array(
            [0.0, 0.0, self.object_half_height + self.PREGRASP_CLEARANCE]
        )
        self.carry_offset = jnp.array([0.0, 0.0, self.CARRY_CLEARANCE])
        self.place_offset = jnp.array([0.0, 0.0, self.PLACE_CLEARANCE])
        self.retreat_offset = jnp.array(
            [0.0, 0.0, self.object_half_height + self.RETREAT_CLEARANCE]
        )
        self._init_reference_targets()
        self._init_nominal_controls()
        self._init_phase_metadata()

        self.phase = Phase.PREGRASP
        self._hold_count = 0

    def _init_reference_targets(self) -> None:
        """Initialize the default object and target reference positions."""
        ref_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, ref_data, 0)
        mujoco.mj_forward(self.mj_model, ref_data)

        self.initial_object_pos = jnp.array(
            ref_data.xpos[self.object_body_idx].copy()
        )
        self.place_height = float(self.initial_object_pos[2])
        ref_data.mocap_pos[0, 2] = self.place_height
        mujoco.mj_forward(self.mj_model, ref_data)
        self.default_target_pos = jnp.array(ref_data.mocap_pos[0].copy())

        self.pregrasp_goal_pos = self.initial_object_pos + self.pregrasp_offset
        self.descend_goal_pos = self.initial_object_pos
        self.lift_goal_pos = self.initial_object_pos + self.carry_offset
        self.transport_goal_pos = self.default_target_pos + self.carry_offset
        self.place_goal_pos = self.default_target_pos + self.place_offset
        self.retreat_goal_pos = self.default_target_pos + self.retreat_offset

    def _init_nominal_controls(self) -> None:
        """Initialize nominal arm configurations and controls per phase."""
        self.pregrasp_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.pregrasp_goal_pos)
        )
        self.descend_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.descend_goal_pos)
        )
        self.lift_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.lift_goal_pos)
        )
        self.transport_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.transport_goal_pos)
        )
        self.place_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.place_goal_pos)
        )
        self.retreat_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.retreat_goal_pos)
        )

        self.pregrasp_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.pregrasp_qpos, self.gripper_open)
        )
        self.descend_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.descend_qpos, self.gripper_open)
        )
        self.close_ctrl = self.descend_ctrl.at[self.gripper_idx].set(
            self.gripper_close
        )
        self.lift_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.lift_qpos, self.gripper_close)
        )
        self.transport_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(
                self.transport_qpos, self.gripper_close
            )
        )
        self.place_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.place_qpos, self.gripper_close)
        )
        self.open_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.place_qpos, self.gripper_open)
        )
        self.retreat_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.retreat_qpos, self.gripper_open)
        )

    def _init_phase_metadata(self) -> None:
        """Initialize phase-indexed metadata tables."""
        self.phase_goal_qpos_map = {
            Phase.PREGRASP: self.pregrasp_qpos,
            Phase.DESCEND: self.descend_qpos,
            Phase.CLOSE: self.descend_qpos,
            Phase.LIFT: self.lift_qpos,
            Phase.TRANSPORT: self.transport_qpos,
            Phase.PLACE: self.place_qpos,
            Phase.OPEN: self.place_qpos,
            Phase.RETREAT: self.retreat_qpos,
            Phase.DONE: self.retreat_qpos,
        }
        self.phase_goal_ctrl_map = {
            Phase.PREGRASP: self.pregrasp_ctrl,
            Phase.DESCEND: self.descend_ctrl,
            Phase.CLOSE: self.close_ctrl,
            Phase.LIFT: self.lift_ctrl,
            Phase.TRANSPORT: self.transport_ctrl,
            Phase.PLACE: self.place_ctrl,
            Phase.OPEN: self.open_ctrl,
            Phase.RETREAT: self.retreat_ctrl,
            Phase.DONE: self.retreat_ctrl,
        }
        self.phase_hold_steps_map = {
            Phase.PREGRASP: self.PREGRASP_HOLD_STEPS,
            Phase.DESCEND: self.DESCEND_HOLD_STEPS,
            Phase.CLOSE: self.CLOSE_CONTACT_HOLD_STEPS,
            Phase.LIFT: self.LIFT_HOLD_STEPS,
            Phase.TRANSPORT: self.TRANSPORT_HOLD_STEPS,
            Phase.PLACE: self.PLACE_HOLD_STEPS,
            Phase.OPEN: self.OPEN_HOLD_STEPS,
            Phase.RETREAT: self.RETREAT_HOLD_STEPS,
            Phase.DONE: 0,
        }
        self.phase_next_map = {
            Phase.PREGRASP: Phase.DESCEND,
            Phase.DESCEND: Phase.CLOSE,
            Phase.CLOSE: Phase.LIFT,
            Phase.LIFT: Phase.TRANSPORT,
            Phase.TRANSPORT: Phase.PLACE,
            Phase.PLACE: Phase.OPEN,
            Phase.OPEN: Phase.RETREAT,
            Phase.RETREAT: Phase.DONE,
            Phase.DONE: Phase.DONE,
        }
        self.phase_cost_weights_map = {
            Phase.PREGRASP: (1.0, 0.0, 250.0),
            Phase.DESCEND: (1.2, 0.2, 0.0),
            Phase.CLOSE: (1.0, 0.4, 0.0),
            Phase.LIFT: (0.8, 1.6, 0.0),
            Phase.TRANSPORT: (0.8, 1.8, 0.0),
            Phase.PLACE: (1.0, 2.0, 0.0),
            Phase.OPEN: (0.7, 2.4, 0.0),
            Phase.RETREAT: (0.8, 2.2, 0.0),
            Phase.DONE: (0.0, 0.0, 0.0),
        }
        self.phase_joint_weights_map = {
            Phase.PREGRASP: (35.0, 6.0),
            Phase.DESCEND: (35.0, 6.0),
            Phase.CLOSE: (35.0, 6.0),
            Phase.LIFT: (20.0, 4.0),
            Phase.TRANSPORT: (20.0, 4.0),
            Phase.PLACE: (20.0, 4.0),
            Phase.OPEN: (15.0, 3.0),
            Phase.RETREAT: (15.0, 3.0),
            Phase.DONE: (15.0, 3.0),
        }
        self.phase_grip_weights_map = {
            Phase.PREGRASP: 20.0,
            Phase.DESCEND: 20.0,
            Phase.CLOSE: 80.0,
            Phase.LIFT: 20.0,
            Phase.TRANSPORT: 20.0,
            Phase.PLACE: 20.0,
            Phase.OPEN: 100.0,
            Phase.RETREAT: 20.0,
            Phase.DONE: 20.0,
        }

    def _solve_nominal_goal_qpos(self, goal_pos: jax.Array) -> np.ndarray:
        """Solve a nominal IK pose for a world-space goal position."""
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, 0)
        self.sync_target(mj_data)
        mujoco.mj_forward(self.mj_model, mj_data)

        goal_rot = np.diag([1.0, -1.0, -1.0])
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))

        for _ in range(100):
            mujoco.mj_forward(self.mj_model, mj_data)
            site_rot = mj_data.site_xmat[self.gripper_site_idx].reshape(3, 3)
            ee_pos = mj_data.site_xpos[self.gripper_site_idx]
            pos_err = np.asarray(goal_pos) - ee_pos
            ori_err = 0.5 * (
                np.cross(site_rot[:, 0], goal_rot[:, 0])
                + np.cross(site_rot[:, 1], goal_rot[:, 1])
                + np.cross(site_rot[:, 2], goal_rot[:, 2])
            )
            if (
                np.linalg.norm(pos_err) < 1e-5
                and np.linalg.norm(ori_err) < 1e-5
            ):
                break

            mujoco.mj_jacSite(
                self.mj_model, mj_data, jacp, jacr, self.gripper_site_idx
            )
            jacobian = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-3 * np.eye(6),
                np.concatenate([pos_err, ori_err]),
            )
            mj_data.qpos[:7] += np.clip(dq, -0.05, 0.05)
            mj_data.qpos[:7] = np.clip(
                mj_data.qpos[:7],
                self.mj_model.jnt_range[:7, 0],
                self.mj_model.jnt_range[:7, 1],
            )

        return mj_data.qpos[:7].copy()

    def _solve_nominal_goal_ctrl(
        self, goal_qpos: jax.Array, finger_target: float
    ) -> np.ndarray:
        """Compensate for steady-state arm error under position control."""
        ctrl = np.concatenate(
            [np.asarray(goal_qpos), np.array([finger_target])]
        )

        for _ in range(6):
            mj_data = mujoco.MjData(self.mj_model)
            mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, 0)
            self.sync_target(mj_data)

            for _ in range(500):
                mj_data.ctrl[:] = ctrl
                mujoco.mj_step(self.mj_model, mj_data)

            joint_err = np.asarray(goal_qpos) - mj_data.qpos[:7]
            if np.linalg.norm(joint_err) < 1e-5:
                break

            ctrl[:7] += joint_err
            ctrl = np.clip(
                ctrl,
                self.mj_model.actuator_ctrlrange[:, 0],
                self.mj_model.actuator_ctrlrange[:, 1],
            )

        ctrl[self.gripper_idx] = finger_target
        return ctrl

    def sync_target(self, mj_data: mujoco.MjData) -> None:
        """Clamp the place marker to a floor-level object placement height."""
        mj_data.mocap_pos[0, 2] = self.place_height

    def _ee(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]

    def _ee_xaxis(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_xaxis_adr : self.ee_xaxis_adr + 3]

    def _ee_zaxis(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_zaxis_adr : self.ee_zaxis_adr + 3]

    def _obj(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.obj_pos_adr : self.obj_pos_adr + 3]

    def _tgt(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.target_pos_adr : self.target_pos_adr + 3]

    def _phase_goal_qpos(self, phase: Phase) -> jax.Array:
        return self.phase_goal_qpos_map[phase]

    def phase_goal_ctrl(self, phase: Phase | None = None) -> jax.Array:
        """Return the nominal control target for a phase."""
        phase = self.phase if phase is None else phase
        return self.phase_goal_ctrl_map[phase]

    def initial_knots(
        self, num_knots: int, phase: Phase | None = None
    ) -> jax.Array:
        """Return a phase-centered spline initialization."""
        return jnp.tile(self.phase_goal_ctrl(phase), (num_knots, 1))

    def _ee_goal_position(self, state: mjx.Data, phase: Phase) -> jax.Array:
        if phase == Phase.PREGRASP:
            return self._obj(state) + self.pregrasp_offset
        if phase in (Phase.DESCEND, Phase.CLOSE):
            return self._obj(state)
        if phase == Phase.LIFT:
            return self.lift_goal_pos
        if phase == Phase.TRANSPORT:
            return self._tgt(state) + self.carry_offset
        if phase in (Phase.PLACE, Phase.OPEN):
            return self._tgt(state) + self.place_offset
        return self._tgt(state) + self.retreat_offset

    def _object_goal_position(self, state: mjx.Data, phase: Phase) -> jax.Array:
        if phase in (Phase.PREGRASP, Phase.DESCEND, Phase.CLOSE):
            return self._obj(state)
        if phase == Phase.LIFT:
            return self.lift_goal_pos
        if phase == Phase.TRANSPORT:
            return self._tgt(state) + self.carry_offset
        if phase == Phase.PLACE:
            return self._tgt(state) + self.place_offset
        return self._tgt(state)

    def goal_position(
        self, state: mjx.Data, phase: Phase | None = None
    ) -> jax.Array:
        """Return the current phase end-effector goal in world coordinates."""
        phase = self.phase if phase is None else phase
        return self._ee_goal_position(state, phase)

    def object_goal_position(
        self, state: mjx.Data, phase: Phase | None = None
    ) -> jax.Array:
        """Return the current phase object goal in world coordinates."""
        phase = self.phase if phase is None else phase
        return self._object_goal_position(state, phase)

    def _target_from_data(self, mj_data: mujoco.MjData) -> jax.Array:
        target = jnp.array(mj_data.mocap_pos[0])
        return target.at[2].set(self.place_height)

    def goal_position_from_data(
        self, mj_data: mujoco.MjData, phase: Phase | None = None
    ) -> jax.Array:
        """Goal position helper for the Python simulation loop."""
        phase = self.phase if phase is None else phase
        obj_pos = jnp.array(mj_data.xpos[self.object_body_idx])
        target_pos = self._target_from_data(mj_data)

        if phase == Phase.PREGRASP:
            return obj_pos + self.pregrasp_offset
        if phase in (Phase.DESCEND, Phase.CLOSE):
            return obj_pos
        if phase == Phase.LIFT:
            return self.lift_goal_pos
        if phase == Phase.TRANSPORT:
            return target_pos + self.carry_offset
        if phase in (Phase.PLACE, Phase.OPEN):
            return target_pos + self.place_offset
        return target_pos + self.retreat_offset

    def object_goal_from_data(
        self, mj_data: mujoco.MjData, phase: Phase | None = None
    ) -> jax.Array:
        """Object goal helper for the Python simulation loop."""
        phase = self.phase if phase is None else phase
        obj_pos = jnp.array(mj_data.xpos[self.object_body_idx])
        target_pos = self._target_from_data(mj_data)

        if phase in (Phase.PREGRASP, Phase.DESCEND, Phase.CLOSE):
            return obj_pos
        if phase == Phase.LIFT:
            return self.lift_goal_pos
        if phase == Phase.TRANSPORT:
            return target_pos + self.carry_offset
        if phase == Phase.PLACE:
            return target_pos + self.place_offset
        return target_pos

    def reset_phase(self) -> None:
        """Reset the phase machine to the pregrasp stage."""
        self.phase = Phase.PREGRASP
        self._hold_count = 0

    def _axis_alignment_cost(
        self, axis: jax.Array, target_axis: jax.Array
    ) -> jax.Array:
        return 1.0 - jnp.clip(jnp.dot(axis, target_axis), -1.0, 1.0)

    def _orientation_cost(self, state: mjx.Data) -> jax.Array:
        z_cost = self._axis_alignment_cost(self._ee_zaxis(state), _DESIRED_Z)
        x_cost = self._axis_alignment_cost(self._ee_xaxis(state), _DESIRED_X)
        return z_cost + 0.5 * x_cost

    def _l2_cost(self, vec: jax.Array) -> jax.Array:
        return jnp.sqrt(jnp.sum(jnp.square(vec)) + 1e-8)

    def _ee_position_cost(self, state: mjx.Data, phase: Phase) -> jax.Array:
        goal = self._ee_goal_position(state, phase)
        pos_err = goal - self._ee(state)
        xy_err = self._l2_cost(pos_err[:2])
        z_err = jnp.abs(pos_err[2])
        return 100.0 * xy_err + 70.0 * z_err

    def _object_position_cost(self, state: mjx.Data, phase: Phase) -> jax.Array:
        goal = self._object_goal_position(state, phase)
        return 120.0 * self._l2_cost(goal - self._obj(state))

    def _clearance_cost(self, state: mjx.Data) -> jax.Array:
        cube_top = self._obj(state)[2] + self.object_half_height
        ee_height = self._ee(state)[2]
        violation = jax.nn.relu(cube_top + self.MIN_CLEARANCE - ee_height)
        return jnp.square(violation)

    def _velocity_cost(self, state: mjx.Data) -> jax.Array:
        return jnp.sum(jnp.square(state.qvel[:8]))

    def _joint_state_cost(self, state: mjx.Data, phase: Phase) -> jax.Array:
        return jnp.sum(
            jnp.square(state.qpos[:7] - self._phase_goal_qpos(phase))
        )

    def _joint_control_cost(
        self, control: jax.Array, phase: Phase
    ) -> jax.Array:
        goal_ctrl = self.phase_goal_ctrl(phase)
        return jnp.sum(jnp.square(control[:7] - goal_ctrl[:7]))

    def _phase_gripper_target(self, phase: Phase) -> float:
        if phase in (
            Phase.PREGRASP,
            Phase.DESCEND,
            Phase.OPEN,
            Phase.RETREAT,
            Phase.DONE,
        ):
            return self.gripper_open
        return self.gripper_close

    def _gripper_cost(self, control: jax.Array, phase: Phase) -> jax.Array:
        return jnp.square(
            control[self.gripper_idx] - self._phase_gripper_target(phase)
        )

    def _phase_cost_weights(self, phase: Phase) -> Tuple[float, float, float]:
        """Return ee/object/clearance weights for the running cost."""
        return self.phase_cost_weights_map[phase]

    def _phase_joint_weights(self, phase: Phase) -> Tuple[float, float]:
        """Return joint-state and joint-control weights for a phase."""
        return self.phase_joint_weights_map[phase]

    def _phase_grip_weight(self, phase: Phase) -> float:
        return self.phase_grip_weights_map[phase]

    def _phase_running_cost(
        self, state: mjx.Data, control: jax.Array
    ) -> jax.Array:
        phase = self.phase
        ee_w, obj_w, clearance_w = self._phase_cost_weights(phase)
        joint_w, ctrl_w = self._phase_joint_weights(phase)

        return (
            ee_w * self._ee_position_cost(state, phase)
            + obj_w * self._object_position_cost(state, phase)
            + 70.0 * self._orientation_cost(state)
            + joint_w * self._joint_state_cost(state, phase)
            + ctrl_w * self._joint_control_cost(control, phase)
            + clearance_w * self._clearance_cost(state)
            + self._phase_grip_weight(phase)
            * self._gripper_cost(control, phase)
            + 0.05 * self._velocity_cost(state)
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Running cost for the current phase of the sequence."""
        if self.phase == Phase.DONE:
            return 0.05 * self._velocity_cost(state)
        return self._phase_running_cost(state, control)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost sharpens final accuracy for the active phase."""
        phase = self.phase
        if phase == Phase.DONE:
            return 2.0 * self._velocity_cost(state)

        return (
            1000.0
            * jnp.sum(
                jnp.square(
                    self._ee_goal_position(state, phase) - self._ee(state)
                )
            )
            + 1200.0
            * jnp.sum(
                jnp.square(
                    self._object_goal_position(state, phase) - self._obj(state)
                )
            )
            + 220.0 * self._orientation_cost(state)
            + 90.0 * self._joint_state_cost(state, phase)
            + 15.0 * self._velocity_cost(state)
        )

    def pose_error_from_data(
        self, mj_data: mujoco.MjData, phase: Phase | None = None
    ) -> Tuple[float, float]:
        """Return end-effector position error and orientation cost."""
        phase = self.phase if phase is None else phase
        goal = self.goal_position_from_data(mj_data, phase)
        ee_pos = jnp.array(mj_data.site_xpos[self.gripper_site_idx])
        xmat = jnp.array(mj_data.site_xmat[self.gripper_site_idx]).reshape(3, 3)

        pos_err = float(jnp.linalg.norm(goal - ee_pos))
        z_cost = float(self._axis_alignment_cost(xmat[:, 2], _DESIRED_Z))
        x_cost = float(self._axis_alignment_cost(xmat[:, 0], _DESIRED_X))
        return pos_err, z_cost + 0.5 * x_cost

    def object_error_from_data(
        self, mj_data: mujoco.MjData, phase: Phase | None = None
    ) -> float:
        """Return the object position error for the active phase."""
        phase = self.phase if phase is None else phase
        goal = self.object_goal_from_data(mj_data, phase)
        obj_pos = jnp.array(mj_data.xpos[self.object_body_idx])
        return float(jnp.linalg.norm(goal - obj_pos))

    def finger_width_from_data(self, mj_data: mujoco.MjData) -> float:
        """Return the average finger joint width."""
        return float(0.5 * (mj_data.qpos[7] + mj_data.qpos[8]))

    def has_grasp_contacts(self, mj_data: mujoco.MjData) -> bool:
        """Check for simultaneous left/right pad contact on the cube."""
        left_contact = False
        right_contact = False

        for idx in range(mj_data.ncon):
            contact = mj_data.contact[idx]
            geom_pair = {contact.geom1, contact.geom2}

            if geom_pair == {self.left_pad_geom_idx, self.object_geom_idx}:
                left_contact = True
            if geom_pair == {self.right_pad_geom_idx, self.object_geom_idx}:
                right_contact = True

            if left_contact and right_contact:
                return True

        return False

    def _is_gripper_open(self, mj_data: mujoco.MjData) -> bool:
        return self.finger_width_from_data(mj_data) >= self.OPEN_FINGER_WIDTH

    def phase_complete(self, mj_data: mujoco.MjData) -> bool:
        """Check whether the active phase has been satisfied."""
        phase = self.phase
        if phase == Phase.DONE:
            return True

        ee_err, ori_err = self.pose_error_from_data(mj_data)
        obj_err = self.object_error_from_data(mj_data)
        result = False

        if phase == Phase.PREGRASP:
            result = (
                ee_err <= self.PREGRASP_POS_TOL
                and ori_err <= self.SUCCESS_ORI_TOL
            )
        elif phase == Phase.DESCEND:
            result = (
                ee_err <= self.DESCEND_POS_TOL
                and ori_err <= self.SUCCESS_ORI_TOL
            )
        elif phase == Phase.CLOSE:
            result = (
                self.finger_width_from_data(mj_data) <= self.CLOSED_FINGER_WIDTH
                and self.has_grasp_contacts(mj_data)
            )
        elif phase in (Phase.LIFT, Phase.TRANSPORT):
            result = (
                obj_err <= self.CARRY_OBJ_TOL
                and ori_err <= self.SUCCESS_ORI_TOL
            )
        elif phase == Phase.PLACE:
            result = (
                obj_err <= self.PLACE_OBJ_TOL
                and ee_err <= self.DESCEND_POS_TOL * 2.0
                and ori_err <= self.SUCCESS_ORI_TOL
            )
        elif phase == Phase.OPEN:
            result = (
                self._is_gripper_open(mj_data)
                and obj_err <= self.PLACE_OBJ_TOL * 1.5
            )
        else:
            result = (
                ee_err <= self.RETREAT_POS_TOL
                and ori_err <= self.SUCCESS_ORI_TOL
                and self._is_gripper_open(mj_data)
                and obj_err <= self.PLACE_OBJ_TOL * 1.5
            )

        return result

    def _phase_hold_steps(self, phase: Phase) -> int:
        return self.phase_hold_steps_map[phase]

    def goal_reached(self, mj_data: mujoco.MjData) -> bool:
        """Compatibility helper for checking current phase completion."""
        return self.phase_complete(mj_data)

    def close_stable(self) -> bool:
        """Return whether the object has been securely grasped."""
        return (
            self.phase >= Phase.LIFT
            or (
                self.phase == Phase.CLOSE
                and self._hold_count >= self.CLOSE_CONTACT_HOLD_STEPS
            )
        )

    def sequence_complete(self) -> bool:
        """Return whether the full pick-and-place sequence is complete."""
        return self.phase == Phase.DONE

    def update_phase(self, mj_data: mujoco.MjData) -> bool:
        """Advance through the pick-and-place phases."""
        old_phase = self.phase

        if self.phase_complete(mj_data):
            self._hold_count += 1
        else:
            self._hold_count = 0

        if self._hold_count >= self._phase_hold_steps(self.phase):
            self.phase = self.phase_next_map[self.phase]
            if self.phase != old_phase:
                self._hold_count = 0

        changed = self.phase != old_phase
        if changed:
            print(f"\n>>> Phase: {old_phase.name} -> {self.phase.name}")
        return changed

    def make_data(self) -> mjx.Data:
        """Allocate MJX data with enough contact capacity for the scene."""
        return super().make_data(nconmax=200)
