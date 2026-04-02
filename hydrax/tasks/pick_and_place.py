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
    """Minimal pregrasp-to-grasp sequence."""

    PREGRASP = 0
    DESCEND = 1
    CLOSE = 2


_DESIRED_X = jnp.array([1.0, 0.0, 0.0])
_DESIRED_Z = jnp.array([0.0, 0.0, -1.0])


class PickAndPlace(Task):
    """Reach above the cube, descend onto it, then close the gripper."""

    PREGRASP_CLEARANCE = 0.05
    DESCEND_CLEARANCE = 0.0
    MIN_CLEARANCE = 0.02
    PREGRASP_POS_TOL = 0.02
    DESCEND_POS_TOL = 0.01
    SUCCESS_ORI_TOL = 0.08
    PREGRASP_HOLD_STEPS = 4
    DESCEND_HOLD_STEPS = 1
    CLOSE_CONTACT_HOLD_STEPS = 4
    CLOSED_FINGER_WIDTH = 0.022

    def __init__(self, impl: str = "jax") -> None:
        """Load the Panda model and configure the three-phase sequence."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/panda/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["gripper"], impl=impl)

        self.ee_pos_adr = mj_model.sensor_adr[mj_model.sensor("ee_pos").id]
        self.ee_xaxis_adr = mj_model.sensor_adr[mj_model.sensor("ee_xaxis").id]
        self.ee_zaxis_adr = mj_model.sensor_adr[mj_model.sensor("ee_zaxis").id]
        self.obj_pos_adr = mj_model.sensor_adr[mj_model.sensor("obj_pos").id]

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
        self.descend_offset = jnp.array([0.0, 0.0, self.DESCEND_CLEARANCE])

        self.pregrasp_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.pregrasp_offset)
        )
        self.pregrasp_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.pregrasp_qpos, self.gripper_open)
        )
        self.descend_qpos = jnp.array(
            self._solve_nominal_goal_qpos(self.descend_offset)
        )
        self.descend_ctrl = jnp.array(
            self._solve_nominal_goal_ctrl(self.descend_qpos, self.gripper_open)
        )
        self.close_ctrl = self.descend_ctrl.at[self.gripper_idx].set(
            self.gripper_close
        )

        self.phase = Phase.PREGRASP
        self._hold_count = 0

    def _solve_nominal_goal_qpos(self, goal_offset: jax.Array) -> np.ndarray:
        """Solve a nominal IK pose for a cube-relative goal offset."""
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, 0)
        mujoco.mj_forward(self.mj_model, mj_data)

        goal_pos = (
            mj_data.xpos[self.object_body_idx].copy() + np.asarray(goal_offset)
        )
        goal_rot = np.diag([1.0, -1.0, -1.0])
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))

        for _ in range(80):
            mujoco.mj_forward(self.mj_model, mj_data)
            site_rot = mj_data.site_xmat[self.gripper_site_idx].reshape(3, 3)
            pos_err = goal_pos - mj_data.site_xpos[self.gripper_site_idx]
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
            err = np.concatenate([pos_err, ori_err])
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 1e-3 * np.eye(6), err
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

    def _ee(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]

    def _ee_xaxis(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_xaxis_adr : self.ee_xaxis_adr + 3]

    def _ee_zaxis(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_zaxis_adr : self.ee_zaxis_adr + 3]

    def _obj(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.obj_pos_adr : self.obj_pos_adr + 3]

    def _phase_offset(self, phase: Phase) -> jax.Array:
        if phase == Phase.PREGRASP:
            return self.pregrasp_offset
        return self.descend_offset

    def _phase_goal_qpos(self, phase: Phase) -> jax.Array:
        if phase == Phase.PREGRASP:
            return self.pregrasp_qpos
        return self.descend_qpos

    def phase_goal_ctrl(self, phase: Phase | None = None) -> jax.Array:
        """Return the nominal control target for a phase."""
        phase = self.phase if phase is None else phase
        if phase == Phase.PREGRASP:
            return self.pregrasp_ctrl
        if phase == Phase.DESCEND:
            return self.descend_ctrl
        return self.close_ctrl

    def initial_knots(
        self, num_knots: int, phase: Phase | None = None
    ) -> jax.Array:
        """Return a phase-centered spline initialization."""
        return jnp.tile(self.phase_goal_ctrl(phase), (num_knots, 1))

    def goal_position(
        self, state: mjx.Data, phase: Phase | None = None
    ) -> jax.Array:
        """Return the current phase goal position in world coordinates."""
        phase = self.phase if phase is None else phase
        return self._obj(state) + self._phase_offset(phase)

    def goal_position_from_data(
        self, mj_data: mujoco.MjData, phase: Phase | None = None
    ) -> jax.Array:
        """Goal position helper for the Python simulation loop."""
        phase = self.phase if phase is None else phase
        obj_pos = jnp.array(mj_data.xpos[self.object_body_idx])
        return obj_pos + self._phase_offset(phase)

    def reset_phase(self) -> None:
        """Reset the phase machine to the pregrasp stage."""
        self.phase = Phase.PREGRASP
        self._hold_count = 0

    def _pose_error(
        self, state: mjx.Data, phase: Phase | None = None
    ) -> jax.Array:
        phase = self.phase if phase is None else phase
        return self.goal_position(state, phase) - self._ee(state)

    def _axis_alignment_cost(
        self, axis: jax.Array, target_axis: jax.Array
    ) -> jax.Array:
        return 1.0 - jnp.clip(jnp.dot(axis, target_axis), -1.0, 1.0)

    def _orientation_cost(self, state: mjx.Data) -> jax.Array:
        z_cost = self._axis_alignment_cost(self._ee_zaxis(state), _DESIRED_Z)
        x_cost = self._axis_alignment_cost(self._ee_xaxis(state), _DESIRED_X)
        return z_cost + 0.5 * x_cost

    def _position_cost(self, state: mjx.Data, phase: Phase) -> jax.Array:
        pos_err = self._pose_error(state, phase)
        xy_err = jnp.sqrt(jnp.sum(jnp.square(pos_err[:2])) + 1e-8)
        z_err = jnp.abs(pos_err[2])
        return 100.0 * xy_err + 70.0 * z_err

    def _clearance_cost(self, state: mjx.Data) -> jax.Array:
        cube_top = self._obj(state)[2] + self.object_half_height
        ee_height = self._ee(state)[2]
        violation = jax.nn.relu(cube_top + self.MIN_CLEARANCE - ee_height)
        return jnp.square(violation)

    def _object_motion_cost(self, state: mjx.Data, phase: Phase) -> jax.Array:
        if phase == Phase.PREGRASP:
            return jnp.array(0.0)

        desired_obj = self.goal_position(state, phase) - self._phase_offset(
            phase
        )
        return jnp.sum(jnp.square(self._obj(state) - desired_obj))

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

    def _gripper_cost(self, control: jax.Array, desired: float) -> jax.Array:
        return jnp.square(control[self.gripper_idx] - desired)

    def _pose_hold_cost(
        self,
        state: mjx.Data,
        control: jax.Array,
        phase: Phase,
        gripper_target: float,
        clearance_weight: float,
        object_weight: float,
    ) -> jax.Array:
        return (
            self._position_cost(state, phase)
            + 70.0 * self._orientation_cost(state)
            + 35.0 * self._joint_state_cost(state, phase)
            + 6.0 * self._joint_control_cost(control, phase)
            + clearance_weight * self._clearance_cost(state)
            + object_weight * self._object_motion_cost(state, phase)
            + 2.0 * self._gripper_cost(control, gripper_target)
            + 0.05 * self._velocity_cost(state)
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Running cost for the current phase of the sequence."""
        if self.phase == Phase.PREGRASP:
            return self._pose_hold_cost(
                state,
                control,
                Phase.PREGRASP,
                self.gripper_open,
                clearance_weight=250.0,
                object_weight=0.0,
            )

        if self.phase == Phase.DESCEND:
            return self._pose_hold_cost(
                state,
                control,
                Phase.DESCEND,
                self.gripper_open,
                clearance_weight=0.0,
                object_weight=30.0,
            )

        return self._pose_hold_cost(
            state,
            control,
            Phase.CLOSE,
            self.gripper_close,
            clearance_weight=0.0,
            object_weight=60.0,
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost sharpens final accuracy for the active phase."""
        phase = self.phase
        pos_err = self._pose_error(state, phase)
        return (
            1000.0 * jnp.sum(jnp.square(pos_err))
            + 220.0 * self._orientation_cost(state)
            + 100.0 * self._joint_state_cost(state, phase)
            + 15.0 * self._velocity_cost(state)
            + 80.0 * self._object_motion_cost(state, phase)
        )

    def pose_error_from_data(
        self, mj_data: mujoco.MjData, phase: Phase | None = None
    ) -> Tuple[float, float]:
        """Return Euclidean position error and orientation cost."""
        phase = self.phase if phase is None else phase
        goal = self.goal_position_from_data(mj_data, phase)
        ee_pos = jnp.array(mj_data.site_xpos[self.gripper_site_idx])
        xmat = jnp.array(mj_data.site_xmat[self.gripper_site_idx]).reshape(3, 3)

        pos_err = float(jnp.linalg.norm(goal - ee_pos))
        z_cost = float(self._axis_alignment_cost(xmat[:, 2], _DESIRED_Z))
        x_cost = float(self._axis_alignment_cost(xmat[:, 0], _DESIRED_X))
        return pos_err, z_cost + 0.5 * x_cost

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

    def phase_complete(self, mj_data: mujoco.MjData) -> bool:
        """Check whether the active phase has been satisfied."""
        if self.phase == Phase.CLOSE:
            return (
                self.finger_width_from_data(mj_data) <= self.CLOSED_FINGER_WIDTH
                and self.has_grasp_contacts(mj_data)
            )

        pos_err, ori_err = self.pose_error_from_data(mj_data)
        pos_tol = (
            self.PREGRASP_POS_TOL
            if self.phase == Phase.PREGRASP
            else self.DESCEND_POS_TOL
        )
        return pos_err <= pos_tol and ori_err <= self.SUCCESS_ORI_TOL

    def goal_reached(self, mj_data: mujoco.MjData) -> bool:
        """Compatibility helper for checking current phase completion."""
        return self.phase_complete(mj_data)

    def close_stable(self) -> bool:
        """Return whether the close phase has been stably achieved."""
        return (
            self.phase == Phase.CLOSE
            and self._hold_count >= self.CLOSE_CONTACT_HOLD_STEPS
        )

    def update_phase(self, mj_data: mujoco.MjData) -> bool:
        """Advance from pregrasp to descend to close when goals are met."""
        old_phase = self.phase

        if self.phase_complete(mj_data):
            self._hold_count += 1
        else:
            self._hold_count = 0

        if self.phase == Phase.PREGRASP:
            if self._hold_count >= self.PREGRASP_HOLD_STEPS:
                self.phase = Phase.DESCEND
                self._hold_count = 0
        elif self.phase == Phase.DESCEND:
            if self._hold_count >= self.DESCEND_HOLD_STEPS:
                self.phase = Phase.CLOSE
                self._hold_count = 0
        elif self.phase == Phase.CLOSE:
            self._hold_count = min(
                self._hold_count, self.CLOSE_CONTACT_HOLD_STEPS
            )

        changed = self.phase != old_phase
        if changed:
            print(f"\n>>> Phase: {old_phase.name} -> {self.phase.name}")
        return changed

    def make_data(self) -> mjx.Data:
        """Allocate MJX data with enough contact capacity for the scene."""
        return super().make_data(nconmax=200)
