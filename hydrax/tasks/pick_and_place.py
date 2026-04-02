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
    """Compatibility enum for the single reaching phase."""

    REACH = 0


_DESIRED_X = jnp.array([1.0, 0.0, 0.0])
_DESIRED_Z = jnp.array([0.0, 0.0, -1.0])


class PickAndPlace(Task):
    """Reach a top-down pregrasp pose above the cube with the Panda arm."""

    GOAL_CLEARANCE = 0.05
    MIN_CLEARANCE = 0.02
    SUCCESS_POS_TOL = 0.02
    SUCCESS_ORI_TOL = 0.08

    def __init__(self, impl: str = "jax") -> None:
        """Load the Panda model and configure the reaching objective."""
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

        self.gripper_idx = 7
        self.gripper_open = 0.04

        self.home_ctrl = jnp.array(mj_model.key_ctrl[0])
        self.home_qpos = jnp.array(mj_model.key_qpos[0, :7])
        self.object_half_height = float(
            mj_model.geom_size[self.object_geom_idx, 2]
        )
        self.goal_offset = jnp.array(
            [0.0, 0.0, self.object_half_height + self.GOAL_CLEARANCE]
        )
        self.goal_qpos = jnp.array(self._solve_nominal_goal_qpos())
        self.goal_ctrl = jnp.array(self._solve_nominal_goal_ctrl())

        self.phase = Phase.REACH

    def _solve_nominal_goal_qpos(self) -> np.ndarray:
        """Solve a nominal IK pose for the default reaching target."""
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, 0)
        mujoco.mj_forward(self.mj_model, mj_data)

        goal_pos = (
            mj_data.xpos[self.object_body_idx].copy()
            + np.asarray(self.goal_offset)
        )
        goal_rot = np.diag([1.0, -1.0, -1.0])
        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))

        for _ in range(64):
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
            damp = 1e-3
            dq = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damp * np.eye(6), err
            )
            dq = np.clip(dq, -0.05, 0.05)
            mj_data.qpos[:7] += dq
            mj_data.qpos[:7] = np.clip(
                mj_data.qpos[:7],
                self.mj_model.jnt_range[:7, 0],
                self.mj_model.jnt_range[:7, 1],
            )

        return mj_data.qpos[:7].copy()

    def _solve_nominal_goal_ctrl(self) -> np.ndarray:
        """Compensate for actuator steady-state error at the IK target."""
        ctrl = np.concatenate(
            [np.asarray(self.goal_qpos), np.array([self.gripper_open])]
        )

        for _ in range(6):
            mj_data = mujoco.MjData(self.mj_model)
            mujoco.mj_resetDataKeyframe(self.mj_model, mj_data, 0)

            for _ in range(500):
                mj_data.ctrl[:] = ctrl
                mujoco.mj_step(self.mj_model, mj_data)

            joint_err = np.asarray(self.goal_qpos) - mj_data.qpos[:7]
            if np.linalg.norm(joint_err) < 1e-5:
                break

            ctrl[:7] += joint_err
            ctrl = np.clip(
                ctrl,
                self.mj_model.actuator_ctrlrange[:, 0],
                self.mj_model.actuator_ctrlrange[:, 1],
            )

        ctrl[self.gripper_idx] = self.gripper_open
        return ctrl

    def _ee(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]

    def _ee_xaxis(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_xaxis_adr : self.ee_xaxis_adr + 3]

    def _ee_zaxis(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_zaxis_adr : self.ee_zaxis_adr + 3]

    def _obj(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.obj_pos_adr : self.obj_pos_adr + 3]

    def goal_position(self, state: mjx.Data) -> jax.Array:
        """Target point 5 cm above the cube's top face."""
        return self._obj(state) + self.goal_offset

    def goal_position_from_data(self, mj_data: mujoco.MjData) -> jax.Array:
        """Goal position helper for the Python simulation loop."""
        return jnp.array(mj_data.xpos[self.object_body_idx]) + self.goal_offset

    def reset_phase(self) -> None:
        """Compatibility shim for the old example API."""
        self.phase = Phase.REACH

    def update_phase(self, mj_data: mujoco.MjData) -> bool:
        """Compatibility shim for the old example API."""
        del mj_data
        return False

    def _pose_error(self, state: mjx.Data) -> jax.Array:
        return self.goal_position(state) - self._ee(state)

    def _axis_alignment_cost(
        self, axis: jax.Array, target_axis: jax.Array
    ) -> jax.Array:
        return 1.0 - jnp.clip(jnp.dot(axis, target_axis), -1.0, 1.0)

    def _orientation_cost(self, state: mjx.Data) -> jax.Array:
        z_cost = self._axis_alignment_cost(self._ee_zaxis(state), _DESIRED_Z)
        x_cost = self._axis_alignment_cost(self._ee_xaxis(state), _DESIRED_X)
        return z_cost + 0.5 * x_cost

    def _position_cost(self, state: mjx.Data) -> jax.Array:
        pos_err = self._pose_error(state)
        xy_err = jnp.sqrt(jnp.sum(jnp.square(pos_err[:2])) + 1e-8)
        z_err = jnp.abs(pos_err[2])
        return 80.0 * xy_err + 50.0 * z_err

    def _clearance_cost(self, state: mjx.Data) -> jax.Array:
        cube_top = self._obj(state)[2] + self.object_half_height
        ee_height = self._ee(state)[2]
        violation = jax.nn.relu(cube_top + self.MIN_CLEARANCE - ee_height)
        return jnp.square(violation)

    def _velocity_cost(self, state: mjx.Data) -> jax.Array:
        return jnp.sum(jnp.square(state.qvel[:8]))

    def _joint_state_cost(self, state: mjx.Data) -> jax.Array:
        return jnp.sum(jnp.square(state.qpos[:7] - self.goal_qpos))

    def _joint_control_cost(self, control: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(control[:7] - self.goal_ctrl[:7]))

    def _gripper_cost(self, control: jax.Array) -> jax.Array:
        return jnp.square(control[self.gripper_idx] - self.gripper_open)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Running cost for stable top-down pose reaching."""
        return (
            self._position_cost(state)
            + 60.0 * self._orientation_cost(state)
            + 30.0 * self._joint_state_cost(state)
            + 5.0 * self._joint_control_cost(control)
            + 250.0 * self._clearance_cost(state)
            + 2.0 * self._gripper_cost(control)
            + 0.05 * self._velocity_cost(state)
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost sharpens final pose accuracy and settling."""
        pos_err = self._pose_error(state)
        return (
            900.0 * jnp.sum(jnp.square(pos_err))
            + 180.0 * self._orientation_cost(state)
            + 80.0 * self._joint_state_cost(state)
            + 10.0 * self._velocity_cost(state)
        )

    def pose_error_from_data(
        self, mj_data: mujoco.MjData
    ) -> Tuple[float, float]:
        """Return Euclidean position error and orientation cost."""
        goal = self.goal_position_from_data(mj_data)
        ee_pos = jnp.array(mj_data.site_xpos[self.gripper_site_idx])
        xmat = jnp.array(mj_data.site_xmat[self.gripper_site_idx]).reshape(3, 3)

        pos_err = float(jnp.linalg.norm(goal - ee_pos))
        z_cost = float(self._axis_alignment_cost(xmat[:, 2], _DESIRED_Z))
        x_cost = float(self._axis_alignment_cost(xmat[:, 0], _DESIRED_X))
        return pos_err, z_cost + 0.5 * x_cost

    def goal_reached(self, mj_data: mujoco.MjData) -> bool:
        """Check whether the end effector is at the target pose."""
        pos_err, ori_err = self.pose_error_from_data(mj_data)
        return (
            pos_err <= self.SUCCESS_POS_TOL
            and ori_err <= self.SUCCESS_ORI_TOL
        )

    def make_data(self) -> mjx.Data:
        """Allocate MJX data with enough contact capacity for the scene."""
        return super().make_data(nconmax=200)
