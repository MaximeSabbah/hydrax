from enum import IntEnum
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Phase(IntEnum):
    PREGRASP = 0
    GRASP = 1
    PLACE = 2


class PickAndPlace(Task):
    """Pick-and-place task with the Franka Emika Panda.

    The robot must pick up a small box from the table and place it at a
    target location indicated by a translucent mocap body.
    """

    def __init__(self, impl: str = "jax") -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/panda/scene.xml"
        )
        super().__init__(
            mj_model,
            trace_sites=["gripper"],
            impl=impl,
        )

        # Sensor addresses
        self.ee_pos_adr = mj_model.sensor_adr[
            mj_model.sensor("ee_pos").id
        ]
        self.obj_pos_adr = mj_model.sensor_adr[
            mj_model.sensor("obj_pos").id
        ]
        self.target_pos_adr = mj_model.sensor_adr[
            mj_model.sensor("target_pos").id
        ]

        # Gripper actuator index (last actuator, index 7)
        self.gripper_idx = 7

        # Gripper fully-open value
        self.gripper_open = 0.04

        # Home joint configuration
        self.q_home = jnp.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853])

        # Hierarchical behavior settings
        self.phase = Phase.PREGRASP
        self.pregrasp_offset = jnp.array([0.0, 0.0, 0.05])  # 5 cm above object
        self.pregrasp_tol = 0.03
        self.grasp_tol = 0.03
        self.phase_hold_steps = 3
        self._phase_counter = 0

    def reset_phase(self) -> None:
        """Reset the task phase at episode start."""
        self.phase = Phase.PREGRASP
        self._phase_counter = 0

    def _get_ee_pos(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]

    def _get_obj_pos(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.obj_pos_adr : self.obj_pos_adr + 3]

    def _get_target_pos(self, state: mjx.Data) -> jax.Array:
        return state.sensordata[self.target_pos_adr : self.target_pos_adr + 3]

    def update_phase(self, state: mjx.Data, control: jax.Array) -> None:
        """Update the phase using the executed trajectory only.

        Call this from the outer MPC loop on the real state, not inside
        batched rollouts.
        """
        ee_pos = self._get_ee_pos(state)
        obj_pos = self._get_obj_pos(state)
        target_pos = self._get_target_pos(state)

        dist_ee_obj = jnp.linalg.norm(ee_pos - obj_pos)
        pregrasp_pos = obj_pos + self.pregrasp_offset
        dist_ee_pregrasp = jnp.linalg.norm(ee_pos - pregrasp_pos)
        obj_to_target = jnp.linalg.norm(obj_pos - target_pos)

        gripper_closed = control[self.gripper_idx] < 0.5 * self.gripper_open
        lifted_or_carrying = jnp.abs(obj_pos[2] - ee_pos[2]) < 0.05

        if self.phase == Phase.PREGRASP:
            if dist_ee_pregrasp < self.pregrasp_tol:
                self._phase_counter += 1
            else:
                self._phase_counter = 0

            if self._phase_counter >= self.phase_hold_steps:
                self.phase = Phase.GRASP
                self._phase_counter = 0

        elif self.phase == Phase.GRASP:
            # Move to PLACE once the hand is close, closed, and the object
            # appears to follow the gripper.
            if dist_ee_obj < self.grasp_tol and gripper_closed and lifted_or_carrying:
                self._phase_counter += 1
            else:
                self._phase_counter = 0

            if self._phase_counter >= self.phase_hold_steps:
                self.phase = Phase.PLACE
                self._phase_counter = 0

        elif self.phase == Phase.PLACE:
            # Stay in PLACE until task completion.
            # You can add a DONE phase later if needed.
            pass

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        ee_pos = self._get_ee_pos(state)
        obj_pos = self._get_obj_pos(state)
        target_pos = self._get_target_pos(state)

        vel_cost = jnp.sum(jnp.square(state.qvel[:7]))
        posture_cost = jnp.sum(jnp.square(state.qpos[:7] - self.q_home))

        if self.phase == Phase.PREGRASP:
            pregrasp_pos = obj_pos + self.pregrasp_offset
            pos_cost = jnp.sum(jnp.square(ee_pos - pregrasp_pos))

            # Keep the gripper open during approach.
            gripper_cost = jnp.square(control[self.gripper_idx] - self.gripper_open)

            total_cost = (
                10.0 * pos_cost
                + 2.0 * gripper_cost
                + 0.002 * vel_cost
                + 0.001 * posture_cost
            )

        elif self.phase == Phase.GRASP:
            reach_cost = jnp.sum(jnp.square(ee_pos - obj_pos))

            # Strongly encourage closing once close to the object.
            desired_gripper = 0.0
            gripper_cost = jnp.square(control[self.gripper_idx] - desired_gripper)

            total_cost = (
                8.0 * reach_cost
                + 8.0 * gripper_cost
                + 0.002 * vel_cost
                + 0.001 * posture_cost
            )

        else:  # Phase.PLACE
            place_cost = jnp.sum(jnp.square(obj_pos - target_pos))

            # Encourage opening after transport.
            gripper_cost = jnp.square(control[self.gripper_idx] - self.gripper_open)

            # Optional: discourage the robot from staying glued to the object.
            ee_away_cost = jnp.sum(jnp.square(ee_pos - target_pos))

            total_cost = (
                15.0 * place_cost
                + 1.0 * gripper_cost
                + 0.5 * ee_away_cost
                + 0.002 * vel_cost
                + 0.001 * posture_cost
            )

        return total_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        obj_pos = self._get_obj_pos(state)
        target_pos = self._get_target_pos(state)

        obj_cost = jnp.sum(jnp.square(obj_pos - target_pos))
        vel_cost = jnp.sum(jnp.square(state.qvel[:7]))

        return 100.0 * obj_cost + 0.05 * vel_cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(
            rng, (n_geoms,), minval=0.5, maxval=2.0
        )
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.005 * jax.random.normal(q_rng, (self.model.nq,))
        v_err = 0.005 * jax.random.normal(v_rng, (self.model.nv,))
        return {"qpos": data.qpos + q_err, "qvel": data.qvel + v_err}

    def make_data(self) -> mjx.Data:
        return super().make_data(nconmax=8000)