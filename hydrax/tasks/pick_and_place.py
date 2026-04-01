from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class PickAndPlace(Task):
    """Pick-and-place task with the Franka Emika Panda.

    The robot must pick up a small box from the table and place it at a
    target location indicated by a translucent mocap body.
    """

    def __init__(self, impl: str = "jax") -> None:
        """Load the MuJoCo model and set task parameters."""
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

        # Proximity kernel width for gripper cost (meters)
        self.sigma = 0.03

        # Home joint configuration (for regularization)
        self.q_home = jnp.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853])

    def _get_ee_pos(self, state: mjx.Data) -> jax.Array:
        """End-effector position in world frame."""
        return state.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]

    def _get_obj_pos(self, state: mjx.Data) -> jax.Array:
        """Object position in world frame."""
        return state.sensordata[self.obj_pos_adr : self.obj_pos_adr + 3]

    def _get_target_pos(self, state: mjx.Data) -> jax.Array:
        """Target position in world frame."""
        return state.sensordata[
            self.target_pos_adr : self.target_pos_adr + 3
        ]

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost l(x_t, u_t).

        Combines reaching, transport, gripper, velocity, and posture costs
        for stable MPPI-based pick-and-place.
        """
        ee_pos = self._get_ee_pos(state)
        obj_pos = self._get_obj_pos(state)
        target_pos = self._get_target_pos(state)

        # 1. Reach: drive end-effector toward the object
        reach_err = ee_pos - obj_pos
        reach_cost = jnp.sum(jnp.square(reach_err))

        # 2. Transport: drive the object toward the target
        obj_err = obj_pos - target_pos
        obj_cost = jnp.sum(jnp.square(obj_err))

        # 3. Gripper: proximity-based open/close
        #    Close (0) when EE is near the object, open (0.04) when far
        proximity = jnp.exp(
            -jnp.sum(jnp.square(reach_err)) / (2.0 * self.sigma**2)
        )
        desired_gripper = self.gripper_open * (1.0 - proximity)
        gripper_cost = jnp.square(control[self.gripper_idx] - desired_gripper)

        # 4. Joint velocity damping (prevents aggressive motions)
        vel_cost = jnp.sum(jnp.square(state.qvel[:7]))

        # 5. Posture regularization (stay near home when not needed)
        posture_cost = jnp.sum(jnp.square(control[:7] - self.q_home))

        return (
            5.0 * reach_cost
            + 10.0 * obj_cost
            + 2.0 * gripper_cost
            + 0.01 * vel_cost
            + 0.001 * posture_cost
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost phi(x_T).

        Strongly penalizes object being far from target at end of horizon.
        """
        obj_pos = self._get_obj_pos(state)
        target_pos = self._get_target_pos(state)
        ee_pos = self._get_ee_pos(state)
        reach_err = ee_pos - obj_pos
        return 50.0 * jnp.sum(jnp.square(obj_pos - target_pos)) + 10.0 * jnp.sum(jnp.square(reach_err))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize friction for robustness to grasp uncertainty."""
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
        """Add small noise to the state estimate."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.005 * jax.random.normal(q_rng, (self.model.nq,))
        v_err = 0.005 * jax.random.normal(v_rng, (self.model.nv,))
        return {"qpos": data.qpos + q_err, "qvel": data.qvel + v_err}

    def make_data(self) -> mjx.Data:
        """Create state with enough contact slots for grasping."""
        return super().make_data(nconmax=8000)
