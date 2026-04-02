from enum import IntEnum
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Phase(IntEnum):
    """Pick-and-place sequencing phases."""

    PREGRASP = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    TRANSPORT = 4
    PLACE = 5
    OPEN = 6
    RETREAT = 7
    DONE = 8


# Desired EE z-axis: straight down [0, 0, -1]
_DESIRED_Z = jnp.array([0.0, 0.0, -1.0])


class PickAndPlace(Task):
    """Pick-and-place with the Franka Emika Panda.

    Uses a phase-based state machine inspired by STORM pick-and-place.
    Each phase produces a different cost function.  Phase lives as a
    plain Python int so the cost seen by JAX is a *static* function per
    phase; all phases are pre-JIT-compiled at startup.
    """

    # ---- geometry / tolerance knobs (metres) ----------------------------
    PREGRASP_HEIGHT = 0.10
    DESCEND_HEIGHT = 0.005
    LIFT_HEIGHT = 0.15
    RETREAT_HEIGHT = 0.10
    POS_TOL = 0.025
    HOLD_STEPS = 5

    def __init__(self, impl: str = "jax") -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/panda/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["gripper"], impl=impl)

        # sensor addresses
        self.ee_pos_adr = mj_model.sensor_adr[
            mj_model.sensor("ee_pos").id
        ]
        self.ee_zaxis_adr = mj_model.sensor_adr[
            mj_model.sensor("ee_zaxis").id
        ]
        self.obj_pos_adr = mj_model.sensor_adr[
            mj_model.sensor("obj_pos").id
        ]
        self.target_pos_adr = mj_model.sensor_adr[
            mj_model.sensor("target_pos").id
        ]

        self.gripper_idx = 7
        self.gripper_open = 0.04
        self.gripper_close = 0.0

        # mutable phase state (updated from the outer sim loop)
        self.phase: Phase = Phase.PREGRASP
        self._hold_count = 0

    # ---- sensor helpers -------------------------------------------------

    def _ee(self, s: mjx.Data) -> jax.Array:
        return s.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]

    def _ee_zaxis(self, s: mjx.Data) -> jax.Array:
        """Z-axis of the gripper frame (approach direction)."""
        return s.sensordata[self.ee_zaxis_adr : self.ee_zaxis_adr + 3]

    def _obj(self, s: mjx.Data) -> jax.Array:
        return s.sensordata[self.obj_pos_adr : self.obj_pos_adr + 3]

    def _tgt(self, s: mjx.Data) -> jax.Array:
        return s.sensordata[self.target_pos_adr : self.target_pos_adr + 3]

    def reset_phase(self) -> None:
        """Reset to initial phase."""
        self.phase = Phase.PREGRASP
        self._hold_count = 0

    # ---- phase machine (called from the sim loop on real state) ---------

    def update_phase(self, mj_data: mujoco.MjData) -> bool:
        """Advance the state machine.  Returns True when the phase changed."""
        ee = mj_data.site_xpos[self.mj_model.site("gripper").id]
        obj = mj_data.xpos[self.mj_model.body("object").id]
        tgt = mj_data.mocap_pos[0]

        old = self.phase

        if self.phase == Phase.PREGRASP:
            goal = obj + [0, 0, self.PREGRASP_HEIGHT]
            self._tick(float(jnp.linalg.norm(jnp.array(ee - goal))))
            if self._hold_count >= self.HOLD_STEPS:
                self.phase = Phase.DESCEND

        elif self.phase == Phase.DESCEND:
            goal = obj + [0, 0, self.DESCEND_HEIGHT]
            self._tick(float(jnp.linalg.norm(jnp.array(ee - goal))))
            if self._hold_count >= self.HOLD_STEPS:
                self.phase = Phase.GRASP

        elif self.phase == Phase.GRASP:
            self._hold_count += 1
            if self._hold_count >= self.HOLD_STEPS * 3:
                self.phase = Phase.LIFT

        elif self.phase == Phase.LIFT:
            goal = obj + [0, 0, self.LIFT_HEIGHT]
            self._tick(float(jnp.linalg.norm(jnp.array(ee - goal))))
            if self._hold_count >= self.HOLD_STEPS:
                self.phase = Phase.TRANSPORT

        elif self.phase == Phase.TRANSPORT:
            goal = tgt + [0, 0, self.LIFT_HEIGHT]
            self._tick(float(jnp.linalg.norm(jnp.array(ee - goal))))
            if self._hold_count >= self.HOLD_STEPS:
                self.phase = Phase.PLACE

        elif self.phase == Phase.PLACE:
            self._tick(float(jnp.linalg.norm(jnp.array(ee - tgt))))
            if self._hold_count >= self.HOLD_STEPS:
                self.phase = Phase.OPEN

        elif self.phase == Phase.OPEN:
            self._hold_count += 1
            if self._hold_count >= self.HOLD_STEPS * 3:
                self.phase = Phase.RETREAT

        elif self.phase == Phase.RETREAT:
            goal = tgt + [0, 0, self.RETREAT_HEIGHT]
            self._tick(float(jnp.linalg.norm(jnp.array(ee - goal))))
            if self._hold_count >= self.HOLD_STEPS:
                self.phase = Phase.DONE

        changed = self.phase != old
        if changed:
            self._hold_count = 0
            print(f"\n>>> Phase: {Phase(old).name} -> {self.phase.name}")
        return changed

    def _tick(self, err: float) -> None:
        if err < self.POS_TOL:
            self._hold_count += 1
        else:
            self._hold_count = 0

    # ---- cost building blocks -------------------------------------------

    def _pos_cost(self, ee: jax.Array, goal: jax.Array) -> jax.Array:
        """Position cost with STORM-style convergence dead-zone.

        Quadratic beyond a threshold, zero inside.  Prevents the
        optimizer from fighting over sub-centimetre errors that cause
        oscillation.
        """
        err_sq = jnp.sum(jnp.square(ee - goal))
        deadzone = 0.01**2  # 1 cm dead-zone radius
        return jnp.maximum(err_sq - deadzone, 0.0)

    def _ori_cost(self, state: mjx.Data) -> jax.Array:
        """Penalize deviation of EE z-axis from straight-down [0,0,-1].

        Uses ||z_ee - z_desired||^2.  Max value = 4 (opposite direction).
        """
        z_ee = self._ee_zaxis(state)
        return jnp.sum(jnp.square(z_ee - _DESIRED_Z))

    def _gripper_cost(self, ctrl: jax.Array, desired: float) -> jax.Array:
        return jnp.square(ctrl[self.gripper_idx] - desired)

    def _vel_cost(self, s: mjx.Data) -> jax.Array:
        return jnp.sum(jnp.square(s.qvel[:7]))

    def _stop_cost(
        self, s: mjx.Data, ee: jax.Array, goal: jax.Array
    ) -> jax.Array:
        """Penalize velocity more as EE nears goal (smooth convergence)."""
        dist_sq = jnp.sum(jnp.square(ee - goal))
        proximity = jnp.exp(-dist_sq / (2.0 * 0.03**2))
        return proximity * jnp.sum(jnp.square(s.qvel[:7]))

    # ---- unified move cost ----------------------------------------------

    def _move_cost(
        self,
        state: mjx.Data,
        control: jax.Array,
        goal: jax.Array,
        gripper_target: float,
        pos_w: float = 30.0,
        grip_w: float = 2.0,
        ori_w: float = 25.0,
    ) -> jax.Array:
        """Unified move-to-goal cost used by most phases."""
        ee = self._ee(state)
        return (
            pos_w * self._pos_cost(ee, goal)
            + ori_w * self._ori_cost(state)
            + grip_w * self._gripper_cost(control, gripper_target)
            + 0.02 * self._vel_cost(state)
            + 3.0 * self._stop_cost(state, ee, goal)
        )

    # ---- phase-specific costs -------------------------------------------

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Running cost — branches on the current (static) phase."""
        obj = self._obj(state)
        tgt = self._tgt(state)

        if self.phase == Phase.PREGRASP:
            goal = obj + jnp.array([0.0, 0.0, self.PREGRASP_HEIGHT])
            return self._move_cost(state, control, goal, self.gripper_open)

        if self.phase == Phase.DESCEND:
            goal = obj + jnp.array([0.0, 0.0, self.DESCEND_HEIGHT])
            return self._move_cost(state, control, goal, self.gripper_open)

        if self.phase == Phase.GRASP:
            return self._move_cost(
                state, control, obj, self.gripper_close,
                pos_w=20.0, grip_w=20.0,
            )

        if self.phase == Phase.LIFT:
            goal = obj + jnp.array([0.0, 0.0, self.LIFT_HEIGHT])
            return self._move_cost(
                state, control, goal, self.gripper_close, grip_w=10.0,
            )

        if self.phase == Phase.TRANSPORT:
            goal = tgt + jnp.array([0.0, 0.0, self.LIFT_HEIGHT])
            return self._move_cost(
                state, control, goal, self.gripper_close, grip_w=10.0,
            )

        if self.phase == Phase.PLACE:
            return self._move_cost(
                state, control, tgt, self.gripper_close, grip_w=10.0,
            )

        if self.phase == Phase.OPEN:
            return self._move_cost(
                state, control, tgt, self.gripper_open,
                pos_w=10.0, grip_w=20.0,
            )

        if self.phase == Phase.RETREAT:
            goal = tgt + jnp.array([0.0, 0.0, self.RETREAT_HEIGHT])
            return self._move_cost(state, control, goal, self.gripper_open)

        # DONE
        return 0.01 * self._vel_cost(state)

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Terminal cost mirrors the running cost goal."""
        ee = self._ee(state)
        obj = self._obj(state)
        tgt = self._tgt(state)

        if self.phase == Phase.PREGRASP:
            goal = obj + jnp.array([0.0, 0.0, self.PREGRASP_HEIGHT])
        elif self.phase == Phase.DESCEND:
            goal = obj + jnp.array([0.0, 0.0, self.DESCEND_HEIGHT])
        elif self.phase in (Phase.GRASP,):
            goal = obj
        elif self.phase == Phase.LIFT:
            goal = obj + jnp.array([0.0, 0.0, self.LIFT_HEIGHT])
        elif self.phase == Phase.TRANSPORT:
            goal = tgt + jnp.array([0.0, 0.0, self.LIFT_HEIGHT])
        elif self.phase in (Phase.PLACE, Phase.OPEN):
            goal = tgt
        elif self.phase == Phase.RETREAT:
            goal = tgt + jnp.array([0.0, 0.0, self.RETREAT_HEIGHT])
        else:
            return jnp.array(0.0)

        return (
            50.0 * self._pos_cost(ee, goal)
            + 10.0 * self._ori_cost(state)
        )

    # ---- domain randomization & data ------------------------------------

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        n_geoms = self.model.geom_friction.shape[0]
        mult = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_f = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * mult
        )
        return {"geom_friction": new_f}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.005 * jax.random.normal(q_rng, (self.model.nq,))
        v_err = 0.005 * jax.random.normal(v_rng, (self.model.nv,))
        return {"qpos": data.qpos + q_err, "qvel": data.qvel + v_err}

    def make_data(self) -> mjx.Data:
        return super().make_data(nconmax=200)
