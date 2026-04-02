"""
Focused PREGRASP tuning script.

Only JIT-compiles the PREGRASP phase (~30s instead of ~5min).
Prints EE position error each loop so we can see convergence.
"""

import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from hydrax.algs import MPPI
from hydrax.tasks.pick_and_place import Phase, PickAndPlace

# ── task & controller ─────────────────────────────────────────────────
task = PickAndPlace()
task.phase = Phase.PREGRASP

ctrl = MPPI(
    task,
    num_samples=128,
    noise_level=0.3,
    temperature=10.0,
    step_size=0.5,
    num_randomizations=1,
    plan_horizon=0.3,
    spline_type="zero",
    num_knots=8,
    iterations=1,
)

# ── simulation setup ──────────────────────────────────────────────────
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

FREQ = 25
sim_steps_per_replan = max(int(1.0 / FREQ / mj_model.opt.timestep), 1)
step_dt = sim_steps_per_replan * mj_model.opt.timestep
actual_freq = 1.0 / step_dt

home_ctrl = jnp.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04])
initial_knots = jnp.tile(home_ctrl, (ctrl.num_knots, 1))

mjx_data = task.make_data()
mjx_data = mjx_data.replace(
    qpos=mj_data.qpos,
    qvel=mj_data.qvel,
    mocap_pos=mj_data.mocap_pos,
    mocap_quat=mj_data.mocap_quat,
)

# ── JIT only PREGRASP ────────────────────────────────────────────────
print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
print(
    f"Planning at {actual_freq:.0f} Hz, "
    f"sim at {1/mj_model.opt.timestep:.0f} Hz, "
    f"horizon {ctrl.plan_horizon}s "
    f"({ctrl.ctrl_steps} steps, {ctrl.num_knots} knots)\n"
)

print("JIT-compiling PREGRASP ...")
st = time.time()
jit_opt = jax.jit(ctrl.optimize)
jit_interp = jax.jit(ctrl.interp_func)

policy_params = ctrl.init_params(initial_knots=initial_knots)
policy_params, _ = jit_opt(mjx_data, policy_params)
policy_params, _ = jit_opt(mjx_data, policy_params)
tq = jnp.arange(sim_steps_per_replan) * mj_model.opt.timestep
jit_interp(tq, policy_params.tk, policy_params.mean[None, ...])
jit_interp(tq, policy_params.tk, policy_params.mean[None, ...])
print(f"JIT done in {time.time() - st:.1f}s\n")

# re-init params from home
policy_params = ctrl.init_params(initial_knots=initial_knots)

# ── IDs for error printout ───────────────────────────────────────────
gripper_site_id = mj_model.site("gripper").id
object_body_id = mj_model.body("object").id

# ── main loop ─────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        t0 = time.time()

        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )

        plan_t0 = time.time()
        policy_params, rollouts = jit_opt(mjx_data, policy_params)
        plan_ms = (time.time() - plan_t0) * 1000

        tq = (
            jnp.arange(sim_steps_per_replan) * mj_model.opt.timestep
            + mj_data.time
        )
        us = np.asarray(
            jit_interp(tq, policy_params.tk, policy_params.mean[None, ...])
        )[0]

        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = us[i]
            mujoco.mj_step(mj_model, mj_data)
        viewer.sync()

        # compute EE error to pregrasp goal
        ee = mj_data.site_xpos[gripper_site_id]
        obj = mj_data.xpos[object_body_id]
        goal = obj + np.array([0, 0, task.PREGRASP_HEIGHT])
        err = np.linalg.norm(ee - goal)

        loop_ms = (time.time() - t0) * 1000
        elapsed = time.time() - t0
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

        rtr = step_dt / (time.time() - t0)
        print(
            f"plan={plan_ms:5.1f}ms  "
            f"loop={loop_ms:5.1f}ms  "
            f"rtr={rtr:.2f}  "
            f"ee_err={err:.3f}m  "
            f"ee={ee[0]:.2f},{ee[1]:.2f},{ee[2]:.2f}  "
            f"goal={goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f}",
            end="\r",
        )

print("\nDone.")
