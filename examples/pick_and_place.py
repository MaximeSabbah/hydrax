import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from hydrax.algs import MPPI
from hydrax.tasks.pick_and_place import Phase, PickAndPlace

"""
Run a minimal pregrasp -> descend -> close sequence for the Panda arm.

The task keeps the controller narrow and explicit: first stabilize 5 cm above
the cube, then descend onto the cube centerline, then hold the wrist pose while
closing the fingers onto the object.
"""

# -- task & controller -------------------------------------------------
task = PickAndPlace()

ctrl = MPPI(
    task,
    num_samples=128,
    noise_level=0.03,
    temperature=0.5,
    step_size=0.2,
    num_randomizations=1,
    plan_horizon=0.5,
    spline_type="zero",
    num_knots=6,
    iterations=1,
)

# -- simulation setup --------------------------------------------------
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

FREQ = 20
sim_steps_per_replan = max(int(1.0 / FREQ / mj_model.opt.timestep), 1)
step_dt = sim_steps_per_replan * mj_model.opt.timestep
actual_freq = 1.0 / step_dt

mj_data.mocap_pos[0] = np.asarray(task.goal_position_from_data(mj_data))
mujoco.mj_forward(mj_model, mj_data)

mjx_data = task.make_data().replace(
    qpos=jnp.array(mj_data.qpos),
    qvel=jnp.array(mj_data.qvel),
    mocap_pos=jnp.array(mj_data.mocap_pos),
    mocap_quat=jnp.array(mj_data.mocap_quat),
)

# -- trace rendering settings ------------------------------------------
SHOW_TRACES = False
MAX_TRACES = 3
TRACE_WIDTH = 5.0
TRACE_COLOR = [0.2, 0.8, 0.2, 0.15]


def warmup_phase(
    phase: Phase, controller: MPPI, warmup_data: mjx.Data
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """JIT-compile the planner for one specific task phase."""
    controller.task.phase = phase

    def _optimize(state: mjx.Data, params: Any) -> tuple[Any, Any]:
        return controller.optimize(state, params)

    def _interp(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
        return controller.interp_func(tq, tk, knots)

    jit_opt = jax.jit(_optimize)
    jit_interp = jax.jit(_interp)
    params = controller.init_params(
        initial_knots=controller.task.initial_knots(controller.num_knots, phase)
    )

    params, _ = jit_opt(warmup_data, params)
    params, _ = jit_opt(warmup_data, params)

    tq = jnp.arange(sim_steps_per_replan) * mj_model.opt.timestep
    jit_interp(tq, params.tk, params.mean[None, ...])
    jit_interp(tq, params.tk, params.mean[None, ...])

    return jit_opt, jit_interp


print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")
print(
    f"Planning at {actual_freq:.0f} Hz, "
    f"sim at {1/mj_model.opt.timestep:.0f} Hz, "
    f"horizon {ctrl.plan_horizon}s "
    f"({ctrl.ctrl_steps} steps, {ctrl.num_knots} knots, "
    f"{ctrl.iterations} iters)\n"
)

print("JIT-compiling pregrasp, descend, and close phases ...")
all_st = time.time()
phase_jit = {}
for phase in Phase:
    st = time.time()
    phase_jit[phase] = warmup_phase(phase, ctrl, mjx_data)
    print(f"  {phase.name:>9s}  {time.time() - st:.1f}s")
print(f"Warmup complete in {time.time() - all_st:.1f}s\n")

task.reset_phase()
policy_params = ctrl.init_params(
    initial_knots=task.initial_knots(ctrl.num_knots)
)

# -- main loop ---------------------------------------------------------
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    num_trace_sites = len(task.trace_site_ids)
    num_traces = min(MAX_TRACES, ctrl.num_samples)
    if SHOW_TRACES:
        for i in range(num_trace_sites * num_traces * ctrl.ctrl_steps):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array(TRACE_COLOR),
            )
            viewer.user_scn.ngeom += 1

    while viewer.is_running():
        t0 = time.time()

        mj_data.mocap_pos[0] = np.asarray(task.goal_position_from_data(mj_data))

        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )

        jit_opt, jit_interp = phase_jit[task.phase]
        rollouts = None
        if task.close_stable():
            plan_ms = 0.0
            us = np.repeat(
                np.asarray(task.phase_goal_ctrl())[None, :],
                sim_steps_per_replan,
                axis=0,
            )
        else:
            plan_t0 = time.time()
            policy_params, rollouts = jit_opt(mjx_data, policy_params)
            plan_ms = (time.time() - plan_t0) * 1000

        if SHOW_TRACES and rollouts is not None:
            ii = 0
            for k in range(num_trace_sites):
                for i in range(num_traces):
                    for j in range(ctrl.ctrl_steps):
                        mujoco.mjv_connector(
                            viewer.user_scn.geoms[ii],
                            mujoco.mjtGeom.mjGEOM_LINE,
                            TRACE_WIDTH,
                            rollouts.trace_sites[i, j, k],
                            rollouts.trace_sites[i, j + 1, k],
                        )
                        ii += 1

        if rollouts is not None:
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

        changed = task.update_phase(mj_data)
        if changed:
            policy_params = ctrl.init_params(
                initial_knots=task.initial_knots(ctrl.num_knots)
            )

        pos_err, ori_err = task.pose_error_from_data(mj_data)
        finger = task.finger_width_from_data(mj_data)
        contacts = int(task.has_grasp_contacts(mj_data))
        success = int(task.goal_reached(mj_data))

        loop_ms = (time.time() - t0) * 1000
        elapsed = time.time() - t0
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

        rtr = step_dt / (time.time() - t0)
        print(
            f"[{task.phase.name:>9s}] "
            f"loop={loop_ms:5.1f}ms  "
            f"plan={plan_ms:5.1f}ms  "
            f"budget={step_dt*1000:.0f}ms  "
            f"rtr={rtr:.2f}  "
            f"pos={100*pos_err:4.1f}cm  "
            f"ori={ori_err:5.3f}  "
            f"finger={100*finger:4.1f}cm  "
            f"contacts={contacts}  "
            f"success={success}",
            end="\r",
        )

print("\nDone.")
