import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from hydrax.algs import MPPI
from hydrax.tasks.pick_and_place import Phase, PickAndPlace

"""
Pick-and-place with the Franka Emika Panda using DIAL-MPC.

State machine: PREGRASP -> DESCEND -> GRASP -> LIFT -> TRANSPORT
               -> PLACE -> OPEN -> RETREAT -> DONE

All phase controllers are pre-JIT-compiled at startup so phase
transitions are instantaneous at runtime.

Double-click the red target and [ctrl + left click] to move the goal.
"""

# ── task & controller ─────────────────────────────────────────────────
task = PickAndPlace()

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

# Neutral / folded config matching the keyframe (EE points straight down)
home_ctrl = jnp.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04])
initial_knots = jnp.tile(home_ctrl, (ctrl.num_knots, 1))

mjx_data = task.make_data()
mjx_data = mjx_data.replace(
    qpos=mj_data.qpos,
    qvel=mj_data.qvel,
    mocap_pos=mj_data.mocap_pos,
    mocap_quat=mj_data.mocap_quat,
)

# ── trace rendering settings ─────────────────────────────────────────
# NOTE: trace rendering is pure-Python (300 mjv_connector calls) and
# adds ~100-140ms per loop.  Disable for performance, enable to debug.
SHOW_TRACES = False
MAX_TRACES = 3
TRACE_WIDTH = 5.0
TRACE_COLOR = [0.2, 0.8, 0.2, 0.15]


# ── pre-JIT every phase at startup ───────────────────────────────────
def jit_for_phase(phase, controller, warmup_data, warmup_knots):
    """JIT-compile controller.optimize for one specific phase.

    Each phase is wrapped in a unique closure so JAX traces independently
    (same underlying method → same cache key → only first phase compiled).
    """
    controller.task.phase = phase

    # Closures create unique function identities, forcing fresh JIT traces
    def _optimize(state, params):
        return controller.optimize(state, params)

    def _interp(tq, tk, knots):
        return controller.interp_func(tq, tk, knots)

    jit_opt = jax.jit(_optimize)
    jit_interp = jax.jit(_interp)

    # init + two warm-up calls to fully compile XLA
    params = controller.init_params(initial_knots=warmup_knots)
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

print("Pre-JIT-compiling all phases ...")
all_st = time.time()
phase_jit = {}
for phase in Phase:
    st = time.time()
    phase_jit[phase] = jit_for_phase(phase, ctrl, mjx_data, initial_knots)
    print(f"  {phase.name:>12s}  {time.time() - st:.1f}s")
print(f"Total pre-JIT: {time.time() - all_st:.1f}s\n")

# restore to initial phase
task.phase = Phase.PREGRASP

# init policy params for the first phase
policy_params = ctrl.init_params(initial_knots=initial_knots)

# ── main loop ─────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

    # allocate trace geometry in the viewer scene
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

        # sync planner state from simulation
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )

        # look up pre-compiled functions for the current phase
        jit_opt, jit_interp = phase_jit[task.phase]

        # plan
        plan_t0 = time.time()
        policy_params, rollouts = jit_opt(mjx_data, policy_params)
        plan_ms = (time.time() - plan_t0) * 1000

        # draw rollout traces
        if SHOW_TRACES:
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

        # interpolate controls for the sim sub-steps
        tq = (
            jnp.arange(sim_steps_per_replan) * mj_model.opt.timestep
            + mj_data.time
        )
        us = np.asarray(
            jit_interp(tq, policy_params.tk, policy_params.mean[None, ...])
        )[0]

        # step simulation
        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = us[i]
            mujoco.mj_step(mj_model, mj_data)
        viewer.sync()

        # advance phase machine (on real state, outside JAX)
        changed = task.update_phase(mj_data)
        if changed:
            policy_params = ctrl.init_params(
                initial_knots=policy_params.mean
            )

        # timing
        loop_ms = (time.time() - t0) * 1000
        elapsed = time.time() - t0
        if elapsed < step_dt:
            time.sleep(step_dt - elapsed)

        rtr = step_dt / (time.time() - t0)
        print(
            f"[{task.phase.name:>12s}] "
            f"loop={loop_ms:5.1f}ms  "
            f"plan={plan_ms:5.1f}ms  "
            f"budget={step_dt*1000:.0f}ms  "
            f"rtr={rtr:.2f}  "
            f"hold={task._hold_count}",
            end="\r",
        )

print("\nDone.")
