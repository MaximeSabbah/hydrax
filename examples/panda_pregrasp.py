import argparse
import json
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import jax
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.algs import FeedbackMPPI
from hydrax.tasks.panda_pregrasp import (
    GOAL_POS,
    TAU_MAX,
    VEL_MAX,
    PandaPregrasp,
)

"""
Run the Panda pregrasp reach task.

The control loop is the deployment-faithful multi-rate one: the planner
replans at 25 Hz, the plant integrates at 1 kHz (the LFC rate), and the
applied torque follows the LFC law

    tau = tau_ff + Kp (q_des - q) + Kd (v_des - v)

zero-order-held planner outputs between updates. In feedforward mode Kp/Kd
are the fixed joint-impedance gains below (LFC with constant gains — the
low-level loop, not the planner, stabilizes the open-loop-unstable arm,
exactly as on the real robot). In feedback mode the F-MPPI gains take this
role (plan Phase 2/3).

Runs are headless: the loop checks the V-A1/V-A4 gates from
doc/feedback_mppi_panda_port_plan.md, writes a report JSON, and records the
trajectory next to it. Visualization is a replay: --replay opens the viewer
and plays the recorded run, with the minimum-jerk reference shown as a
transparent ghost robot (--show_reference).
"""

# Fixed joint-impedance gains of the feedforward-mode low level.
# NOTE: sync with the LFC configuration used on the robot before Tier B.
KP_FIXED = np.array([1000.0, 1000.0, 1000.0, 1000.0, 20.0, 10.0, 5.0])
KD_FIXED = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 1.0])

REPORT_DIR = Path(__file__).parent.parent / "validation" / "reports"

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Panda pregrasp reach: headless multi-rate run by default, "
    "--replay to visualize a recorded run."
)
parser.add_argument(
    "--mode",
    choices=["feedforward", "feedback"],
    default="feedforward",
    help="Apply only tau_ff (feedforward) or also the F-MPPI gains.",
)
parser.add_argument(
    "--hold",
    type=float,
    default=1.5,
    help="Extra settle time at the goal after the plan ends (s).",
)
parser.add_argument(
    "--replay",
    nargs="?",
    const="latest",
    default=None,
    metavar="TRAJ_NPZ",
    help="Replay a recorded run in the viewer instead of running the "
    "controller. With no value, replays the latest recording.",
)
parser.add_argument(
    "--show_reference",
    action="store_true",
    help="Replay: show the reference trajectory as a 'ghost' robot.",
)
args = parser.parse_args()


if args.replay is not None:
    # ------------------------------------------------------------------
    # Replay mode: kinematic playback of a recorded run, robot vs ghost.
    # ------------------------------------------------------------------
    import mujoco.viewer

    if args.replay == "latest":
        recordings = list(REPORT_DIR.glob("*_traj.npz"))
        if not recordings:
            raise SystemExit(f"no recordings found in {REPORT_DIR}")
        traj_path = max(recordings, key=lambda p: p.stat().st_mtime)
    else:
        traj_path = Path(args.replay)
    rec = np.load(traj_path)
    rec_q, rec_t = rec["qpos"], rec["time"]
    plan_q, plan_fps = rec["plan_qpos"], float(rec["plan_fps"])
    print(f"replaying {traj_path} ({rec_t[-1]:.1f} s, {len(rec_t)} frames)")

    # The visual scene wrapper (floor, lights, object) — same nq as the
    # planning model, so recorded joint trajectories replay directly.
    mj_model = mujoco.MjModel.from_xml_path(
        ROOT + "/models/panda/pregrasp_scene.xml"
    )
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = rec_q[0]
    mujoco.mj_forward(mj_model, mj_data)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if args.show_reference:
            ref_data = mujoco.MjData(mj_model)
            ref_data.qpos[:] = plan_q[0]
            mujoco.mj_forward(mj_model, ref_data)
            vopt = mujoco.MjvOption()
            vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
            pert = mujoco.MjvPerturb()
            catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        frame_step = max(1, int(0.01 / (rec_t[1] - rec_t[0])))  # ~100 fps
        while viewer.is_running():
            for k in range(0, len(rec_t), frame_step):
                start = time.time()
                mj_data.qpos[:] = rec_q[k]
                mujoco.mj_forward(mj_model, mj_data)
                if args.show_reference:
                    i_ref = min(
                        int(rec_t[k] * plan_fps), plan_q.shape[0] - 1
                    )
                    ref_data.qpos[:] = plan_q[i_ref]
                    mujoco.mj_forward(mj_model, ref_data)
                    mujoco.mjv_updateScene(
                        mj_model,
                        ref_data,
                        vopt,
                        pert,
                        viewer.cam,
                        catmask,
                        viewer.user_scn,
                    )
                viewer.sync()
                if not viewer.is_running():
                    break
                elapsed = time.time() - start
                if elapsed < 0.01:
                    time.sleep(0.01 - elapsed)
    raise SystemExit(0)


# ----------------------------------------------------------------------
# Headless multi-rate run (V-A1 feedforward / V-A4 feedback).
# ----------------------------------------------------------------------
if args.mode == "feedback":
    raise NotImplementedError(
        "feedback mode arrives with FeedbackMPPI (plan Phase 2/3)"
    )

# Define the task (cost and dynamics)
task = PandaPregrasp()

# Set up the controller, warm-started from the feedforward torque plan.
# Tuned in the 2026-07-03 Phase 1 sweeps, with the fixed-impedance low
# level in the loop: per-joint noise (0.03 * tau_max) lets the wrists
# refine at ~0.4 Nm while the shoulders explore at ~2.6 Nm — a scalar
# noise cannot do both (measured 10.3 mm vs 4.0 mm terminal error).
ctrl = FeedbackMPPI(
    task,
    num_samples=1024,
    noise_std=0.03 * TAU_MAX,
    temperature=0.01,
    plan_horizon=0.4,
    spline_type="cubic",
    num_knots=4,
)
knot_times = np.linspace(0.0, ctrl.plan_horizon, ctrl.num_knots)
knot_idx = np.minimum(
    np.round(knot_times / task.dt).astype(int),
    task.reference_ctrl.shape[0] - 1,
)
initial_knots = task.reference_ctrl[knot_idx]

# Define the model used for simulation: same robot, 1 kHz (the LFC rate)
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.001

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = task.start_q

# Reference plan on the numpy side (x_des for the LFC law + metrics)
plan_q = np.asarray(task.reference_qpos)
plan_v = np.asarray(task.reference_qvel)

# Initialize the controller and JIT-warm it from the initial state
params = ctrl.init_params(initial_knots=initial_knots)
jit_optimize = jax.jit(ctrl.optimize)
mjx_data = mjx.make_data(task.model)
print("Jitting the controller...")
st = time.time()
state = mjx_data.replace(
    qpos=mj_data.qpos.copy(), qvel=mj_data.qvel.copy(), time=0.0
)
params, _ = jit_optimize(state, params)
jax.block_until_ready(params.mean)
print(f"Time to jit: {time.time() - st:.3f} seconds")

control_period = task.dt
steps_per_cycle = int(round(control_period / mj_model.opt.timestep))
n_steps = int(round((task.duration + args.hold) / mj_model.opt.timestep))
site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

print(f"mode={args.mode}  duration={task.duration:.1f}s + hold {args.hold}s")
tau_ff = np.zeros(mj_model.nu)
q_des, v_des = plan_q[0], plan_v[0]
solve_ms: list[float] = []
q_hist = np.zeros((n_steps + 1, mj_model.nq))
t_hist = np.arange(n_steps + 1) * mj_model.opt.timestep
q_hist[0] = mj_data.qpos
q_err_sq_sum = np.zeros(mj_model.nq)
tau_frac_max = np.zeros(mj_model.nu)
vel_frac_max = np.zeros(mj_model.nv)
nan_seen = False

for k in range(n_steps):
    t = k * mj_model.opt.timestep
    i_ref = min(int(round(t / control_period)), plan_q.shape[0] - 1)

    if k % steps_per_cycle == 0:
        state = mjx_data.replace(
            qpos=mj_data.qpos.copy(), qvel=mj_data.qvel.copy(), time=t
        )
        t0 = time.perf_counter()
        params, rollouts = jit_optimize(state, params)
        jax.block_until_ready(params.mean)
        solve_ms.append(1000.0 * (time.perf_counter() - t0))
        tau_ff = np.asarray(ctrl.get_action(params, t), dtype=np.float64)
        q_des, v_des = plan_q[i_ref], plan_v[i_ref]
        nan_seen |= not np.all(np.isfinite(tau_ff))
        if k % (steps_per_cycle * 25) == 0:
            ee_err = np.linalg.norm(mj_data.site_xpos[site_id] - GOAL_POS)
            print(
                f"  t={t:5.1f}s  ee_err={ee_err:.4f} m  "
                f"solve={solve_ms[-1]:.1f} ms",
                flush=True,
            )

    # The 1 kHz LFC law, zero-order-held planner outputs. The command is
    # NOT clipped here: the torque margin below must see the true demand
    # (a demand beyond the limits would trip the real robot's reflexes,
    # not silently saturate). The plant actuators clamp to ctrlrange,
    # like the hardware.
    tau_cmd = (
        tau_ff
        + KP_FIXED * (q_des - mj_data.qpos)
        + KD_FIXED * (v_des - mj_data.qvel)
    )
    mj_data.ctrl[:] = tau_cmd
    mujoco.mj_step(mj_model, mj_data)

    q_hist[k + 1] = mj_data.qpos
    q_err_sq_sum += np.square(mj_data.qpos - plan_q[i_ref])
    tau_frac_max = np.maximum(tau_frac_max, np.abs(tau_cmd) / TAU_MAX)
    vel_frac_max = np.maximum(vel_frac_max, np.abs(mj_data.qvel) / VEL_MAX)

# Metrics and gates
mujoco.mj_forward(mj_model, mj_data)
terminal_ee_err = float(np.linalg.norm(mj_data.site_xpos[site_id] - GOAL_POS))
rms_q_err = np.sqrt(q_err_sq_sum / n_steps)
steady = np.asarray(solve_ms[1:])
gates = {
    "terminal_ee_error": (terminal_ee_err <= 0.010, terminal_ee_err),
    "torque_margin": (
        bool(np.all(tau_frac_max <= 0.90)),
        float(tau_frac_max.max()),
    ),
    "velocity_margin": (
        bool(np.all(vel_frac_max <= 0.80)),
        float(vel_frac_max.max()),
    ),
    "no_nan": (not nan_seen, nan_seen),
    "solve_mean_ms": (float(steady.mean()) <= 30.0, float(steady.mean())),
    "solve_p95_ms": (
        float(np.percentile(steady, 95)) <= 40.0,
        float(np.percentile(steady, 95)),
    ),
}
ok = all(passed for passed, _ in gates.values())

check = "V-A1" if args.mode == "feedforward" else "V-A4"
print(f"\n-- {check} gates --")
for name, (passed, value) in gates.items():
    print(f"  {name:20s} {'PASS' if passed else 'FAIL'}  ({value})")
print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
print(f"rms q err per joint [rad]: {np.round(rms_q_err, 4)}")

# Report JSON + trajectory recording for --replay. Fixed filenames: each
# run overwrites the previous artifacts for this check+mode (nothing
# accumulates); the longitudinal record is one compact line per run
# appended to history.jsonl. Milestone numbers worth keeping are quoted in
# doc/feedback_mppi_panda_port_plan.md.
REPORT_DIR.mkdir(parents=True, exist_ok=True)
report = {
    "check": check,
    "date": datetime.now().isoformat(timespec="seconds"),
    "git_sha": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip(),
    "mode": args.mode,
    "config": {
        "num_samples": ctrl.num_samples,
        "noise_std": np.asarray(ctrl.noise_std).tolist(),
        "temperature": ctrl.temperature,
        "plan_horizon": ctrl.plan_horizon,
        "num_knots": ctrl.num_knots,
        "spline_type": ctrl.spline_type,
        "iterations": ctrl.iterations,
        "duration": task.duration,
    },
    "gates": {k: {"pass": p, "value": v} for k, (p, v) in gates.items()},
    "rms_q_err_per_joint": rms_q_err.tolist(),
    "verdict": "PASS" if ok else "FAIL",
}
report_path = REPORT_DIR / f"{check}_{args.mode}.json"
report_path.write_text(json.dumps(report, indent=2))

history_line = {
    "date": report["date"],
    "git_sha": report["git_sha"],
    "check": check,
    "mode": args.mode,
    "verdict": report["verdict"],
    "terminal_ee_m": terminal_ee_err,
    "tau_frac": float(tau_frac_max.max()),
    "vel_frac": float(vel_frac_max.max()),
    "solve_p95_ms": gates["solve_p95_ms"][1],
}
with open(REPORT_DIR / "history.jsonl", "a") as f:
    f.write(json.dumps(history_line) + "\n")

traj_path = REPORT_DIR / f"{check}_{args.mode}_traj.npz"
np.savez(
    traj_path,
    qpos=q_hist,
    time=t_hist,
    plan_qpos=plan_q,
    plan_fps=task.reference_fps,
)
print(f"report: {report_path}")
print(f"replay: uv run python examples/panda_pregrasp.py --replay {traj_path}")
