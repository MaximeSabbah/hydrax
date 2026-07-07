import argparse
import json
import subprocess
import time
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import jax
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.algs import FeedbackMPPI
from hydrax.configs import load_pregrasp_config
from hydrax.tasks.panda_pregrasp import PandaPregrasp

"""
Run the Panda pregrasp reach task.

The control loop is the deployment-faithful multi-rate one: the planner
replans at 25 Hz, the plant integrates at 1 kHz (the LFC rate), and the
applied torque follows the LFC law

    tau = tau_ff + Kp (q_des - q) + Kd (v_des - v)

zero-order-held planner outputs between updates. In feedforward mode Kp/Kd
are the fixed joint-impedance gains from the task options servoing the plan
reference (LFC with constant gains — the low-level loop, not the planner,
stabilizes the open-loop-unstable arm, exactly as on the real robot). In
feedback mode the F-MPPI law fully replaces that impedance (user decision
2026-07-06):

    tau = tau_ff + K (x - x0),   K = du*/dx0

anchored at x0, the state the plan was solved from — NOT the reference:
the planner already responded to the tracking error when it solved from
x0 (it is baked into tau_ff), so K multiplies only the drift accumulated
since planning, continuously approximating a re-solve between the 25 Hz
planner updates. Reference tracking is entirely tau_ff's job.

Scenario flags implement the V-A4 protocol: --disturb (5 N·m, 50 ms
torque pulse on joint 2 mid-reach; run both modes, the feedback report
gates itself against the feedforward baseline), --mass_scale (plant
link-mass mismatch), --latency (planner outputs activate a whole number
of planner periods late). Feedback runs also evaluate the V-A3 gain
health gates on the K series actually applied.

Runs are headless: the loop checks the V-A1/V-A4 gates from
doc/feedback_mppi_panda_port_plan.md, writes a report JSON, and records the
trajectory next to it. Visualization is a replay: --replay opens the viewer
and plays the recorded run, with the minimum-jerk reference shown as a
transparent ghost robot (--show_reference).
"""

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
    "--disturb",
    action="store_true",
    help="V-A4(b): 5 N·m, 50 ms torque pulse on joint 2 mid-reach.",
)
parser.add_argument(
    "--mass_scale",
    type=float,
    default=None,
    help="V-A4(c): scale the plant link masses (e.g. 0.9 or 1.1).",
)
parser.add_argument(
    "--latency",
    type=float,
    default=0.0,
    help="V-A4(d): delay planner outputs by this many seconds "
    "(a multiple of the 0.04 s planner period).",
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
# The scenario suffix keeps every check's artifacts separate (fixed name
# per check+mode+scenario, overwritten each run).
scenario_tags = []
if args.disturb:
    scenario_tags.append("disturb")
if args.mass_scale is not None:
    scenario_tags.append(f"mass{args.mass_scale:g}")
if args.latency > 0.0:
    scenario_tags.append(f"latency{args.latency:g}")
scenario = "_".join(scenario_tags)

# Define the task (cost and dynamics). The OCP tuning comes from the single
# tuning surface, hydrax/configs/pregrasp.yaml — the same file the ROS
# planner adapter loads, so this gate certifies what deploys.
options, config = load_pregrasp_config()
task = PandaPregrasp(options=options)
tau_max = np.asarray(task.options.tau_max)
vel_max = np.asarray(task.options.vel_max)
goal_pos = np.asarray(task.options.goal_pos)
kp_fixed = np.asarray(task.options.kp_fixed)
kd_fixed = np.asarray(task.options.kd_fixed)

# Pair the controller with the task (the same glue exists in the ROS
# planner adapter; V-B1 asserts the two stay equivalent), warm-started
# from the feedforward torque plan.
ctrl = FeedbackMPPI(
    task,
    num_samples=config.num_samples,
    noise_std=config.noise_scale * tau_max,
    temperature=config.temperature,
    mean_adaptation_rate=config.mean_adaptation_rate,
    num_gain_samples=config.num_gain_samples,
    compute_gains=(args.mode == "feedback"),
    plan_horizon=config.plan_horizon,
    spline_type=config.spline_type,
    num_knots=config.num_knots,
    iterations=config.iterations,
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
if args.mass_scale is not None:
    # V-A4(c): the PLANT is mismatched; the planner's model is untouched
    mj_model.body_mass *= args.mass_scale

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

# V-A4(d): planner outputs activate a whole number of planner periods
# late (a one-slot-per-cycle queue — the bridge's transport delay model)
delay_cycles = int(round(args.latency / control_period))
if abs(delay_cycles * control_period - args.latency) > 1e-9:
    raise SystemExit("--latency must be a multiple of the 0.04 s period")

# V-A4(b): torque pulse on joint 2 (index 1), mid-reach
pulse_start, pulse_len, pulse_joint, pulse_torque = (
    0.5 * task.duration,
    0.05,
    1,
    5.0,
)

print(
    f"mode={args.mode}  scenario={scenario or 'nominal'}  "
    f"duration={task.duration:.1f}s + hold {args.hold}s"
)
# Planner outputs currently applied by the 1 kHz law. Before the first
# activation (only reachable with --latency): hold the start pose under
# the impedance, like the real stack waiting for its first control
# message.
active = {
    "tau_ff": np.zeros(mj_model.nu),
    "q_des": plan_q[0],
    "v_des": plan_v[0],
    "K": None,
}
pending: deque = deque()
solve_ms: list[float] = []
q_hist = np.zeros((n_steps + 1, mj_model.nq))
t_hist = np.arange(n_steps + 1) * mj_model.opt.timestep
q_hist[0] = mj_data.qpos
q_dev_hist = np.zeros(n_steps)
q_err_sq_sum = np.zeros(mj_model.nq)
tau_frac_max = np.zeros(mj_model.nu)
vel_frac_max = np.zeros(mj_model.nv)
K_series: list[np.ndarray] = []
ess_series: list[float] = []
nominal_weight_series: list[float] = []
nan_seen = False

for k in range(n_steps):
    t = k * mj_model.opt.timestep
    i_ref = min(int(round(t / control_period)), plan_q.shape[0] - 1)

    if k % steps_per_cycle == 0:
        cycle = k // steps_per_cycle
        state = mjx_data.replace(
            qpos=mj_data.qpos.copy(), qvel=mj_data.qvel.copy(), time=t
        )
        t0 = time.perf_counter()
        params, rollouts = jit_optimize(state, params)
        jax.block_until_ready(params.mean)
        solve_ms.append(1000.0 * (time.perf_counter() - t0))
        out = {
            "tau_ff": np.asarray(
                ctrl.get_action(params, t), dtype=np.float64
            ),
            "q_des": plan_q[i_ref],
            "v_des": plan_v[i_ref],
            "K": None,
        }
        nan_seen |= not np.all(np.isfinite(out["tau_ff"]))
        if args.mode == "feedback":
            # The F-MPPI gains, anchored at the state this solve used
            out["K"] = np.asarray(params.gains, dtype=np.float64)
            out["q0"] = mj_data.qpos.copy()
            out["v0"] = mj_data.qvel.copy()
            nan_seen |= not np.all(np.isfinite(out["K"]))
            K_series.append(out["K"])
            ess_series.append(float(params.gain_ess))
            nominal_weight_series.append(float(params.gain_nominal_weight))
        pending.append((cycle + delay_cycles, out))
        while pending and pending[0][0] <= cycle:
            active = pending.popleft()[1]
        if k % (steps_per_cycle * 25) == 0:
            ee_err = np.linalg.norm(mj_data.site_xpos[site_id] - goal_pos)
            gain_info = (
                f"  |K|={np.linalg.norm(out['K']):.1f}"
                f"  ess={ess_series[-1]:.0f}"
                if args.mode == "feedback"
                else ""
            )
            print(
                f"  t={t:5.1f}s  ee_err={ee_err:.4f} m  "
                f"solve={solve_ms[-1]:.1f} ms{gain_info}",
                flush=True,
            )

    # The 1 kHz LFC law, zero-order-held planner outputs. The command is
    # NOT clipped here: the torque margin below must see the true demand
    # (a demand beyond the limits would trip the real robot's reflexes,
    # not silently saturate). The plant actuators clamp to ctrlrange,
    # like the hardware.
    if active["K"] is not None:
        # Feedback mode: the F-MPPI law tau_ff + K (x - x0). K fully
        # replaces the fixed impedance; reference tracking is tau_ff's job.
        dx = np.concatenate(
            [mj_data.qpos - active["q0"], mj_data.qvel - active["v0"]]
        )
        tau_cmd = active["tau_ff"] + active["K"] @ dx
    else:
        # Feedforward mode: fixed joint impedance servoing the reference
        tau_cmd = (
            active["tau_ff"]
            + kp_fixed * (active["q_des"] - mj_data.qpos)
            + kd_fixed * (active["v_des"] - mj_data.qvel)
        )
    if args.disturb:
        in_pulse = pulse_start <= t < pulse_start + pulse_len
        mj_data.qfrc_applied[pulse_joint] = pulse_torque if in_pulse else 0.0
    mj_data.ctrl[:] = tau_cmd
    mujoco.mj_step(mj_model, mj_data)

    q_hist[k + 1] = mj_data.qpos
    q_dev_hist[k] = np.linalg.norm(mj_data.qpos - plan_q[i_ref])
    q_err_sq_sum += np.square(mj_data.qpos - plan_q[i_ref])
    tau_frac_max = np.maximum(tau_frac_max, np.abs(tau_cmd) / tau_max)
    vel_frac_max = np.maximum(vel_frac_max, np.abs(mj_data.qvel) / vel_max)

# Metrics and gates
mujoco.mj_forward(mj_model, mj_data)
terminal_ee_err = float(np.linalg.norm(mj_data.site_xpos[site_id] - goal_pos))
rms_q_err = np.sqrt(q_err_sq_sum / n_steps)
# Hold-window wander (the jitter at the reached pose): joint span and
# velocity over the final 1 s. The sweep target for mean_adaptation_rate.
hold = q_hist[-1000:]
hold_q_span = float((hold.max(axis=0) - hold.min(axis=0)).max())
hold_qvel_rms = float(np.sqrt(np.mean(np.square(np.diff(hold, axis=0) / mj_model.opt.timestep))))
steady = np.asarray(solve_ms[1:])
check = "V-A1" if args.mode == "feedforward" else "V-A4"
run_name = f"{check}_{args.mode}" + (f"_{scenario}" if scenario else "")
extras = {}

gates = {
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
if not scenario:
    # Terminal accuracy is gated on nominal runs only ((b)-(d) gate
    # stability and margins; their tracking numbers are informational)
    gates["terminal_ee_error"] = (terminal_ee_err <= 0.010, terminal_ee_err)

if args.mode == "feedback" and not scenario:
    # V-A4(a): the feedback reach must track at least as well as the
    # certified feedforward run (compared against its report)
    ff = json.loads((REPORT_DIR / "V-A1_feedforward.json").read_text())
    ff_terminal = ff["gates"]["terminal_ee_error"]["value"]
    ff_rms = float(np.mean(ff["rms_q_err_per_joint"]))
    gates["tracking_vs_feedforward"] = (
        terminal_ee_err <= ff_terminal
        and float(np.mean(rms_q_err)) <= ff_rms,
        {
            "terminal_ee": [terminal_ee_err, ff_terminal],
            "mean_rms_q": [float(np.mean(rms_q_err)), ff_rms],
        },
    )

if args.disturb:
    # Post-pulse metrics; in feedback mode, gate against the identical
    # feedforward run: V-A4(b) requires >= 20 % better peak and settling
    k_pulse = int(round(pulse_start / mj_model.opt.timestep))
    s0 = int(round((pulse_start + 1.5) / mj_model.opt.timestep))
    s1 = int(round((pulse_start + 2.5) / mj_model.opt.timestep))
    peak_deviation = float(q_dev_hist[k_pulse:].max())
    settle_rms = float(np.sqrt(np.mean(q_dev_hist[s0:s1] ** 2)))
    extras["disturbance"] = {
        "peak_deviation_rad": peak_deviation,
        "settle_rms_rad": settle_rms,
    }
    if args.mode == "feedback":
        base = json.loads(
            (REPORT_DIR / "V-A1_feedforward_disturb.json").read_text()
        )["disturbance"]
        gates["disturbance_rejection"] = (
            peak_deviation <= 0.8 * base["peak_deviation_rad"]
            and settle_rms <= 0.8 * base["settle_rms_rad"],
            {
                "peak": [peak_deviation, base["peak_deviation_rad"]],
                "settle": [settle_rms, base["settle_rms_rad"]],
            },
        )

if args.mode == "feedback":
    # V-A3: gain health on the K series actually applied in this run
    K = np.asarray(K_series)
    kf = np.linalg.norm(K, axis=(1, 2))
    dk = np.linalg.norm(np.diff(K, axis=0), axis=(1, 2))
    med = np.array(  # trailing 1 s (25-cycle) running median of |K|_F
        [np.median(kf[max(0, i - 24) : i + 1]) for i in range(1, len(kf))]
    )
    ess = np.asarray(ess_series)
    nom_w = np.asarray(nominal_weight_series)
    pos_diag_max = float(np.abs(K[:, np.arange(7), np.arange(7)]).max())
    vel_diag_max = float(np.abs(K[:, np.arange(7), 7 + np.arange(7)]).max())
    smooth_fraction = float((dk <= 0.3 * med).mean())
    flap_ratio = float((dk / np.maximum(med, 1e-9)).max())
    nom_w_fraction = float((nom_w > 0.5).mean())
    gates["health_k_pos_diag"] = (pos_diag_max <= 600.0, pos_diag_max)
    gates["health_k_vel_diag"] = (vel_diag_max <= 60.0, vel_diag_max)
    gates["health_k_smoothness"] = (smooth_fraction >= 0.99, smooth_fraction)
    gates["health_k_no_flap"] = (flap_ratio < 10.0, flap_ratio)
    gates["health_ess"] = (
        float(ess.min()) >= 0.1 * ctrl.num_gain_samples,
        float(ess.min()),
    )
    gates["health_nominal_weight"] = (nom_w_fraction < 0.20, nom_w_fraction)
    extras["gain_health"] = {
        "k_frobenius_median": float(np.median(kf)),
        "k_frobenius_max": float(kf.max()),
        "ess_median": float(np.median(ess)),
    }

ok = all(passed for passed, _ in gates.values())

print(f"\n-- {check} gates ({scenario or 'nominal'}) --")
for name, (passed, value) in gates.items():
    print(f"  {name:26s} {'PASS' if passed else 'FAIL'}  ({value})")
print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
print(f"rms q err per joint [rad]: {np.round(rms_q_err, 4)}")
print(
    f"hold wander: q_span={hold_q_span:.4f} rad  "
    f"qvel_rms={hold_qvel_rms:.4f} rad/s"
)

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
    "scenario": scenario or "nominal",
    "config": {
        "num_samples": ctrl.num_samples,
        "noise_std": np.asarray(ctrl.noise_std).tolist(),
        "temperature": ctrl.temperature,
        "num_gain_samples": ctrl.num_gain_samples,
        "plan_horizon": ctrl.plan_horizon,
        "num_knots": ctrl.num_knots,
        "spline_type": ctrl.spline_type,
        "iterations": ctrl.iterations,
        "duration": task.duration,
    },
    "gates": {k: {"pass": p, "value": v} for k, (p, v) in gates.items()},
    "rms_q_err_per_joint": rms_q_err.tolist(),
    "terminal_ee_error": terminal_ee_err,
    "hold_q_span_rad": hold_q_span,
    "hold_qvel_rms": hold_qvel_rms,
    "verdict": "PASS" if ok else "FAIL",
    **extras,
}
report_path = REPORT_DIR / f"{run_name}.json"
report_path.write_text(json.dumps(report, indent=2))

history_line = {
    "date": report["date"],
    "git_sha": report["git_sha"],
    "check": check,
    "mode": args.mode,
    "scenario": scenario or "nominal",
    "verdict": report["verdict"],
    "terminal_ee_m": terminal_ee_err,
    "tau_frac": float(tau_frac_max.max()),
    "vel_frac": float(vel_frac_max.max()),
    "solve_p95_ms": gates["solve_p95_ms"][1],
}
with open(REPORT_DIR / "history.jsonl", "a") as f:
    f.write(json.dumps(history_line) + "\n")

feedback_arrays = (
    {
        "gains": np.asarray(K_series),
        "ess": np.asarray(ess_series),
        "nominal_weight": np.asarray(nominal_weight_series),
        "q_dev": q_dev_hist,
    }
    if args.mode == "feedback"
    else {}
)
traj_path = REPORT_DIR / f"{run_name}_traj.npz"
np.savez(
    traj_path,
    qpos=q_hist,
    time=t_hist,
    plan_qpos=plan_q,
    plan_fps=task.reference_fps,
    **feedback_arrays,
)
print(f"report: {report_path}")
print(f"replay: uv run python examples/panda_pregrasp.py --replay {traj_path}")
