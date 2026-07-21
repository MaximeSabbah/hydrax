import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import jax
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.algs import FeedbackMPPI
from hydrax.configs import load_pick_place_config
from hydrax.tasks.panda_pick_place import (
    PandaPickPlace,
    Phase,
    PickPlacePhaseMachine,
)

"""
Run the Panda cube pick-and-place (P-A2, doc/pick_place_plan.md).

The control loop is the deployment-faithful multi-rate one from the
pregrasp example: the planner replans at 25 Hz, the plant integrates at
1 kHz, and the applied arm torque follows the LFC law (fixed impedance in
feedforward mode, tau_ff + K (x - x0) in feedback mode). The differences
from the pregrasp run:

- the PLANT is models/panda/scene.xml — the full robot with articulated
  fingers, contacts, the free-joint cube and the placement target; the
  planner keeps the 7-DoF contact-free arm model. The plant does the
  grasping; the planner never knows the cube exists.
- the solver's clock is the PLAN CLOCK of the PickPlacePhaseMachine: it
  advances normally inside a phase segment and crosses a segment boundary
  only once the arm has converged there, so a slow phase stalls the
  reference instead of being abandoned (the hard timeout below is the
  stall gate).
- the gripper is a discrete device, not part of the MPPI action space:
  ctrl[7] follows the phase machine's command (position servo,
  0.04 = open, 0.0 = closed), like the gripper action on the robot.

Headless by default: checks the P-A2 gates, writes a report JSON and a
recording; --replay plays it back in the viewer.
"""

REPORT_DIR = Path(__file__).parent.parent / "validation" / "reports"

parser = argparse.ArgumentParser(description="Panda cube pick-and-place.")
parser.add_argument(
    "--mode",
    choices=["feedforward", "feedback"],
    default="feedback",
    help="1 kHz law: fixed impedance on the reference (feedforward) or "
    "the F-MPPI gains (feedback, the deployed exact_feedback mode).",
)
parser.add_argument(
    "--hold",
    type=float,
    default=1.5,
    help="Extra settle time after the sequence completes (s).",
)
parser.add_argument(
    "--q0_noise",
    type=float,
    default=None,
    help="P-A3: offset the plant's initial arm configuration by a uniform "
    "per-joint draw in [-q0_noise, +q0_noise] rad (hand-placed robot; the "
    "reference timeline still starts at the nominal start_q).",
)
parser.add_argument(
    "--q0_seed", type=int, default=0, help="P-A3: seed of the q0 draw."
)
parser.add_argument(
    "--cube_noise",
    type=float,
    default=None,
    help="P-A3: offset the TRUE cube position by a uniform draw in "
    "[-cube_noise, +cube_noise] m in x and y, while the task keeps the "
    "nominal configured pose — the hand-measurement-error axis; the blind "
    "grasp must still succeed.",
)
parser.add_argument(
    "--cube_seed", type=int, default=0, help="P-A3: seed of the cube draw."
)
parser.add_argument(
    "--mass_scale",
    type=float,
    default=None,
    help="P-A3: scale the plant body masses (e.g. 0.9 or 1.1); the "
    "planner's model is untouched.",
)
parser.add_argument(
    "--cube_friction",
    type=float,
    default=None,
    help="P-A3: scale the cube's sliding friction (e.g. 0.5) — the slip "
    "axis; the planner never knows the cube exists.",
)
parser.add_argument(
    "--replay",
    nargs="?",
    const="latest",
    default=None,
    metavar="TRAJ_NPZ",
    help="Replay a recorded run in the viewer instead of running.",
)
args = parser.parse_args()

scenario_tags = []
if args.q0_noise is not None:
    scenario_tags.append(f"q0noise{args.q0_noise:g}_seed{args.q0_seed}")
if args.cube_noise is not None:
    scenario_tags.append(f"cube{args.cube_noise:g}_seed{args.cube_seed}")
if args.mass_scale is not None:
    scenario_tags.append(f"mass{args.mass_scale:g}")
if args.cube_friction is not None:
    scenario_tags.append(f"fric{args.cube_friction:g}")
scenario = "_".join(scenario_tags)


if args.replay is not None:
    # ------------------------------------------------------------------
    # Replay: kinematic playback of a recorded run (full plant state).
    # ------------------------------------------------------------------
    import mujoco.viewer

    if args.replay == "latest":
        recordings = list(REPORT_DIR.glob("P-A2_*_traj.npz"))
        if not recordings:
            raise SystemExit(f"no P-A2 recordings found in {REPORT_DIR}")
        traj_path = max(recordings, key=lambda p: p.stat().st_mtime)
    else:
        traj_path = Path(args.replay)
    rec = np.load(traj_path)
    rec_q, rec_t = rec["qpos"], rec["time"]
    print(f"replaying {traj_path} ({rec_t[-1]:.1f} s, {len(rec_t)} frames)")

    mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/panda/scene.xml")
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = rec_q[0]
    mujoco.mj_forward(mj_model, mj_data)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        frame_step = max(1, int(0.01 / (rec_t[1] - rec_t[0])))
        while viewer.is_running():
            for k in range(0, len(rec_t), frame_step):
                start = time.time()
                mj_data.qpos[:] = rec_q[k]
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                if not viewer.is_running():
                    break
                elapsed = time.time() - start
                if elapsed < 0.01:
                    time.sleep(0.01 - elapsed)
    raise SystemExit(0)


# ----------------------------------------------------------------------
# Headless multi-rate run (P-A2).
# ----------------------------------------------------------------------
options, config = load_pick_place_config()
task = PandaPickPlace(options=options)
tau_max = np.asarray(task.options.tau_max)
vel_max = np.asarray(task.options.vel_max)
kp_fixed = np.asarray(task.options.kp_fixed)
kd_fixed = np.asarray(task.options.kd_fixed)

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

# The PLANT: full scene at 1 kHz (articulated fingers, contacts, cube)
mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/panda/scene.xml")
if args.mass_scale is not None:
    mj_model.body_mass *= args.mass_scale
if args.cube_friction is not None:
    mj_model.geom("object_geom").friction[0] *= args.cube_friction
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
CUBE_Q = slice(9, 12)  # cube free-joint position in the plant qpos
if args.q0_noise is not None:
    # only the PLANT starts off the plan (hand-placed robot); the
    # reference timeline still begins at start_q
    offset = np.random.default_rng(args.q0_seed).uniform(
        -args.q0_noise, args.q0_noise, 7
    )
    mj_data.qpos[:7] = np.clip(
        np.asarray(task.start_q) + offset,
        mj_model.jnt_range[:7, 0] + 0.05,
        mj_model.jnt_range[:7, 1] - 0.05,
    )
if args.cube_noise is not None:
    # the TRUE cube is off the hand-measured config pose (x, y)
    mj_data.qpos[9:11] += np.random.default_rng(args.cube_seed).uniform(
        -args.cube_noise, args.cube_noise, 2
    )

plan_q = np.asarray(task.reference_qpos)
plan_v = np.asarray(task.reference_qvel)
target_pos = np.asarray(task.options.target_pos)

# Initialize the controller and JIT-warm it from the initial state
params = ctrl.init_params(initial_knots=initial_knots)
jit_optimize = jax.jit(ctrl.optimize)
mjx_data = mjx.make_data(task.model)
print("Jitting the controller...")
st = time.time()
state = mjx_data.replace(
    qpos=mj_data.qpos[:7].copy(), qvel=mj_data.qvel[:7].copy(), time=0.0
)
params, _ = jit_optimize(state, params)
jax.block_until_ready(params.mean)
print(f"Time to jit: {time.time() - st:.3f} seconds")

control_period = task.dt
steps_per_cycle = int(round(control_period / mj_model.opt.timestep))
# Stall protection: the clock gating may stretch the sequence — under a
# 10 % mass mismatch (P-A3) the certified-safe convergence waits add up
# to ~8 s of legitimate stalls — but a healthy run must still terminate.
# Exceeding the budget fails the no_stall gate.
time_budget = 2.0 * task.duration + args.hold
max_steps = int(round(time_budget / mj_model.opt.timestep))

pm = PickPlacePhaseMachine(task)
active = {
    "tau_ff": np.zeros(7),
    "q_des": plan_q[0],
    "v_des": plan_v[0],
    "K": None,
}
grip_prev = False
grip_events: dict[str, dict] = {}
solve_ms: list[float] = []
q_hist, t_hist, phase_hist = [mj_data.qpos.copy()], [0.0], [pm.phase.name]
q_err_sq_sum = np.zeros(7)
tau_frac_max = np.zeros(7)
vel_frac_max = np.zeros(7)
K_series: list[np.ndarray] = []
ess_series: list[float] = []
nominal_weight_series: list[float] = []
nan_seen = False
phase_entries: dict[str, dict] = {}
cube_lift_max = float(mj_data.qpos[CUBE_Q][2])
done_at: float | None = None

print(
    f"mode={args.mode}  scenario={scenario or 'nominal'}  "
    f"timeline={task.duration:.1f}s "
    f"(budget {time_budget:.1f}s) + hold {args.hold}s"
)

k = 0
while k < max_steps:
    t = k * mj_model.opt.timestep

    if k % steps_per_cycle == 0:
        q_arm, v_arm = mj_data.qpos[:7].copy(), mj_data.qvel[:7].copy()
        prev_phase = pm.phase
        pm.update(q_arm, v_arm)
        if pm.phase != prev_phase:
            # record the plant state at each phase entry (the precision
            # gates read the CLOSE/OPEN entries)
            phase_entries[pm.phase.name] = {
                "t": t,
                "ee": mj_data.site_xpos[site_id].copy(),
                "cube": mj_data.qpos[CUBE_Q].copy(),
            }
            print(
                f"  t={t:6.2f}s  -> {pm.phase.name:9s} "
                f"(plan_time {pm.plan_time:5.2f}s)",
                flush=True,
            )
            if pm.phase == Phase.DONE:
                done_at = t
        if pm.gripper_closed != grip_prev:
            # the gripper-command instants ARE the grasp/place precision
            grip_events["close" if pm.gripper_closed else "open"] = {
                "t": t,
                "ee": mj_data.site_xpos[site_id].copy(),
                "cube": mj_data.qpos[CUBE_Q].copy(),
            }
            grip_prev = pm.gripper_closed
        i_ref = min(int(pm.plan_time * task.reference_fps), plan_q.shape[0] - 1)

        state = mjx_data.replace(qpos=q_arm, qvel=v_arm, time=pm.plan_time)
        t0 = time.perf_counter()
        params, rollouts = jit_optimize(state, params)
        jax.block_until_ready(params.mean)
        solve_ms.append(1000.0 * (time.perf_counter() - t0))
        active = {
            "tau_ff": np.asarray(
                ctrl.get_action(params, pm.plan_time), dtype=np.float64
            ),
            "q_des": plan_q[i_ref],
            "v_des": plan_v[i_ref],
            "K": None,
        }
        nan_seen |= not np.all(np.isfinite(active["tau_ff"]))
        if args.mode == "feedback":
            active["K"] = np.asarray(params.gains, dtype=np.float64)
            active["q0"], active["v0"] = q_arm, v_arm
            nan_seen |= not np.all(np.isfinite(active["K"]))
            K_series.append(active["K"])
            ess_series.append(float(params.gain_ess))
            nominal_weight_series.append(float(params.gain_nominal_weight))
        # the gripper is a discrete device: position servo on ctrl[7]
        mj_data.ctrl[7] = 0.0 if pm.gripper_closed else 0.04

    # The 1 kHz LFC law on the arm, zero-order-held planner outputs.
    # Unclipped demand, as in the pregrasp example: the torque margin
    # must see what the real robot's reflexes would see.
    if active["K"] is not None:
        dx = np.concatenate(
            [mj_data.qpos[:7] - active["q0"], mj_data.qvel[:7] - active["v0"]]
        )
        tau_cmd = active["tau_ff"] + active["K"] @ dx
    else:
        tau_cmd = (
            active["tau_ff"]
            + kp_fixed * (active["q_des"] - mj_data.qpos[:7])
            + kd_fixed * (active["v_des"] - mj_data.qvel[:7])
        )
    mj_data.ctrl[:7] = tau_cmd
    mujoco.mj_step(mj_model, mj_data)
    k += 1

    q_hist.append(mj_data.qpos.copy())
    t_hist.append(k * mj_model.opt.timestep)
    phase_hist.append(pm.phase.name)
    i_ref = min(int(pm.plan_time * task.reference_fps), plan_q.shape[0] - 1)
    q_err_sq_sum += np.square(mj_data.qpos[:7] - plan_q[i_ref])
    tau_frac_max = np.maximum(tau_frac_max, np.abs(tau_cmd) / tau_max)
    vel_frac_max = np.maximum(vel_frac_max, np.abs(mj_data.qvel[:7]) / vel_max)
    if pm.phase in (Phase.LIFT, Phase.TRANSPORT):
        cube_lift_max = max(cube_lift_max, float(mj_data.qpos[CUBE_Q][2]))
    if done_at is not None and t - done_at >= args.hold:
        break

# ----------------------------------------------------------------------
# Metrics and P-A2 gates
# ----------------------------------------------------------------------
mujoco.mj_forward(mj_model, mj_data)
q_hist = np.asarray(q_hist)
cube_final = mj_data.qpos[CUBE_Q].copy()
cube_start_z = float(mj_model.key("home").qpos[CUBE_Q][2])
rms_q_err = np.sqrt(q_err_sq_sum / k)
steady = np.asarray(solve_ms[1:])

close_cmd = grip_events.get("close")
open_entry = phase_entries.get(Phase.OPEN.name)

# Grasp precision at the close-command instant: EE vs the COMMANDED
# grasp pose (nominal cube + grasp offset). Under --cube_noise the actual
# cube is elsewhere by construction — the arm's job is to hit the
# commanded pose; whether the offset cube still gets captured is what
# cube_lifted / cube_placed certify. The actual cube is in the report.
grasp_err = None
if close_cmd is not None:
    d = close_cmd["ee"] - (
        np.asarray(task.options.cube_pos)
        + np.asarray(task.options.grasp_offset)
    )
    grasp_err = {
        "lateral_m": float(np.linalg.norm(d[:2])),
        "vertical_m": float(abs(d[2])),
    }

gates = {
    "no_stall_sequence_done": (done_at is not None, done_at),
    "grasp_entry_lateral": (
        grasp_err is not None and grasp_err["lateral_m"] <= 0.010,
        grasp_err,
    ),
    "grasp_entry_vertical": (
        grasp_err is not None and grasp_err["vertical_m"] <= 0.005,
        grasp_err,
    ),
    "cube_lifted": (
        cube_lift_max >= cube_start_z + 0.08,
        cube_lift_max,
    ),
    "cube_placed": (
        # a blind grasp cannot correct the cube-pose measurement error
        # along the finger axis, so the placed cube inherits it: the
        # tolerance widens by the injected offset under --cube_noise
        bool(
            np.linalg.norm(cube_final[:2] - target_pos[:2])
            <= 0.010 + (args.cube_noise or 0.0)
            and abs(cube_final[2] - cube_start_z) <= 0.005
        ),
        cube_final.tolist(),
    ),
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
    # the no-recompile gate: a phase transition must never trigger a
    # re-trace (that would show as a multi-second solve mid-run)
    "solve_max_ms": (float(steady.max()) <= 100.0, float(steady.max())),
}

extras: dict = {
    "phase_entries": {
        name: {"t": e["t"], "ee": e["ee"].tolist(), "cube": e["cube"].tolist()}
        for name, e in phase_entries.items()
    },
    "gripper_commands": {
        name: {"t": e["t"], "ee": e["ee"].tolist(), "cube": e["cube"].tolist()}
        for name, e in grip_events.items()
    },
}
if open_entry is not None:
    d = open_entry["ee"] - (target_pos + np.asarray(task.options.place_offset))
    extras["place_entry_error_m"] = {
        "lateral": float(np.linalg.norm(d[:2])),
        "vertical": float(abs(d[2])),
    }

if args.mode == "feedback":
    # Gain health on the applied K series (V-A3 bounds; smoothness is the
    # known-moot sample-noise metric — reported, not gated, as in V-A5)
    K = np.asarray(K_series)
    kf = np.linalg.norm(K, axis=(1, 2))
    dk = np.linalg.norm(np.diff(K, axis=0), axis=(1, 2))
    med = np.array(
        [np.median(kf[max(0, i - 24) : i + 1]) for i in range(1, len(kf))]
    )
    ess = np.asarray(ess_series)
    nom_w = np.asarray(nominal_weight_series)
    pos_diag_max = float(np.abs(K[:, np.arange(7), np.arange(7)]).max())
    vel_diag_max = float(np.abs(K[:, np.arange(7), 7 + np.arange(7)]).max())
    flap_ratio = float((dk / np.maximum(med, 1e-9)).max())
    gates["health_k_pos_diag"] = (pos_diag_max <= 600.0, pos_diag_max)
    gates["health_k_vel_diag"] = (vel_diag_max <= 60.0, vel_diag_max)
    gates["health_k_no_flap"] = (flap_ratio < 10.0, flap_ratio)
    gates["health_ess"] = (
        float(ess.min()) >= 0.1 * ctrl.num_gain_samples,
        float(ess.min()),
    )
    gates["health_nominal_weight"] = (
        float((nom_w > 0.5).mean()) < 0.20,
        float((nom_w > 0.5).mean()),
    )
    extras["gain_health"] = {
        "k_frobenius_median": float(np.median(kf)),
        "smoothness_fraction": float((dk <= 0.3 * med).mean()),
        "ess_median": float(np.median(ess)),
    }

ok = all(passed for passed, _ in gates.values())

print(f"\n-- P-A2 gates ({args.mode}) --")
for name, (passed, value) in gates.items():
    print(f"  {name:26s} {'PASS' if passed else 'FAIL'}  ({value})")
print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
print(f"rms q err per joint [rad]: {np.round(rms_q_err, 4)}")
print(
    f"cube: lifted to {cube_lift_max:.3f} m, final "
    f"{np.round(cube_final, 4)} (target {target_pos.tolist()})"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
run_name = f"P-A2_{args.mode}" + (f"_{scenario}" if scenario else "")
report = {
    "check": "P-A2",
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
        "temperature": ctrl.temperature,
        "plan_horizon": ctrl.plan_horizon,
        "spline_type": ctrl.spline_type,
        "iterations": ctrl.iterations,
        "timeline_duration": task.duration,
    },
    "gates": {k: {"pass": p, "value": v} for k, (p, v) in gates.items()},
    "rms_q_err_per_joint": rms_q_err.tolist(),
    "verdict": "PASS" if ok else "FAIL",
    **extras,
}
report_path = REPORT_DIR / f"{run_name}.json"
report_path.write_text(json.dumps(report, indent=2))

with open(REPORT_DIR / "history.jsonl", "a") as f:
    f.write(
        json.dumps(
            {
                "date": report["date"],
                "git_sha": report["git_sha"],
                "check": "P-A2",
                "mode": args.mode,
                "scenario": scenario or "nominal",
                "verdict": report["verdict"],
                "cube_final": cube_final.tolist(),
                "tau_frac": float(tau_frac_max.max()),
                "solve_p95_ms": gates["solve_p95_ms"][1],
            }
        )
        + "\n"
    )

traj_path = REPORT_DIR / f"{run_name}_traj.npz"
np.savez(
    traj_path,
    qpos=q_hist,
    time=np.asarray(t_hist),
    phase=np.asarray(phase_hist),
    plan_qpos=plan_q,
    plan_fps=task.reference_fps,
)
print(f"report: {report_path}")
print(f"replay: uv run python examples/panda_pick_place.py --replay {traj_path}")
