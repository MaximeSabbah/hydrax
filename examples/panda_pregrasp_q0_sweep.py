import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

"""
V-A5 — initial-configuration robustness sweep.

On the real robot the arm is hand-placed near (not at) the nominal home
pose before each session, but the min-jerk reference plan always starts
at the hardcoded start_q: nothing re-seeds the plan from the measured
state. Each run of this sweep therefore starts the PLANT from a perturbed
configuration while the reference stays nominal, and checks the
controller closes the gap, lands, and settles.

It drives examples/panda_pregrasp.py --q0_noise MAG --q0_seed K over a
magnitude x seed grid, one process per run (the persistent JAX
compilation cache in hydrax/__init__.py keeps the per-process JIT cheap
after the first run). Every run evaluates the full per-run gate set of
the example — margins, solve timing, gain health in feedback mode — plus
the V-A5 landing gates (terminal <= 15 mm, hold qvel RMS <= 0.05).

Per-run artifacts are deleted on PASS and kept on FAIL, so a failing run
can be replayed (--replay) and its report inspected. The aggregate grid
is written to validation/reports/V-A5_q0_robustness_<mode>.json.
"""

REPORT_DIR = Path(__file__).parent.parent / "validation" / "reports"
EXAMPLE = Path(__file__).parent / "panda_pregrasp.py"

parser = argparse.ArgumentParser(
    description="V-A5: initial-configuration robustness sweep."
)
parser.add_argument(
    "--mode",
    choices=["feedforward", "feedback"],
    default="feedback",
    help="Controller mode to sweep (feedback = the deployed exact_feedback "
    "law; feedforward quantifies the impedance mode's initial yank).",
)
parser.add_argument(
    "--noise",
    type=float,
    nargs="+",
    default=[0.05, 0.1, 0.2],
    help="Per-joint offset magnitudes to sweep (rad).",
)
parser.add_argument(
    "--seeds",
    type=int,
    default=8,
    help="Offset draws per magnitude.",
)
args = parser.parse_args()

check = "V-A1" if args.mode == "feedforward" else "V-A4"

# The V-A5 question is relative: does a perturbed start break anything the
# certified NOMINAL run does not already flag? Gates that fail on the
# nominal report (at the frozen config that is health_k_smoothness — the K
# estimate is sample-noisy step to step, documented known-moot in the plan
# doc) fail identically here and carry no robustness signal, so runs are
# judged on the remaining gates.
nominal_path = REPORT_DIR / f"{check}_{args.mode}.json"
if not nominal_path.exists():
    raise SystemExit(
        f"missing nominal baseline {nominal_path}; run "
        f"examples/panda_pregrasp.py --mode {args.mode} first"
    )
known_moot = {
    k
    for k, g in json.loads(nominal_path.read_text())["gates"].items()
    if not g["pass"]
}

runs = []
print(
    f"V-A5 {args.mode}: {len(args.noise)} magnitudes x {args.seeds} seeds "
    f"= {len(args.noise) * args.seeds} runs"
)
if known_moot:
    print(f"ignoring known-moot nominal failures: {sorted(known_moot)}")
for mag in args.noise:
    for seed in range(args.seeds):
        run_name = f"{check}_{args.mode}_q0noise{mag:g}_seed{seed}"
        report_path = REPORT_DIR / f"{run_name}.json"
        traj_path = REPORT_DIR / f"{run_name}_traj.npz"
        t0 = time.time()
        proc = subprocess.run(
            [
                sys.executable,
                str(EXAMPLE),
                "--mode",
                args.mode,
                "--q0_noise",
                str(mag),
                "--q0_seed",
                str(seed),
            ],
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0 or not report_path.exists():
            print(f"  q0={mag:g} seed={seed}  CRASH ({elapsed:.0f}s)")
            print("\n".join(proc.stdout.splitlines()[-15:]))
            print(proc.stderr.strip().splitlines()[-5:] if proc.stderr else "")
            runs.append(
                {"noise_rad": mag, "seed": seed, "verdict": "CRASH"}
            )
            continue
        report = json.loads(report_path.read_text())
        gates = report["gates"]
        failed = [
            k
            for k, g in gates.items()
            if not g["pass"] and k not in known_moot
        ]
        verdict = "PASS" if not failed else "FAIL"
        row = {
            "noise_rad": mag,
            "seed": seed,
            "verdict": verdict,
            "offset_norm_rad": report["q0"]["offset_norm_rad"],
            "terminal_ee_m": report["terminal_ee_error"],
            "tau_frac": gates["torque_margin"]["value"],
            "vel_frac": gates["velocity_margin"]["value"],
            "hold_qvel_rms": report["hold_qvel_rms"],
            "failed_gates": failed,
        }
        if "health_ess" in gates:
            row["ess_min"] = gates["health_ess"]["value"]
        runs.append(row)
        ess_info = (
            f"  ess_min={row['ess_min']:.0f}" if "ess_min" in row else ""
        )
        print(
            f"  q0={mag:g} seed={seed}  {verdict}  "
            f"terminal={1000 * row['terminal_ee_m']:.1f}mm  "
            f"tau={row['tau_frac']:.2f}  vel={row['vel_frac']:.2f}"
            f"{ess_info}  ({elapsed:.0f}s)"
            + (f"  FAILED: {failed}" if failed else ""),
            flush=True,
        )
        if verdict == "PASS":
            report_path.unlink()
            traj_path.unlink(missing_ok=True)
        else:
            print(f"    kept for replay: {traj_path}")

completed = [r for r in runs if r["verdict"] != "CRASH"]
ok = bool(completed) and all(r["verdict"] == "PASS" for r in runs)
summary = {
    "check": "V-A5",
    "date": datetime.now().isoformat(timespec="seconds"),
    "git_sha": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip(),
    "mode": args.mode,
    "grid": {"noise_rad": args.noise, "seeds_per_noise": args.seeds},
    "known_moot_gates": sorted(known_moot),
    "runs": runs,
    "worst": (
        {
            "terminal_ee_m": max(r["terminal_ee_m"] for r in completed),
            "tau_frac": max(r["tau_frac"] for r in completed),
            "vel_frac": max(r["vel_frac"] for r in completed),
            "hold_qvel_rms": max(r["hold_qvel_rms"] for r in completed),
            **(
                {"ess_min": min(r["ess_min"] for r in completed)}
                if completed and "ess_min" in completed[0]
                else {}
            ),
        }
        if completed
        else {}
    ),
    "verdict": "PASS" if ok else "FAIL",
}
summary_path = REPORT_DIR / f"V-A5_q0_robustness_{args.mode}.json"
summary_path.write_text(json.dumps(summary, indent=2))

with open(REPORT_DIR / "history.jsonl", "a") as f:
    f.write(
        json.dumps(
            {
                "date": summary["date"],
                "git_sha": summary["git_sha"],
                "check": "V-A5",
                "mode": args.mode,
                "scenario": f"q0 grid {args.noise} x {args.seeds} seeds",
                "verdict": summary["verdict"],
                **summary["worst"],
            }
        )
        + "\n"
    )

print(f"\n-- V-A5 ({args.mode}) --")
if summary["worst"]:
    for name, value in summary["worst"].items():
        print(f"  worst {name:16s} {value:.4f}")
print(f"VERDICT: {summary['verdict']}")
print(f"report: {summary_path}")
if not ok:
    raise SystemExit(1)
