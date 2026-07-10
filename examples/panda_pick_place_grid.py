import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

"""
P-A3 — pick-and-place robustness grid (doc/pick_place_plan.md).

Drives examples/panda_pick_place.py over the three deployment axes, one
process per run (the persistent JAX cache keeps the per-process JIT
cheap):

- initial arm configuration (hand-placed robot, the V-A5 axis);
- TRUE cube position vs the hand-measured config value — the
  deployment-critical axis of the fixed-poses decision;
- plant mass scale and cube sliding friction (the slip axis).

Runs are judged baseline-relative (gates that already fail on the
nominal P-A2 report carry no robustness signal; at the frozen config
the nominal report is all-green, so every gate counts). Grasp success
(cube lifted AND placed) is required on every run. Per-run artifacts are
deleted on PASS and kept on FAIL for --replay; the aggregate grid lands
in validation/reports/P-A3_grid_<mode>.json.
"""

REPORT_DIR = Path(__file__).parent.parent / "validation" / "reports"
EXAMPLE = Path(__file__).parent / "panda_pick_place.py"

parser = argparse.ArgumentParser(description="P-A3 robustness grid.")
parser.add_argument(
    "--mode", choices=["feedforward", "feedback"], default="feedback"
)
parser.add_argument(
    "--q0", type=float, nargs="+", default=[0.1, 0.2],
    help="start-pose offset magnitudes (rad)",
)
parser.add_argument(
    "--cube", type=float, nargs="+", default=[0.005, 0.01],
    help="cube-pose offset magnitudes (m, x/y)",
)
parser.add_argument(
    "--seeds", type=int, default=5, help="draws per magnitude"
)
args = parser.parse_args()

# the grid: (scenario suffix, extra CLI flags) per run
runs_spec: list[tuple[str, list[str]]] = []
for mag in args.q0:
    for seed in range(args.seeds):
        runs_spec.append(
            (
                f"q0noise{mag:g}_seed{seed}",
                ["--q0_noise", str(mag), "--q0_seed", str(seed)],
            )
        )
for mag in args.cube:
    for seed in range(args.seeds):
        runs_spec.append(
            (
                f"cube{mag:g}_seed{seed}",
                ["--cube_noise", str(mag), "--cube_seed", str(seed)],
            )
        )
for scale in (0.9, 1.1):
    runs_spec.append((f"mass{scale:g}", ["--mass_scale", str(scale)]))
runs_spec.append(("fric0.5", ["--cube_friction", "0.5"]))

nominal_path = REPORT_DIR / f"P-A2_{args.mode}.json"
if not nominal_path.exists():
    raise SystemExit(
        f"missing nominal baseline {nominal_path}; run "
        f"examples/panda_pick_place.py --mode {args.mode} first"
    )
known_moot = {
    k
    for k, g in json.loads(nominal_path.read_text())["gates"].items()
    if not g["pass"]
}

print(f"P-A3 {args.mode}: {len(runs_spec)} runs")
if known_moot:
    print(f"ignoring known-moot nominal failures: {sorted(known_moot)}")

rows = []
for suffix, flags in runs_spec:
    run_name = f"P-A2_{args.mode}_{suffix}"
    report_path = REPORT_DIR / f"{run_name}.json"
    traj_path = REPORT_DIR / f"{run_name}_traj.npz"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(EXAMPLE), "--mode", args.mode, *flags],
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0 or not report_path.exists():
        print(f"  {suffix:22s} CRASH ({elapsed:.0f}s)")
        print("\n".join(proc.stdout.splitlines()[-10:]))
        rows.append({"scenario": suffix, "verdict": "CRASH"})
        continue
    report = json.loads(report_path.read_text())
    gates = report["gates"]
    failed = [
        k for k, g in gates.items() if not g["pass"] and k not in known_moot
    ]
    verdict = "PASS" if not failed else "FAIL"
    grasp = gates["grasp_entry_lateral"]["value"] or {}
    row = {
        "scenario": suffix,
        "verdict": verdict,
        "grasp_lateral_m": grasp.get("lateral_m"),
        "cube_final": gates["cube_placed"]["value"],
        "tau_frac": gates["torque_margin"]["value"],
        "solve_max_ms": gates["solve_max_ms"]["value"],
        "failed_gates": failed,
    }
    rows.append(row)
    print(
        f"  {suffix:22s} {verdict}  "
        f"grasp_lat={1000 * (row['grasp_lateral_m'] or -1):.1f}mm  "
        f"tau={row['tau_frac']:.2f}  ({elapsed:.0f}s)"
        + (f"  FAILED: {failed}" if failed else ""),
        flush=True,
    )
    if verdict == "PASS":
        report_path.unlink()
        traj_path.unlink(missing_ok=True)
    else:
        print(f"    kept for replay: {traj_path}")

completed = [r for r in rows if r["verdict"] != "CRASH"]
ok = bool(completed) and all(r["verdict"] == "PASS" for r in rows)
summary = {
    "check": "P-A3",
    "date": datetime.now().isoformat(timespec="seconds"),
    "git_sha": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip(),
    "mode": args.mode,
    "grid": {"q0": args.q0, "cube": args.cube, "seeds": args.seeds},
    "known_moot_gates": sorted(known_moot),
    "runs": rows,
    "verdict": "PASS" if ok else "FAIL",
}
summary_path = REPORT_DIR / f"P-A3_grid_{args.mode}.json"
summary_path.write_text(json.dumps(summary, indent=2))

with open(REPORT_DIR / "history.jsonl", "a") as f:
    f.write(
        json.dumps(
            {
                "date": summary["date"],
                "git_sha": summary["git_sha"],
                "check": "P-A3",
                "mode": args.mode,
                "scenario": f"grid q0={args.q0} cube={args.cube} x{args.seeds}",
                "verdict": summary["verdict"],
            }
        )
        + "\n"
    )

print(f"\nVERDICT: {summary['verdict']}")
print(f"report: {summary_path}")
if not ok:
    raise SystemExit(1)
