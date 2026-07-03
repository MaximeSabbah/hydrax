# Feedback-MPPI Panda port plan (sbmpc → hydrax)

Status: **in progress** — written 2026-07-03. Phase 0 mostly done (branch, model, env).
Scope: implement the Feedback-MPPI (Riccati-gain) controller and the Panda
pregrasp reach problem in this hydrax fork, then swap it in as the planner
backend behind the existing `sbmpc_ros` stack.

## Decisions log

| Date | Decision |
|------|----------|
| 2026-07-03 | **sbmpc results are not a reference.** No baseline fixtures, no parity comparisons against sbmpc runs. All validation gates are absolute and self-contained in hydrax. (sbmpc code is still the source for the ported gain math and for config starting points — its *outputs* are not a target.) |
| 2026-07-03 | Port lives **in-tree in this hydrax fork** (`hydrax/algs/`, `hydrax/tasks/`, `examples/`), not a separate package. Upstream (`vincekurtz/hydrax`) stays a configured remote; rebase cost accepted. |
| 2026-07-03 | **No bridge-side gain conditioning.** EMA filtering / norm clamping / `feedback_gain_scale` in the ROS bridge are rejected. K must come out of the optimizer deployable; gain quality is a solver-level requirement (Phase 3). |
| 2026-07-03 | **Two-tier validation.** Tier A validates the controller entirely inside this repo (one example script + pytest, no ROS). Tier B validates the machinery through the `sbmpc_ros` launch with LFC. A phase may not enter Tier B until its Tier A checks pass. |
| 2026-07-03 | **Single example script.** One `examples/panda_pregrasp.py` with `--mode feedforward\|feedback`; its simulation loop is always multi-rate (1 kHz plant / 25 Hz planner, planner outputs zero-order-held between updates) to mirror LFC exactly. No separate multirate script. |
| 2026-07-03 | hydrax runs in its own **uv** environment (per hydrax README), not the sbmpc pixi env. How the Phase 4 bridge imports hydrax is decided at Phase 4. |
| 2026-07-02 | `sbmpc_ros` stays **frozen** as the test harness until the hydrax planner is validated in sim (Phase 4 adds only a backend switch). The thin bridge rebuild (Phase 6) happens only after, per the consolidation plan in `sbmpc_ros/doc/`. |

## Goal

Reach a pregrasp pose with the Panda by tracking a minimum-jerk joint plan
(q/v/τ references over a moving horizon) generated at task setup from the
start state to the pregrasp IK solution — with Feedback-MPPI gains, deployed
on the real Franka through the existing `sbmpc_ros` bridge
(`tau = tau_ff + K(x_des − x_meas)` at 1 kHz, planner at 25 Hz).

**Definition of done:** exact-feedback pregrasp reach on the real robot with no
velocity/torque reflex trips, tracking error ≤ the feedforward baseline run,
and a 25 Hz planner cycle (solve + gains) ≤ 40 ms on the deployment GPU.

## Non-goals

- Pick-and-place phases (pregrasp reach only, for now).
- Reproducing sbmpc's behavior or comparing against its outputs.
- Rewriting `sbmpc_ros` (deferred to Phase 6 / consolidation plan).
- Bridge-side conditioning of K (rejected — see decisions log).
- Domain randomization, risk strategies, other hydrax algs.

## What carries over from the sbmpc work (knowledge, not outputs)

- The F-MPPI gain formulation (softmax-weighted einsum over per-sample cost
  gradients × first-step control deltas) is FD-proven correct math
  (2026-07-01); it is ported as-is and re-proven here by V-A2.
- The known instability drivers are problem-design-level: large cost weights
  (gradients ~1e6) and a near-degenerate softmax (tiny sampling std → the
  nominal sample wins → K flaps 0↔2465). Phase 3 addresses them at the source;
  V-A3's ESS metric is the direct readout of the degeneracy.
- Reverse-mode AD does not work through MJX solver loops → gains use
  forward-mode `jax.jvp`.
- Timing: at 25 Hz the full cycle must fit 40 ms; on this GPU sbmpc measured
  h=10/1024 samples/128 gain samples at p95 ≈ 34 ms, so the budget is tight
  but feasible.
- Feedback cannot be tested in a loop where the plant advances once per
  planner step (x_des == x_meas at the application instant ⇒ K·0 = 0) — this
  is why the example's loop is always multi-rate.

## Architecture: files to add (all in this fork)

```
hydrax/models/panda/            # panda.xml + scene.xml + meshes (done, from sbmpc model)
hydrax/tasks/panda_pregrasp.py  # PandaPregraspTask(Task): min-jerk plan + IK generated
                                # at construction; time-indexed tracking costs
hydrax/algs/feedback_mppi.py    # FeedbackMPPI(MPPI) + gain computation
examples/panda_pregrasp.py      # THE reach script: --mode feedforward|feedback,
                                # multi-rate loop (1 kHz plant / 25 Hz planner, ZOH,
                                # LFC law; K≡0 in feedforward), scenario flags
                                # --disturb --mass-scale --latency,
                                # writes a report JSON per run
tests/test_feedback_gains.py    # V-A2 FD gain checks
validation/reports/             # per-run report JSONs (gitignored except a kept summary)
```

`PandaPregraspTask` generates its own references at construction: a damped
least-squares IK for the pregrasp pose, then a quintic (minimum-jerk) joint
plan from the start state to the IK solution, with `tau_ff` from MuJoCo
inverse dynamics along the plan. The plan arrays live on the task and are
indexed by `x.time` inside `running_cost(x, u)` / `terminal_cost(x)`
(`humanoid_mocap` is the in-repo precedent for time-indexed references). Cost
terms: position, velocity, control regularization — the same three terms as
sbmpc's pregrasp config, whose weights serve only as the starting point for
Phase 3 tuning.

`examples/panda_pregrasp.py` simulates deployment faithfully in both modes:
plant integrated at 1 kHz, planner called at 25 Hz, and between planner
updates the applied torque is `tau_ff + K(x_des − x_now)` with K ≡ 0 in
feedforward mode. Feedforward and feedback runs share one loop and are
directly comparable; the mode flag is the only difference.

`FeedbackMPPI` extends `MPPI`:
1. `sample_knots`: force sample 0 to zero noise (the warm-started nominal)
   and use **per-joint** noise std (scale × torque limit
   [87,87,87,87,12,12,12]; hydrax's scalar `noise_level` cannot express this).
2. After `update_params`: select nominal + (num_gain_samples − 1) lowest-cost
   rollouts; compute per-sample dJ/dx₀ by `jax.jvp` of the rollout cost w.r.t.
   (qpos, qvel) via `data.replace(...)` — 14 basis tangents × gain batch;
   then the softmax-weighted gain combination ported from the F-MPPI paper
   implementation.
3. Control delta for the gain formula is taken at the **application time**
   `t₀ = state.time` by evaluating each sample's spline there — hydrax
   warm-starts by shifting knot *times*, not by index-rolling.

## Starting parameters (from sbmpc's pregrasp config; Phase 3 retunes freely)

| parameter | value |
|-----------|-------|
| plan horizon | 0.4 s (task `dt=0.04` × 10 steps) |
| num_samples | 1024 |
| temperature | 0.01 |
| noise std | 0.01 × per-joint τ_max |
| num_knots | 4, `spline_type="cubic"` |
| gain batch | 128 |
| reach duration | 7.5 s, `max_velocity_fraction` 0.20 |

---

## Validation protocol

Two tiers. **Tier A** runs entirely in this repo (the example script + pytest,
plant = same MJX model) and validates the *controller* on absolute criteria.
**Tier B** runs through the `sbmpc_ros` launch and validates the *machinery*:
adapter contract, bridge timing, LFC application of (tau_ff, K), then the
robot. The A→B boundary is the attribution line: V-B2 runs the same scenario
as V-A4(a), so any behavioral delta between them is ROS/transport/LFC, not
the controller.

Every check writes a report JSON to `validation/reports/` (metrics +
pass/fail + git SHA + config hash) so gates are reproducible and comparable
across runs.

### Tier A — in-hydrax (controller)

**V-A1 — feedforward tracking** (`panda_pregrasp.py --mode feedforward`,
Phase 1 gate). Full 7.5 s reach on the task-generated plan. Pass, all of:
- terminal EE position error ≤ 10 mm;
- max |τ_i|/τ_max,i ≤ 0.90 and max |q̇_i|/q̇_max,i ≤ 0.80 for every joint,
  evaluated at 1 kHz (reflex margins — joint 2 velocity is the historical
  tripper);
- no NaN/Inf in costs or controls;
- solve time (post-JIT-warmup) mean ≤ 30 ms, p95 ≤ 40 ms on the deployment
  GPU.
Per-joint RMS tracking error is recorded in the report (informational — it
becomes the reference the feedback mode must beat in V-A4).

**V-A2 — gain correctness** (`tests/test_feedback_gains.py`, Phase 2 gate).
Self-contained FD protocol: fix the RNG key, set gain batch = all samples so
the analytic gain is exact, finite-difference du₀/dx₀ element-wise (ε = 1e-4
on q, 1e-3 on q̇), compare relative Frobenius error.
- (a) toy task (hydrax `particle` or double integrator): ≤ 1 %;
- (b) Panda, fresh-seeded at the start state: ≤ 2 %;
- (c) Panda, warm-started mid-reach: ≤ 10 %.

**V-A3 — gain health along the reach** (logged by `panda_pregrasp.py`,
Phase 3 gate, evaluated after retuning). Time series over the full reach:
- ‖K_t‖ bounded: position-block diagonal entries ≤ 600 Nm/rad,
  velocity-block ≤ 60 Nm·s/rad (the range of stock Franka joint-impedance
  stiffness; to be ratified against LFC experience);
- smoothness: ‖K_t − K_{t−1}‖_F ≤ 0.3 × running-median ‖K‖_F for ≥ 99 % of
  steps, and never an order-of-magnitude step-to-step flap;
- softmax health: effective sample size ESS = 1/Σwᵢ² of the gain-batch
  weights ≥ 10 % of the gain batch at every step, and the nominal sample
  carries weight > 0.5 in < 20 % of steps (direct measurements of the
  degeneracy failure mode).

**V-A4 — closed-loop feedback** (`panda_pregrasp.py --mode feedback`,
Phase 3 gate — same script and loop as V-A1, only the mode flag differs):
- (a) nominal: stable full reach; tracking error ≤ the V-A1 feedforward run;
  τ and q̇ margins as in V-A1;
- (b) disturbance rejection (`--disturb`): 5 N·m × 50 ms torque pulse on
  joint 2 mid-reach — peak deviation and settling error ≥ 20 % better than
  the identical run in feedforward mode (feedback must *help*, not merely
  not hurt);
- (c) model mismatch (`--mass-scale 0.9|1.1`): plant link masses ±10 % —
  stable, margins hold;
- (d) latency (`--latency 0.04`): one full planner period of delay on the
  planner output — stable, margins hold (the historical real-robot failure
  mode);
- (e) V-A3 health bounds recomputed on the K actually applied in (a)–(d) —
  they must hold in closed loop, not just along a nominal run.

### Tier B — through sbmpc_ros (machinery)

**V-B1 — adapter contract, no ROS runtime** (pytest in `sbmpc_ros`, pattern of
`test_planner_smoke` / `test_fake_ros_loop`). `HydraxPlannerAdapter.step`
returns a `PlannerOutput` with correct shapes (K is 7×14), all-finite values,
`mpc_dt == 0.04`; stepping the adapter with the plan clock reproduces the
standalone task references exactly (catches the `mjx.Data.time` indexing
gotcha); `warmup` → `reset_runtime_state_after_warmup` → `step` sequencing
works; unimplemented viz hooks return None without raising.

**V-B2 — sim bringup parity** (`sbmpc_franka_bringup backend:=mujoco`, LFC
applying `tau_ff + K(x_des − x)` at 1 kHz). Run the V-A4(a) scenario through
the full launch. Pass:
- `validate_sim` passes;
- tracking and margin metrics within 1.5× of the V-A4(a) report (any excess
  is machinery-caused by construction — investigate the bridge, not the
  controller);
- 25 Hz planner deadline misses < 1 % of cycles over the reach; watchdog
  never trips; replay recording produced and re-playable.

**V-B3 — regression net.** Existing `sbmpc_ros` test suites still pass
untouched: model-parity, EE-parity smoke, bridge config, safety, msg adapter,
watchdog. (A failure means the backend switch leaked.)

**V-B4 — robot protocol** (Phase 5; each stage is go/no-go, every run
recorded with the existing replay tooling):
1. *Offline replay*: feed recorded real joint-state streams (from previous
   sessions) through the hydrax planner offline; V-A3 gain-health bounds and
   the 40 ms budget must hold on real signals before the robot moves.
2. *Feedforward reach* at `max_velocity_fraction=0.10`, then 0.20: no reflex
   trips; terminal EE error ≤ 1.5× the V-B2 value.
3. *Exact feedback* at 0.10, then 0.20: no reflex trips; tracking ≤ the
   feedforward run at the same fraction; logged ‖K‖ time series satisfies
   V-A3 bounds throughout.

### Phase → gate map

| Phase | Gate = all of |
|-------|----------------|
| 0 | hydrax example runs on the GPU; Panda scene loads and steps in MJX |
| 1 | V-A1 |
| 2 | V-A2 (a–c), cycle-with-gains ≤ 40 ms |
| 3 | V-A3, V-A4 (a–e) |
| 4 | V-B1, V-B2, V-B3 |
| 5 | V-B4 stages 1→3 |

---

## Phases

**Phase 0 — scaffolding.** Branch `feedback-mppi-panda` off `main`
(base `33ec819`) ✅; Panda model + meshes copied to `hydrax/models/panda/` ✅;
uv environment synced, GPU smoke of an existing task passed (pendulum MPPI,
steady ~2 ms) ✅. Remaining: load + step the Panda scene in MJX.

**Phase 1 — task + feedforward.** `PandaPregraspTask` (IK + min-jerk +
inverse-dynamics references generated at construction) + plain hydrax `MPPI`,
`panda_pregrasp.py` with the multi-rate loop (feedforward mode). Gate: V-A1.

**Phase 2 — FeedbackMPPI.** Implement the gain path. Gate: V-A2, timing.

**Phase 3 — gain quality at the source (user direction).** Make K deployable
from the optimizer, no downstream conditioning. Levers: cost normalization,
sampling std / temperature (V-A3's ESS metric is the direct readout of
degeneracy), gain-batch size, knot count. Wire `--mode feedback` + scenario
flags into the example. Gate: V-A3, V-A4. Freeze the tuned config as
canonical before Tier B.

**Phase 4 — ROS integration.** `HydraxPlannerAdapter` implementing the
duck-typed surface `SbMpcPlannerAdapter` consumes today, selected by a
backend switch (lands on the consolidation-plan launch structure);
`sbmpc_ros` otherwise unmodified. Decide here how the bridge env imports
hydrax (uv env vs install into pixi). Gate: V-B1, V-B2, V-B3.

**Phase 5 — robot.** Gate: V-B4; definition of done.

**Phase 6 — thin bridge rebuild (deferred).** Execute the `sbmpc_ros/doc/`
consolidation plan informed by what the hydrax planner actually needs; keep
LFC msg adapter contract, joint mapping, safety, watchdog, replay, parity
tests. V-B2/V-B3 rerun as the regression net for the rebuild.

## Risks

- **R1 — gains-path compute.** `jax.jvp` through `mjx.step` (full contact
  physics) may blow the 40 ms budget (14 tangents × 128 gain rollouts).
  Mitigations: the pregrasp reach is contact-free → disable contacts in the
  planning model; reduce gain batch; measured at the Phase 2 gate.
- **R2 — softmax degeneracy persists.** If the Phase 3 levers can't satisfy
  V-A3 + V-A4, the fallback is feedforward deployment while Phase 3
  continues — not bridge conditioning.
- **R3 — upstream drift.** In-tree fork; pinned base commit (`33ec819`);
  rebase deliberately, not continuously.

## Reuse map

| Asset | Source | How used |
|-------|--------|----------|
| Panda MJX model | sbmpc model tree | copied to `hydrax/models/panda/` (done) |
| Min-jerk plan + pregrasp IK | concept from sbmpc | **reimplemented** small and self-contained inside the task (quintic + DLS IK + MuJoCo inverse dynamics) |
| Cost structure + starting weights | sbmpc pregrasp config | rewritten directly in the task; retuned in Phase 3 |
| Gain math | F-MPPI paper implementation in sbmpc | ported into `feedback_mppi.py`, re-proven by V-A2 |
| FD gain-check methodology | 2026-07-01 probes | V-A2 |
| ROS stack (bridge, safety, watchdog, bringup, benches) | `sbmpc_ros` | unchanged until Phase 4 switch; rebuilt in Phase 6 |
| Bench facts | 25 Hz budget, ROS_DOMAIN_ID=29 | operational knowledge only |
