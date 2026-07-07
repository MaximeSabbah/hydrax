# Feedback-MPPI Panda port plan (sbmpc → hydrax)

Status: **in progress** — written 2026-07-03. Phase 0 mostly done (branch, model, env).
Scope: implement the Feedback-MPPI (Riccati-gain) controller and the Panda
pregrasp reach problem in this hydrax fork, then swap it in as the planner
backend behind the existing `sbmpc_ros` stack.

## Decisions log

| Date | Decision |
|------|----------|
| 2026-07-06 | **`FeedbackMPPI` is a self-contained `SamplingBasedController` subclass** (user direction): every alg in `hydrax/algs` derives from the base directly, so the F-MPPI file carries its own (deliberately MPPI-identical) update, and the gain code mirrors sbmpc's `gains.py`/`solvers.py` structure so the two implementations stay diff-able. |
| 2026-07-03 | **sbmpc results are not a reference — its implementation is.** No baseline fixtures, no parity comparisons against sbmpc runs; all validation gates are absolute and self-contained in hydrax. sbmpc remains the reference for *code*: the gain implementation, how the dynamics is wired, the 25 Hz control-loop timing structure, and solver parameter starting points. |
| 2026-07-03 | **Cost functions are hydrax-native.** Written in the house style of the existing hydrax tasks (plain quadratics, O(1) weights); sbmpc's cost magnitudes are never copied — they belong to a differently normalized pipeline. |
| 2026-07-03 | **Future (post-plan): an "optimizer" task mode** where MPPI finds its own trajectory from meta costs (e.g. end-effector pose), like sbmpc's optimizer fallback. Out of scope until trajectory tracking is validated; the task keeps its reference machinery separable so this mode can be added without a rewrite. Design record for it: the `references:` and `trajectory.horizon_reference: window\|constant` semantics in sbmpc's `ocp_configs/pregrasp.yaml` (that yaml is otherwise superseded — it remains only the tuning surface of the sbmpc fallback backend until Phase 6). |
| 2026-07-03 | Port lives **in-tree in this hydrax fork** (`hydrax/algs/`, `hydrax/tasks/`, `examples/`), not a separate package. Upstream (`vincekurtz/hydrax`) stays a configured remote; rebase cost accepted. |
| 2026-07-03 | **No bridge-side gain conditioning.** EMA filtering / norm clamping / `feedback_gain_scale` in the ROS bridge are rejected. K must come out of the optimizer deployable; gain quality is a solver-level requirement (Phase 3). |
| 2026-07-03 | **Two-tier validation.** Tier A validates the controller entirely inside this repo (one example script + pytest, no ROS). Tier B validates the machinery through the `sbmpc_ros` launch with LFC. A phase may not enter Tier B until its Tier A checks pass. |
| 2026-07-03 | **Single example script.** One `examples/panda_pregrasp.py` with `--mode feedforward\|feedback`; its simulation loop is always multi-rate (1 kHz plant / 25 Hz planner, planner outputs zero-order-held between updates) to mirror LFC exactly. No separate multirate script. |
| 2026-07-03 | hydrax runs in its own **uv** environment (per hydrax README), not the sbmpc pixi env. How the Phase 4 bridge imports hydrax is decided at Phase 4. |
| 2026-07-03 | **Phase 1.5: early Tier B with the feedforward controller.** The validated V-A1 controller goes through the ROS/LFC machinery now (adapter + backend switch + V-B1/2/3), so integration unknowns are debugged with a trivial controller and the multi-rate LFC-law model is validated against the real LFC before Phase 3 tunes against it. Phases 2–3 then proceed hydrax-side; old Phase 4 shrinks to flipping the adapter to feedback mode + rerunning Tier B. |
| 2026-07-03 | **The hydrax planner node runs in its own uv env** — hydrax is NOT installed into the sbmpc pixi env. A uv wrapper/supervisor mirrors the existing pixi one: ROS paths sourced, uv venv interpreter wins. |
| 2026-07-03 | **Both backends coexist in sbmpc_ros until Phase 6** (default, final call deferred by user): sbmpc stays the robot-validated fallback and regression baseline; the backend switch is a launch argument per the consolidation plan, not deep dual plumbing. |
| 2026-07-02 | `sbmpc_ros` stays **frozen** as the test harness until the hydrax planner is validated in sim (backend-switch integration adds only the adapter + launch path). The thin bridge rebuild (Phase 6) happens only after, per the consolidation plan in `sbmpc_ros/doc/`. |

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
terms: position, velocity, control-around-feedforward — plain quadratics with
O(1) weights in the house style of the existing hydrax tasks.

`examples/panda_pregrasp.py` simulates deployment faithfully in both modes:
plant integrated at 1 kHz, planner called at 25 Hz, and between planner
updates the applied torque is `tau_ff + Kp(q_des − q) + Kd(v_des − v)`. In
**feedforward mode Kp/Kd are fixed joint-impedance gains** (LFC with
constant gains — the low level, not the planner, stabilizes the
open-loop-unstable arm, exactly as deployed; sync the values with the LFC
config before Tier B). In feedback mode the F-MPPI gains take this role
(how they combine with or replace the fixed gains is a Phase 3 design
point). The torque-margin metric is computed on the *unclipped* command:
a demand beyond the limits trips the real robot's reflexes rather than
silently saturating. Runs are headless and record their trajectory;
`--replay` plays a recorded run in the viewer with the reference plan as a
transparent ghost robot (user decision: visualization is replay-based, not
live).

`FeedbackMPPI` (self-contained `SamplingBasedController` subclass, house
pattern of `hydrax/algs`; the policy update is deliberately identical to
`MPPI`'s):
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

## Parameter architecture (settled 2026-07-03; who sets what, concretely)

**One tuning surface**: `hydrax/configs/pregrasp.yaml` holds the OCP tuning
(cost weights, solver knobs, plan timing) — values in hydrax-native names.
Both the Tier A example and the ROS planner adapter load it through the
same `hydrax.configs.load_pregrasp_config()`, so the gates always certify
exactly the file that deploys, and there is no promotion step: you tune
the file, rerun a Tier A gate and/or relaunch the ROS sim, done. The ROS
parameter layer carries **zero** tuning values (a guard test enforces it)
— this keeps sbmpc's good idea (its yaml header: "the single tuning
surface") while removing what broke it: nothing competes with or silently
overrides the file, and the loader validates every key against the
dataclass schema so typos fail loudly instead of being ignored.

Two orthogonal launch axes: `backend:=mujoco|real` (where the plant is) and
`planner:=hydrax|sbmpc` (which solver; **defaults to hydrax**, user
decision 2026-07-03). All four combinations are valid. The `planner`
argument drives everything it implies: the bridge preset yaml, the runtime
wrapper (`uv_ros_run.sh` | `pixi_ros_run.sh`), and the node's
`planner_impl` parameter.

```
hydrax/configs/pregrasp.yaml       THE tuning surface (values today;
                                   see growth path below)
hydrax/configs/__init__.py         load_pregrasp_config(): yaml → typed
                                   dataclasses; unknown keys are an error
hydrax/tasks/panda_pregrasp.py     PandaPregraspOptions (problem: costs,
                                   goal, start, limits, kp/kd impedance,
                                   DR) + PregraspControllerConfig (solver)
                                   — the typed schema; defaults fill keys
                                   the yaml omits; non-yaml fields (goal,
                                   limits, impedance...) are code-only
examples/panda_pregrasp.py         loads the yaml, pairs task+FeedbackMPPI
                                   (the ~8-line glue hydrax examples always
                                   contain); CLI flags are run modes only
sbmpc_ros_bridge/
  hydrax_planner_adapter.py        same loader + same pairing glue (V-B1
                                   asserts equivalence); plan clock; mode
                                   switch = which K is published
  lfc_bridge_node.py               planner_impl param selects the adapter
sbmpc_bringup/
  launch/sbmpc_franka_bringup...   planner:=hydrax|sbmpc axis
  config/hydrax_bridge.yaml        transport ONLY (topics, 25 Hz rate,
                                   warmup, planner_impl, planner_mode)
```

**Growth path — the yaml becomes the OCP description (planned feature).**
Today the file carries tuning values; the intent is to grow it into
describing the OCP itself: selecting the task mode (tracking | optimizer)
and composing **code-defined** cost terms with their weights, parameters
and reference sources — what sbmpc's `running_terms`/`references` sections
were reaching for. The invariants that keep it sane do not change as it
grows: one file, schema-validated by the loader (every new yaml capability
lands together with its schema and dataclass), nothing in the ROS layer can
override it, and the cost terms themselves are implemented and reviewed as
Python in the task — the yaml selects and parameterizes them, it never
defines math.

Worked examples of the flow:
- *Try 2048 samples on the robot*: set `planner_num_samples: 2048` in
  `hydrax_bridge.yaml`, relaunch. Code untouched, experiment recorded in
  the preset.
- *Make it the new default*: change `PregraspControllerConfig.num_samples`
  in hydrax, rerun the Tier A gates, commit — every consumer (example,
  adapter, tests) picks it up because all of them go through the factory.
- *Move the goal pose*: `PandaPregraspOptions.goal_pos` — a problem change,
  so it is a code change with a gate rerun, not a launch knob (individual
  options get promoted to ROS params only when a deployment genuinely needs
  to vary them).

## Parameters (tuned in the 2026-07-03 Phase 1 sweeps; Phase 3 retunes freely)

| parameter | value | note |
|-----------|-------|------|
| plan horizon | 0.4 s (task `dt=0.04` × 10 steps) | |
| num_samples | 1024 | |
| temperature | 0.01 | 0.002 destabilizes (sharp selection) |
| noise std | **0.03 × per-joint τ_max** | per-joint, not scalar: 4.0 mm vs 10.3 mm terminal error at matched shoulder authority |
| num_knots | 4, `spline_type="cubic"` | |
| iterations | 1 | |
| fixed impedance | Kp [1000,1000,1000,1000,20,10,5], Kd [5,5,5,5,2,2,1] | feedforward-mode low level; real LFC values (user, 2026-07-03) |
| gain batch | 128 (Phase 2) | |
| reach duration | 7.5 s, `max_velocity_fraction` 0.20 | |

**Phase 1 findings (2026-07-03).** The torque-driven arm is open-loop
unstable along the reach (errors e-fold ~0.2 s: a tau_ff-only replay on the
1 kHz plant explodes by t≈3 s; Coulomb friction does not save it). Without
a low-level stabilizer, MPPI's sampling noise doubles as the feedback
authority and must be huge (0.12·τ_max reached only ~12 mm); with the
deployment-realistic fixed-impedance low level, noise returns to small
(0.03·τ_max) and V-A1 passes at 4.0 mm with 9 ms p95 solves. Per-joint
noise remains necessary either way — the torque action space is
heterogeneous (87 vs 12 Nm), unlike the G1 tasks whose position-servo
actuators normalize the action space (which is why plain MPPI never needed
it). The zero-noise nominal sample improves terminal error ~2×. Steepening
the saturated cost destabilizes — the mocap-style cost stays untouched.
MPC value-add over impedance-only: 4.0 mm vs 21 mm terminal error (measured
with stiff placeholder wrists). With the **real LFC gains** (soft wrists,
Kp 20/10/5) the terminal error is **9.9 mm — a marginal PASS** against the
10 mm gate (run variance ±3 mm); joint 6 carries 29 mrad RMS, so the wrist
burden falls on tau_ff — margin should come from Phase 2/3's MPPI gains
(or, if needed sooner: iterations=2 or higher wrist noise_std).
**V-A1: PASS** (latest report `validation/reports/V-A1_feedforward.json`,
run log in `validation/reports/history.jsonl`).

---

## Validation protocol

Two tiers. **Tier A** runs entirely in this repo (the example script + pytest,
plant = same MJX model) and validates the *controller* on absolute criteria.
**Tier B** runs through the `sbmpc_ros` launch and validates the *machinery*:
adapter contract, bridge timing, LFC application of (tau_ff, K), then the
robot. The A→B boundary is the attribution line: V-B2 runs the same scenario
as V-A4(a), so any behavioral delta between them is ROS/transport/LFC, not
the controller.

Every check writes a report JSON (metrics + pass/fail + git SHA + config)
and a replayable trajectory recording to `validation/reports/`, under a
**fixed name per check+mode — each run overwrites the last**, so artifacts
never accumulate. The longitudinal record is `history.jsonl` (one compact
line per run: date, SHA, verdict, key metrics). The directory is gitignored;
milestone numbers that matter are quoted in this document.

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

**V-A5 — initial-configuration robustness** (`panda_pregrasp_q0_sweep.py`,
pre-robot gate, added 2026-07-07). On the real robot the arm is hand-placed
near — not at — the home pose before each session, but the min-jerk
reference plan always starts at the hardcoded `start_q`: the adapter builds
the task once at startup and never re-seeds the plan from the measured
state (`step()` does solve from the measured state — only the reference
anchor is fixed). The sweep starts the PLANT from perturbed configurations
(uniform per-joint offsets, default 0.05/0.1/0.2 rad × 8 seeds, clipped
clear of the joint limits) while the reference stays nominal, driving
`panda_pregrasp.py --q0_noise MAG --q0_seed K` one process per run. Pass,
per run:
- the full per-run gate set (margins, no-NaN, solve timing, gain health in
  feedback mode) minus the gates that already fail on the certified nominal
  report — those are known-moot at the frozen config (`health_k_smoothness`:
  the K estimate is sample-noisy step to step, identical value on nominal
  and perturbed runs) and carry no robustness signal;
- landing: terminal EE error ≤ 15 mm (the 10 mm nominal gate widened by the
  hold-wander spread — the terminal error is one draw from the hold jitter
  and the nominal run already sits at 9.9 mm) and hold q̇ RMS ≤ 0.05 rad/s.

Both modes are swept. Feedforward's fixed impedance servos the reference
from t = 0, so a hand-placed offset produces an immediate `kp·Δq` torque
demand at arming — the feedforward sweep quantifies how much offset the
flip-back mode tolerates before the torque margin (or the real robot's
reflexes) trips. Failing runs keep their report + trajectory for `--replay`;
the aggregate grid lands in `validation/reports/V-A5_q0_robustness_<mode>.json`.

**Results (2026-07-07, frozen linear-spline config):**
- *feedback (exact_feedback, the deployed mode): 24/24 PASS.* Placement
  offsets up to ±0.2 rad/joint (offset norms 0.06–0.36 rad) are absorbed
  within the reach: every run lands at 9.9 mm and settles at hold q̇ RMS
  ≤ 0.022. The closed loop is strongly contracting — perturbed starts
  converge onto the nominal trajectory before the mid-reach peaks, so the
  run maxima match nominal to print precision (worst τ fraction 0.378 =
  the nominal value at every magnitude; only the velocity peak shows the
  transient at ±0.2, 0.15 → 0.29, still under the 0.80 gate). ESS min
  94 everywhere; no NaN; solve timing unchanged. Mechanism: the solver
  re-plans from the measured state every cycle and the placement error
  decays over the first ~5 s of the reach (smoke trace: EE error rejoins
  the nominal tube by t ≈ 6 s at ±0.1 rad).
- *feedforward (the flip-back mode): 6/8 PASS at ±0.05, 1/8 at ±0.1, 0/8
  at ±0.2 rad — aggregate FAIL, by design a tolerance measurement.* The
  fixed impedance demands `kp·Δq` at arming: worst commanded demand 2.5×
  τ_max and 3.2× the velocity limit at ±0.2 rad (in sim the actuators
  clamp and the run still lands at 0.8 mm; on the real robot this trips
  the reflexes). Deployment rule derived: **only arm feedforward mode
  with the arm parked at the home pose**; hand placement is certified for
  exact_feedback only. Recorded in the README robot protocol (Step 4).

*Tier B counterpart (2026-07-07, user-requested): `initial_q:=home|random`
launch argument*, sim-only, for watching the placement gap absorbed live.
`random` regenerates the scene's `home` keyframe through the planner
runtime (`sbmpc_bringup/initial_q.py`): qpos drawn fresh each launch
within the certified ±0.2 rad envelope (clipped clear of joint limits)
AND the keyframe's gravity-compensation ctrl recomputed at the drawn pose
(it holds the arm before the LFC stack activates — a moved qpos with the
home ctrl would sag). The generated scene is written next to the
install-space original (relative include/assets resolve unchanged,
gitignored) and reaches the sim via the xacro `mujoco_model` override;
`initial_keyframe` and the post-warmup ResetWorld still say `home`, so no
other wiring moves. The draw is logged at launch. Verified end-to-end
headless: sim loaded the generated scene, armed, reach completed from a
draw with offsets to 0.19 rad — 0 deadline misses / 3304 solves, planning
30.5/32.4 ms, task error min 1.7 mm, gains 9–21, 2697/2697 accepted (same
numbers as the nominal V-B2 run). Unit tests `test_initial_q.py` 4/4;
launch argument-surface guard updated (ten arguments).

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
| 5 | V-A5 (before the robot moves), then V-B4 stages 1→3 |

---

## Phases

**Phase 0 — scaffolding.** Branch `feedback-mppi-panda` off `main`
(base `33ec819`) ✅; Panda model + meshes copied to `hydrax/models/panda/` ✅;
uv environment synced, GPU smoke of an existing task passed (pendulum MPPI,
steady ~2 ms) ✅. Remaining: load + step the Panda scene in MJX.

**Phase 1 — task + feedforward.** `PandaPregraspTask` (IK + min-jerk +
inverse-dynamics references generated at construction) + plain hydrax `MPPI`,
`panda_pregrasp.py` with the multi-rate loop (feedforward mode). Gate: V-A1.

**Phase 1.5 — early Tier B (feedforward through ROS/LFC). COMPLETE
(2026-07-05): V-B1 7/7, V-B2 PASS, V-B3 124/0.** The hydrax feedforward
controller ran through the full `backend:=mujoco` bringup with LFC at
1 kHz: EE error 11.3 mm at plan end converging to 1.4 mm after ~8 s of
hold (0.1 mm floor in long holds), solve wall mean 10.6 / p95 12.7 ms,
**0 deadline misses over 6,138 solves**, `validate_sbmpc_sim` verdict
"stable", 400/400 planner outputs accepted. Tier A ↔ Tier B parity is
essentially exact — torque margin 0.39 vs 0.39, velocity margin 0.14 vs
0.14 — confirming the multi-rate Tier A loop is a faithful model of the
deployed LFC law (the attribution line holds). Reports:
`validation/reports/V-B2_feedforward.json`, replay `V-B2_replay.json`.
Original step list:
1. env spike: rclpy + hydrax + JAX-GPU importable in one uv-run process;
2. uv wrapper/supervisor (mirroring the pixi one) + launch path;
3. `HydraxPlannerAdapter` implementing the `PlannerOutput` surface
   (tau_ff, q_des/v_des from the plan clock, and K — see mode switch below);
4. V-B1 contract test (pure Python, no ROS runtime) — must pass before any
   launch is touched;
5. `backend:=mujoco` bringup → V-B2 parity against the V-A1 report; V-B3
   regression suite.
Gate: V-B1, V-B2, V-B3 (feedforward scope).

*Where the feedforward/feedback switch lives:* in ONE place — the mode
parameter of `HydraxPlannerAdapter` (mirroring sbmpc's `gain_mode` bridge
parameter, selected by the same preset/launch argument). LFC always applies
`tau = tau_ff + K(x_des − x_meas)` with whatever K arrives, and the bridge
machinery is mode-agnostic; the mode only decides **which K the adapter
publishes**: the constant impedance matrix (feedforward, Phase 1.5) or the
FeedbackMPPI gains (feedback, Phase 4; whether they add to or replace the
fixed impedance is settled in Phase 3). The Tier A example's `--mode` flag
is the same switch in the same place — the K source, nothing else.

**Phase 2 — FeedbackMPPI. COMPLETE (2026-07-06): V-A2 PASS, timing PASS.**
Gain path implemented (`compute_gains=True` → K in `params.gains`;
`num_gain_samples` joined the tuning surface). FD relative Frobenius
error: (a) toy 0.020 % / (b) Panda fresh 0.121 % / (c) warm mid-reach
0.118 % against tolerances 1/2/10 %. Cycle-with-gains at the deployment
config (1024 samples, gain batch 128): mean 21.1 ms, p95 22.6 ms against
the 40 ms budget. Warm-started ‖K‖_F ≈ 23 with fresh ≈ 23 — none of
sbmpc's 0↔2465 flapping, consistent with the O(1) normalized costs
removing the degeneracy driver at the source (V-A3 quantifies in
Phase 3). Finding: jvp tangents through MJX **joint-limit constraint
rows** can go NaN even far from the limit (an MJX differentiability
quirk, control-value dependent — V-A2(a) therefore tests the formula
with the toy's limits disabled). The pregrasp model is constraint-free
by design: 0/1024 non-finite gradients measured; the nan_to_num guard
inherited from sbmpc covers stray cases, and the non-finite-gradient
count is a candidate V-A3 health metric (to ratify in Phase 3).

**Phase 3 — gain quality at the source (user direction).** Make K deployable
from the optimizer, no downstream conditioning. Levers: cost normalization,
sampling std / temperature (V-A3's ESS metric is the direct readout of
degeneracy), gain-batch size, knot count. Wire `--mode feedback` + scenario
flags into the example. Gate: V-A3, V-A4. Freeze the tuned config as
canonical before Tier B.

*Phase 3 status (2026-07-06) — feedback mode implemented and measured;
BLOCKED on design decisions (below).* The feedback law is the paper's pure
replacement `tau = tau_ff + K (x − x₀)`, x₀ = the state the plan was
solved from (user decision; K's anchor is the linearization point, not
the tracking reference — anchoring at the reference double-counts the
tracking error already baked into tau_ff). Solver exposes the V-A3
readouts (`gain_ess`, `gain_nominal_weight`); the example implements the
V-A4 scenarios and evaluates the V-A3 gates on the applied K series.
Retune campaign over 12 configs (temperature 0.002–0.01 × iterations 1–3
× noise 0.02/0.03 × knots 4/6 × samples 1024/2048 × weight rebalance ×
saturated-vs-quadratic cost). Frozen candidate: **T=0.007,
iterations=3** (in the yaml). Findings, all measured:
1. The retune helps FEEDFORWARD dramatically: terminal 9.9 → **2.3 mm**,
   mean RMS 6.8 → 2.6 mrad, wrist RMS 29 → 7 mrad (new V-A1 baseline).
2. Pure-replacement feedback is **stable everywhere** (all margins, no
   NaN, latency 0.04 s tolerated — the historical robot killer; mass
   ±10 % stable) with healthy gains (ESS ≥ 32, nominal never dominant,
   ‖K‖_F ≈ 15–35, no flapping), terminal 5.9 mm — but it is dominated
   by the retuned feedforward on every tracking metric: nominal RMS
   15.4 vs 2.6 mrad (moving-reference lag ∝ reference speed, converges
   during hold), disturbance peak 0.101 vs 0.024 rad (V-A4(b) fails 4×
   the wrong way), mass-0.9 sag 95 vs 4 mrad.
3. Structural cause: at a healthy softmax the deployable K is O(10–30)
   N·m/rad — it cannot match a Kp≈1000 impedance's rejection;
   sharpening temperature to chase stiffness collapses ESS to 1–13 and
   reproduces sbmpc's degeneracy (controlled demonstration of the same
   disease). Cost shape is NOT a lever: saturated vs plain quadratic is
   identical at tracking error scales (diagnostic run, reverted).
4. **V-A2 × iterations tension:** the gain formula is exact only at
   iterations=1; at the frozen it=3 config the FD error is 43 %/78 %
   (fresh/warm) — the formula covers only the final iteration's update.
   Candidate fix to evaluate: per-iteration gain summation
   (K ≈ Σ_iters K_i, leading-order chain), gated by the same FD test;
   costs ~3× the gain compute.
5. Cycle-with-gains at it=3: 37.0 / 38.9 ms (mean/p95, under partial
   GPU contention) — fits the 40 ms budget.
6. Gate calibration issues found: `health_k_smoothness` fails in every
   run at small K (ΔK ≈ 0.3–1.5 × ‖K‖ per solve, but the torque step
   ΔK·(x−x₀) ~ 0.01 N·m — the gate should bound the applied-torque
   discontinuity, not relative ΔK); V-A4(a)'s mean-RMS comparator
   structurally favors the 1 kHz stiff impedance (user goal is terminal
   precision).
*Pending user decisions:* deployment law for feedback mode (additive
impedance+K — empirically motivated now — vs feedforward-only fallback
per R2 vs keep pure and re-scope gates); V-A4(a)/(b) comparators; V-A3
smoothness recalibration; the iterations×gains resolution.

**Phase 4 — ROS integration.** `HydraxPlannerAdapter` implementing the
duck-typed surface `SbMpcPlannerAdapter` consumes today, selected by a
backend switch (lands on the consolidation-plan launch structure);
`sbmpc_ros` otherwise unmodified. Decide here how the bridge env imports
hydrax (uv env vs install into pixi). Gate: V-B1, V-B2, V-B3.

*Phase 4 status (2026-07-06) — exact_feedback deployed through ROS/LFC
in sim.* User direction: deploy feedback despite the Phase 3 tracking
gap — the future optimizer mode has no reference trajectory, so the
F-MPPI law is the only viable low level there; interim performance
accepted. Implemented: adapter `exact_feedback` mode (K =
`FeedbackMPPIParams.gains` in du/dx convention; `gain_ess` /
`gain_nominal_weight` in diagnostics; `compute_gains` only in this
mode), anchor semantics ride the existing machinery — the bridge's
reference substitution is feedforward-gated, so `control.initial_state`
stays the measured solve state x₀ and LFC applies
`tau_ff + K (x − x₀)`. **The mode switch is `hydrax_bridge.yaml`'s
`planner_mode` (user decision: preset yaml, no launch argument; the
install space must be rebuilt after editing it).** V-B1 10/10 (new
contract tests: solver gains published, K_lfc = −K, x₀ anchor
round-trip); guard tests updated (preset pins exact_feedback); V-B3
124/0. **V-B2 (exact_feedback, backend:=mujoco): functionally clean** —
armed after warmup, full reach converged (task error min 2.8 mm),
1917/1917 outputs accepted, no watchdog trips, ‖K‖ 9–27 flowing
end-to-end. Two deployment findings:
1. **25 Hz deadline: no headroom.** Cycle mean 38.8 / p95 41.2 ms →
   ~24 % deadline misses (an external process held the GPU at 60–100 %
   throughout — timing polluted, but the margin is structurally ~zero
   at iterations=3). Mitigations to choose after a free-GPU
   re-measurement: iterations 3→2 (~31 ms, terminal 8.1 mm),
   num_gain_samples 128→96/64, jit get_action (−0.4 ms).
2. **Hold-phase wander.** With the reference static, the pure law has
   no position anchor: task error oscillates 2.8→23 mm during the hold
   (tail joint spans ≈ 0.04 rad, joint-velocity RMS 0.044) — the arm
   random-walks on solver sampling noise restrained only by the weak
   K-spring. Visible at the pregrasp pose on the robot; options: accept
   (research mode), a phase machine that switches mode after the reach
   (future planner-side work), or revisiting the law.

*Phase 4 finale (2026-07-06) — both issues addressed, config frozen.*
User confirmed both findings in their own viewer run. Measured fixes:
- Dead ends (all measured, kept out): `mean_adaptation_rate` α < 1
  (CEM-style partial mean update — implemented, FD-verified, but α 0.5
  /0.3 worsened terminal AND wander: the arm drifts on the weak
  K-spring and a slower mean update slows the chase; the knob stays at
  the neutral 1.0), velocity_weight ↑ (wander −25 % but chronic ESS
  collapse), gain batch 64 (the gain path is latency-bound in the H
  sequential steps, not batch-bound), `jax.linearize` gains (tape is
  memory-bound; the vmap-of-jvp recompute is faster — documented in
  the code).
- **The winning lever: plan_horizon 0.4 → 0.32 s** (H 10 → 8), cutting
  the sequential depth shared by sampling and gains: cycle 29.7/31.2 ms
  (9 ms margin), terminal 7.5 mm, ESS min 87, wander 0.0184 rad — the
  best measured point on every axis simultaneously.
- **Frozen config**: T=0.007, plan_horizon=0.32, iterations=3,
  1024 samples, gain batch 128, noise 0.03·τ_max, weights 1.0/0.1/0.01,
  mean_adaptation_rate 1.0, **spline_type=linear** (user choice
  2026-07-07 after the spline probe: linear is measurably calmer than
  cubic at the hold — no interpolation overshoot between noisy knots —
  wander span 0.0159 vs 0.0184 rad, hold q̇ RMS 0.0218 vs 0.0267;
  terminal 9.9 vs 7.5 mm, both under the 10 mm gate; "zero" fails
  terminal at 18 mm. Re-gated at linear: V-A1 ff 0.77 mm / wander
  0.006; V-A4 nominal 9.9 mm, ESS 94, scenarios all stable ESS 92–96;
  V-A2 4/4 FD ≤ 0.21 %, cycle 29.6/31.4 ms; V-B1 10/10; **V-B2
  0 deadline misses / 946 solves, mean 30.3 / p95 32.1 ms, hold q̇ RMS
  0.0225** — every axis equal or better than cubic except the 2 mm of
  terminal). Earlier full re-gate at the cubic variant: V-A1
  (feedforward baseline 1.6 mm), V-A2 4/4 (FD 0.02–0.16 %; the FD cases
  certify the formula at iterations=1, the deployed K is the
  final-iteration gain per the Phase 3 decision), V-A4 scenarios all
  stable (ESS 80–88; disturbance rejection remains 2.7× behind the
  impedance — known and accepted), V-B1 10/10 (adapter get_action now
  jitted), V-B3 124/0, and **V-B2: 1 deadline miss / 928 solves
  (0.1 %, gate < 1 % PASS), planning mean 30.7 / p95 32.4 ms,
  928/928 outputs accepted, exact_feedback end-to-end.** Residual hold
  wander ≈ 0.02 rad spans (joint-velocity RMS down ~35 % from the first
  deployment) — accepted for the first robot sessions; the phase
  machine stays the future option. Next: V-B4 stage 1–2 on the robot.

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
  **Resolved 2026-07-06: p95 22.6 ms at the deployment config — within
  budget with ~17 ms headroom.**
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
