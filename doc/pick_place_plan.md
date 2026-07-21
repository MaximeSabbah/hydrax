# Panda cube pick-and-place plan (trajectory tracking)

Canonical plan for the phase after the Feedback-MPPI port
(`doc/feedback_mppi_panda_port_plan.md`, V-B4 real-robot success
2026-07-09). Same discipline: hydrax-side first, gates before ROS, ROS
before robot, one tuning surface, user reviews every phase.

## Goal

The full cube pick-and-place on the real Franka: PREGRASP → DESCEND →
CLOSE → LIFT → TRANSPORT → PLACE → OPEN → RETREAT, built on the
**certified trajectory-tracking controller deployed today**. A
planner-side phase machine sequences per-phase min-jerk joint segments
(replanned at each phase entry) and the gripper events; the solver, the
cost, and the deployed law `τ = τ_ff + K(x − x₀)` are the pregrasp ones,
unchanged. Everything new lives in the task layer.

## Decisions log

| date | decision |
|------|----------|
| 2026-07-09 | **One mode: trajectory tracking.** User decision after reviewing a dual-mode draft ("the implementation of the controller must commit to one mode for now"). No `mode:` key, no swappable cost/reference layer in code. Rationale: nothing in this scope (fixed poses, no obstacles, no perception) requires optimizer-style planning; tracking is the certified strength exactly where the task is hardest (grasp precision), and every new piece built here — phase machine, IK goal tables, gripper orchestration, cube-pose robustness — is what a future optimizer mode would need anyway. Optimizer mode is deferred to a later, contained experiment measured against this working baseline (see *Deferred*). |
| 2026-07-09 | **Cube/target poses are fixed, measured by hand** (config values, like the pregrasp goal today). Perception is a later input channel, not part of this plan. Consequence: hand-measurement error (±1 cm) is a first-class robustness axis to certify (P-A3), not an afterthought. |
| 2026-07-09 | **Planner-side phase machine owns the sequence**, including gripper events. The task/adapter emits `gripper_command` with each output (`HydraxPlannerOutput.gripper_command` already exists in the contract); the bridge executes it via the backend's gripper action. One brain; the bridge stays transport. |
| 2026-07-09 | **Full sequence certified in sim before any robot session** (user choice over pick-only staging). One robot campaign at the end (P-B4). |
| 2026-07-09 | Planning model stays **contact-free in every phase** (transport ignores the cube or welds it to the EE — it is light and DR covers the mismatch; the plant keeps real contacts). Protects the hard-won 40 ms budget (frozen pregrasp solver: H=8, 1024 samples, it=3, ~30 ms). |
| 2026-07-09 | **Gripper is not in the MPPI action space.** It is a position-action device on both backends (sim `gripper_action_controller`, real `agimus_franka_gripper`); open/close are discrete events at phase transitions, with a dwell while the planner holds pose. |
| 2026-07-09 | **Model consolidation (user decision: "one model for the panda… and one scene model").** `hydrax/models/panda/` is exactly `panda.xml` (THE robot: arm + articulated gripper, deployment actuation) + `scene.xml` (THE environment: cube, in-front target, floor, sensors, gravity-comp keyframe, 1 kHz) + assets. The planning variant is **no longer a committed file**: `PandaPregrasp._derive_arm_planning_model()` derives it from `panda.xml` at load time via MjSpec (finger joints removed keeping inertia, gripper actuator + coupling dropped, contact flag disabled, dt = 0.04, home keyframe = `options.start_q`). The derivation was proven structurally identical to the deleted `pregrasp.xml` across every model field; `tests/test_panda_model.py` pins the invariants and plant parity. `pregrasp_scene.xml` deleted too — replay plays 7-joint recordings on `scene.xml`, fingers/cube at the keyframe. Full re-gate green: V-A1 0.77 mm, V-A4 9.9 mm / ESS 94 (same two known-moot fails, smoothness bit-identical), V-A2 4/4, V-B1 10/10. |
| 2026-07-10 | **P2 CLOSED (user decision — no further grid work).** P-A3 grid 25/28: q0/cube/friction axes all green; the mass axis drove three fixes now in the code (stall-engagement gating on `transition_q_tolerance`, `place_offset` z 0.002, time budget 2.0× duration). Final state: nominal + mass 1.1 pass every gate (placed 1.9/3.9 mm); mass 0.9 is a documented envelope edge (16.8 mm placement at −10 % global mass — the heavier direction, the real-deployment one, passes). `solve_mean` ~31–32 ms is the genuine pick-place number (GPU uncontended, user-confirmed) vs the inherited 30 ms gate — fine against the 40 ms/25 Hz deadline. Config frozen as committed in `configs/pick_place.yaml`. |
| 2026-07-10 | **P3 design (user-approved).** (1) **One `control_msgs/GripperCommand` action client** for both backends — the agimus gripper node serves the same action type (`/fer_gripper/gripper_action`) and internally maps close→grasp(force)/open→move, so the action *name* is the only per-backend difference (a bridge parameter). (2) **Dwell-exit waits on the action result by freezing the plan clock in the adapter** (skip `pm.update` while a gripper action is in flight): mid-dwell everything is constant, so the pause is indistinguishable from a longer dwell and the certified task layer stays untouched. (3) **Task selection reuses `planner_ocp`** in `hydrax_bridge.yaml` (yaml key + install rebuild, the `planner_mode` precedent): `pick_place` makes the adapter load `load_pick_place_config` + `PandaPickPlace` + the phase machine; default path stays pregrasp-compatible. The dwell impedance needs **zero bridge changes**: the adapter flips `diagnostics.gain_mode` per cycle (motion → `exact_feedback`, K from the solve anchored at x₀; hold → `feedforward`, fixed impedance K with `reference_q/v` = the dwell goal, which the existing bridge substitution anchors on the reference). Gripper action goal *values* (close/open positions, effort) are ROS-side deployment wiring like the existing `max_effort` in `franka_controllers.yaml`; sim defaults certified now, real values set in P4 staging. P-B2 measures the cube from mujoco_ros2_control's built-in free-joint odometry (`odom_free_joint_name: object_joint` hardware param — no custom code). |
| 2026-07-16 | **Sim gripper controller = servo-equivalent PID (user decision, after P-B2 diagnosis).** The inherited agimus-demos `gripper_action_controller` gains were written for their stack (on the real robot the grasp runs through `agimus_franka_gripper`, never this controller) and failed P-B2 three measured ways: stall threshold 0.001 sat on the grip-noise floor (close took 11 s → tripped the 10 s bridge backstop — the user-reported timeout); p=20 squeezes 0.4 N at the grasp (cube 0.49 N slipped out 3 mm into LIFT); the open stalled 3.8 mm short of goal_tolerance 0.001 and failed release verification. Fix: p=350/d=10/i=0 — algebraically the Tier A `actuator8` servo law (gainprm 350, biasprm 0 −350 −10, ~7 N squeeze) whose grasp physics P-A2/P-A3 certified — plus goal_tolerance 0.005 and stall_velocity_threshold 0.005. Rejected alternative: literal actuator8 swap (model+xacro+controller type — same physics, more moving parts). |
| 2026-07-21 | **One gain calculation throughout exact-feedback execution (user-approved after ROS MuJoCo A/B).** The automatic exact-feedback→fixed-impedance override at grasp/place precision windows is removed from both the ROS adapter and direct Hydrax example. `planner_mode=exact_feedback` now always publishes the solve's F-MPPI K anchored at x₀; `feedforward` remains the explicit whole-run fixed-impedance mode. In the paired ROS run, Feedback-MPPI-only completed PREGRASP→DONE, verified close, lifted the cube 158 mm with 10.2 mm cube-to-EE drift, and reached TRANSPORT with zero rejected outputs/deadline misses. It removed all K≈2000 packets and reduced the precision-window requested-effort step from 28.4 to 0.51 Nm in norm. Close EE error increased from 3.2 to 7.2 mm but remained within the grasp criterion. Placement remained inaccurate in both runs (108.6 vs 81.7 mm), so this decision certifies the grasp/lift law choice, not the later placement behavior. `precision_hold` is retained as a diagnostic precision-window marker only. This supersedes the 2026-07-09 dwell-law decision without erasing its historical measurements. |
| 2026-07-09 | **Tier A plant derivation (settled after two rejected drafts; file names superseded by the consolidation row above).** The user requires an articulated, movable gripper in Tier A (a no-grasp-physics simplification was rejected once the pregrasp models' frozen fingers were surfaced) but not a copy of the ROS model files. Final: `pick_place.xml` derives from `sbmpc/models/panda_pick_place/panda.xml` — the same parent `pregrasp.xml` was generated from, so arm parity with the planning model is inherited — with only the 7 arm position actuators swapped for the pregrasp torque motors; fingers, tendon-equality coupling and the position-servo gripper `actuator8` kept. The **planning model stays `pregrasp.xml` unchanged** (the MPPI plans the arm; the plant does the grasping). Plant timestep is 0.001 (deployment rate): the sbmpc original's 0.005 planning timestep is measurably unstable as a plant (arm falls within 1 s under exact gravity comp; the fer scene at 0.001 holds exactly). Place-target default in front of the robot (0.65, 0, 0.025), per user. |

## Non-goals

- Perception (camera/AprilTag/mocap) — poses are config values.
- Optimizer mode (goal costs instead of references) — deferred, see below.
- Moving or unknown objects, regrasp / failure recovery beyond a safe
  abort (disarm) on a failed grasp.
- sbmpc-backend parity for pick-and-place (hydrax planner only).
- Gripper force control (the franka grasp action's epsilon/force
  parameters are used as-is).

## Design record: sbmpc's `panda_pick_and_place.py`

The sbmpc controller (`sbmpc/controller/franka_emika_panda/panda_pick_and_place.py`)
is the reference for the *structure*, not the numbers:

- 9 phases (enum above, plus DONE); per-phase goal = object/target pos +
  clearance offsets (pregrasp 0.05 m, place 0.005 m, carry/retreat
  offsets);
- per-phase goal posture via IK at the goal pose;
- transitions on position tolerance + a hold-steps dwell (CLOSE/OPEN
  phases are pure dwells while the gripper actuates).

## Architecture: files to add

```
hydrax (this fork)
  hydrax/models/panda/panda.xml              DONE (P0 + consolidation):
                                             THE robot — articulated
                                             gripper, torque arm motors
  hydrax/models/panda/scene.xml              DONE (P0 + consolidation):
                                             THE environment — cube +
                                             in-front target + contact
                                             floor + task sensors, 1 kHz,
                                             gravity-comp keyframe. The
                                             PLANNING model is derived
                                             from panda.xml at load time
                                             (no committed variant files)
  hydrax/tasks/panda_pick_place.py           task: PhaseMachine + per-phase
                                             reference generation over the
                                             pregrasp machinery
                                             (_solve_ik, _min_jerk_plan,
                                             inverse dynamics); tracking
                                             cost unchanged; config
                                             dataclasses (typed schema,
                                             same pattern as pregrasp)
  hydrax/configs/pick_place.yaml             THE tuning surface: solver
                                             section (start from the
                                             frozen pregrasp values),
                                             cube/target poses, per-phase
                                             clearance offsets, segment
                                             durations / velocity
                                             fractions, tolerances, dwells
  examples/panda_pick_place.py               Tier A multi-rate run
                                             (contactful plant, gripper
                                             actuated, LFC-law loop as in
                                             panda_pregrasp.py) + gates +
                                             replay
  tests/test_pick_place_phase_machine.py     P-A1

sbmpc_ros
  hydrax_planner_adapter.py                  loads pick_place config when
                                             selected; passes phase +
                                             gripper_command through
                                             (fields exist)
  lfc_bridge_node.py                         gripper action client:
                                             executes gripper_command
                                             changes via the backend's
                                             action (sim GripperCommand /
                                             real agimus grasp+move);
                                             action result (grasp width /
                                             success) fed back for grasp
                                             verification
  sbmpc_bringup/config/hydrax_bridge.yaml    task selection key (which
                                             hydrax config the adapter
                                             loads) — transport-only rule
                                             unchanged: no tuning here
```

## Controller design

**Phase machine (hydrax task side).** States and transition structure as
in the sbmpc record; transition inputs are the measured state and the
config poses (no perception). Phase is reported in the planner output
(the adapter already relays a phase string) and drives `gripper_command`.
CLOSE/OPEN advance after the gripper dwell (sim: fixed dwell; ROS:
action-result confirmation — the franka grasp action reports
width/success, which is the grasp verification with a fixed cube pose).

**The reference: one chained timeline + an event-gated plan clock (P1
amendment, 2026-07-09, as built).** The original draft replanned each
segment from the measured posture at phase entry; that is mechanically
unsound — the reference arrays are compile-time CONSTANTS of the jitted
cost (swapping their contents at runtime silently does nothing, and
re-tracing violates the no-recompile requirement). The implemented
design keeps the intent: the task precomputes ONE timeline chaining
min-jerk segments goal-to-goal (velocity-capped, floored at
`min_segment_sec`; constant segments for the dwells; inverse-dynamics
feedforward throughout — the pregrasp generator, run once at
construction), and `PickPlacePhaseMachine` owns `state.time`: the clock
advances normally inside a segment and refuses to cross a segment
boundary until the arm has converged on the segment goal (tighter
`precision_q_tolerance` on the two boundaries entering dwells, where the
gripper acts). The properties survive:
- **event-driven progress**: a slow phase stalls the reference on its
  goal instead of being abandoned; each segment starts from a
  converged (measured-to-tolerance) posture, so the impedance law is
  safe at every boundary;
- **no recompilation by construction**: the reference is a constant,
  exactly like the pregrasp; the P-A2 `solve_max_ms` gate asserts it.
  Gotcha for the record: an `np.float64` leaking from the boundary
  array into `state.time` changes the traced dtype and caused a silent
  35 s recompile at the first clock stall — the clock is pinned to a
  python float and P-A1 asserts the type.

**Cost and solver: unchanged.** The pregrasp tracking cost (saturated
q/v/u errors, weights 1.0/0.1/0.01) against the active segment; frozen
solver config (T=0.007, H=0.32 s, it=3, 1024 samples, gain batch 128,
linear spline, σ=0.03·τ_max) as the starting point, re-gated.

**Deployment law — one selected mode end-to-end.** `planner_mode` is
invariant across every motion, boundary wait, and dwell. The deployed
`exact_feedback` mode always publishes the current solve's F-MPPI K
anchored at x₀. The explicit `feedforward` mode uses the fixed impedance
for the whole sequence. `precision_hold` marks the stationary grasp/place
windows in diagnostics but never changes the gain or anchor.

*Historical measurement (2026-07-09, P-A2; superseded law choice):*
pure exact_feedback delivered
**13.2 mm lateral / 10.7 mm vertical** EE error at the close command —
the controller's known ~1 cm steady-state precision, insufficient for
the 1 cm / 5 mm grasp gates. **The fixed-impedance dwell override was
adopted** with settle-then-actuate timing: for the first
`dwell_settle_sec` of each dwell the 1 kHz law was the stiff impedance on
the stationary dwell reference, and only then did the gripper actuate.
Result: **1.5 mm lateral / 1.0 mm vertical** at the close command, cube
placed 8 mm from target — P-A2 all 16 gates PASS. The 2026-07-21 decision
above supersedes that automatic override.

*Superseding measurement and decision (2026-07-21, ROS MuJoCo A/B):*
with the current task, gripper, and ROS control path, exact feedback
through the precision windows verified the grasp, lifted and retained the
cube, and reached TRANSPORT. Its 7.2 mm total close error was sufficient,
while the discontinuous fixed-impedance handover produced a 28.4 Nm
requested-effort step. The automatic dwell override is therefore removed;
the OCP and phase gates stay unchanged so the real experiment changes one
variable. Placement remains a separate open behavior.

## Validation protocol

### Tier A — in-hydrax

**P-A1 — phase machine** (pytest, no GPU): transition table, goal/IK
tables, dwell logic, gripper-command schedule; goals move correctly with
perturbed cube/target config values; per-phase segments start at the
handed-in posture and respect velocity caps.

**P-A2 — full sequence, nominal** (`examples/panda_pick_place.py`):
contactful plant at 1 kHz, planner at 25 Hz, gripper actuated by the
phase machine. Pass: cube placed within 1 cm of the target; every phase
converges and transitions (no stall); EE within grasp tolerance at CLOSE
(≤ 5 mm vertical, ≤ 1 cm lateral — the dwell-precision measurement that
decides the law schedule); torque ≤ 0.90 / velocity ≤ 0.80 margins; no
NaN; solve mean ≤ 30 ms, p95 ≤ 40 ms, **no solve-time spike at phase
transitions** (the no-recompile gate); F-MPPI health gates (ESS, diag
caps, no-flap) across phases; grasp holds (cube tracks the EE through
TRANSPORT — detects slip in sim).

**P-A3 — robustness grid** (sweep driver, V-A5 pattern):
- initial arm configuration ±0.2 rad (machinery exists);
- **cube-pose error**: true cube up to ±1 cm (x, y) off the config value
  — the fixed-measured-pose decision makes this the deployment-critical
  axis; the blind grasp must still succeed;
- plant mass/friction randomization on cube and arm (±10 % arm masses as
  in V-A4(c); cube friction is the new slip axis).
Judged baseline-relative like V-A5 (known-moot gates from the nominal
report ignored); grasp success required on every run.

### Tier B — through sbmpc_ros

**P-B1 — adapter/bridge contract** (pytest, no ROS runtime): pick-place
config loads through the same glue (config-match asserts); phase and
gripper_command semantics (command changes exactly at CLOSE/OPEN entry);
K/anchor contract across every phase, including blocked dwell-entry waits
and CLOSE/OPEN precision windows; gripper action client unit-tested against
a fake action server.

**P-B2 — sim bringup, full sequence** (`backend:=mujoco`): the ROS scene
already contains the cube + target + gripper actuator. Pass: cube placed
within 1.5× the P-A2 tolerance; deadline misses < 1 %; phase sequence in
diagnostics matches Tier A; gripper action round-trips (sim
GripperCommand).

**P-B3 — regression net**: the full existing suite still passes (the
pregrasp task must be untouched by the pick-place addition).

**P-B4 — robot protocol** (staged, each go/no-go, all recorded):
1. offline: recorded real joint streams through the pick-place planner —
   phase transitions and gain health on real signals, budget holds;
2. `max_velocity_fraction: 0.10`, disarmed bringup, PD-hold verified,
   arm; PREGRASP→DESCEND only (gripper disabled) — watch precision at
   the cube without touching it;
3. add CLOSE/LIFT (first real grasp; grasp verification via the gripper
   action result width);
4. full sequence at 0.10, then 0.20.

### Phase → gate map

| Phase | Gate = all of |
|-------|----------------|
| P0 | planning scene loads + steps in MJX; plant-vs-planning model parity checks; plan reviewed |

**P1 status (2026-07-09): COMPLETE — the cube is picked and placed in
Tier A.** Built: `hydrax/tasks/panda_pick_place.py` (PandaPickPlace over
the pregrasp machinery, `PandaPickPlaceOptions` +
`PickPlaceControllerConfig` mirroring the pregrasp pair per user review,
`PickPlacePhaseMachine`), `configs/pick_place.yaml` + `load_pick_place_config`
(shared `_load_yaml_values` helper), `examples/panda_pick_place.py`,
`tests/test_pick_place_phase_machine.py`. Gates: **P-A1 9/9**; **P-A2
feedback nominal 16/16 PASS** — full sequence in 16.8 s (timeline 15.3 s
+ convergence stalls, all phases transitioned), grasp precision at the
close command 1.5 mm lateral / 1.0 mm vertical (impedance dwell), cube
lifted to 0.17 m and placed 8 mm from the target, torque margin 0.54,
velocity 0.21, ESS min 94, solve 29.6 mean / 31.1 p95 / 34.3 max ms (no
recompiles). Pregrasp untouched: V-A1 re-run bit-identical (0.7678 mm).
(decisions log). Plant built and verified: arm parity vs the planning
model IDENTICAL (joint names/ranges/damping/armature, ctrlranges, link
masses, hand subtree mass); 2 s open-loop hold under the computed
gravity-comp keyframe ctrl drifts 3e-6 rad with the cube stationary;
gripper actuates (ctrl[7]: 0 → closed, 0.04 → 0.08 m width). The layout
is `panda.xml` + `scene.xml` + assets only; the planning variant is a
load-time derivation proven structurally identical to the retired
`pregrasp.xml`, pinned by `tests/test_panda_model.py` (3/3). Post-
consolidation re-gate all green (V-A1 0.77 mm, V-A4 9.9 mm/ESS 94,
V-A2 4/4, V-B1 10/10).
| P1 | P-A1, P-A2 (nominal) |
| P2 | P-A3; dwell-precision remedy decided (recorded here) and re-gated; config frozen — **COMPLETE 2026-07-10** (decisions log; user closed the testing phase) |
| P3 | P-B1, P-B2, P-B3 — **P-B1 27/27 + P-B3 green 2026-07-10; P-B2 `complete` 2026-07-16** after the gripper-controller fix (decisions log): full sequence PREGRASP→DONE through `backend:=mujoco`, close verified in 1.4 s (stall at 0.0200), grasp held (lift 0.175 m), open `reached_goal` in 0.2 s, cube placed 12.1 mm from target (gate 15 = 1.5× Tier A), deadline misses 1/508 (0.2 %), 0 rejected, gripper round-trip 2/2 verified (`validate_pick_place --assert-complete`) |
| P4 | P-B4 stages 1→4 |

## Risks

- **Dwell precision at CLOSE/OPEN:** the 2026-07-21 ROS A/B showed the
  current exact-feedback law meets grasp/lift needs without the unsafe
  fixed-impedance handover. Continue logging EE pose/velocity at close on
  hardware; `feedforward` remains an explicit whole-run mode, not an
  automatic phase override.
- **Contact-free transport mismatch** (cube inertia unmodeled): measure
  in P-A2 by comparing plant-with-cube vs planner-without tracking; the
  cube is light, DR covers it — verify, don't assume.
- **Gripper action latency** at CLOSE (~0.5–2 s on real hardware): the
  dwell must cover the action duration; ROS-side the transition waits for
  the action result, so this is a liveness (not safety) concern.
- **Cube-pose measurement error** — promoted to a certified axis (P-A3).
- **Reference-swap correctness at transitions** (the one new mechanic):
  gated twice — segment starts at the measured posture (P-A1) and no
  solve-time spike (P-A2).
- **Rollout-marker launch issue on real** (open from V-B4, error text not
  captured): markers stay off on real until the log is examined; the
  marker path is backend-agnostic in the bridge, so this is likely
  environmental — do not let it block P-B4.

## Deferred: optimizer mode

The stated endgame (MPPI optimizing against goal costs, no reference —
where the F-MPPI low level is strictly required) is intentionally NOT in
this plan. When it comes, it arrives as a contained experiment: the phase
machine, goal/IK tables, gripper orchestration, robustness axes, and ROS
wiring built here are exactly its prerequisites, and the tracking
implementation is its measuring stick. Known homework recorded for that
day: stationary-goal wander vs tracking hold, O(1) goal-cost basin
design, and per-phase goals as traced data (mocap/userdata) to avoid
recompilation.

Same slot, second candidate (user survey request, 2026-07-09): **cuRobo
in plan-once-before-arming mode** as a collision-aware segment generator
(`plan_grasp` since v0.7.5 covers approach/retract; batch-planning during
warmup avoids GPU contention with the 25 Hz loop). Costs deferred with
it: torch+CUDA next to JAX, a third robot model under the parity
discipline, NVIDIA research license. Both candidates slot behind the
phase machine without touching the controller or the ROS contract.

## Reuse map

| exists | reused as |
|--------|-----------|
| `sbmpc/models/panda_pick_place/{panda,scene}.xml` | planning + plant scene source (cube, target) |
| `sbmpc .../panda_pick_and_place.py` | phase machine structure, goal/offset/IK tables, transition pattern — **not the weight scales** |
| pregrasp `_solve_ik` / `_min_jerk_plan` / inverse dynamics + tracking cost | per-phase reference generation, replanned at phase entry |
| `fer_pick_place_ros2_control_scene.xml` (bringup) | ROS plant already has cube + target + gripper actuator |
| `HydraxPlannerOutput.gripper_command`, phase string | contract fields already in place |
| frozen pregrasp solver config | starting point for `pick_place.yaml` solver section |
| V-A5 sweep machinery (`--q0_noise`, driver, baseline-relative judging) | P-A3 robustness grid |
| `initial_q:=random` launch argument | P-B2 sim sessions from varied starts |
| FeedbackMPPI (solver, gains, ESS/health readouts) | unchanged — the task, not the solver, is what grows |
