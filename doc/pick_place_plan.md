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

**Per-phase references (the only new controller-facing mechanics).** On
phase entry the task plans a min-jerk joint segment from the *measured*
posture to the phase's IK goal, with the phase's velocity cap — the
pregrasp generator, replanned per phase (CPU, milliseconds, once per
phase, off the 25 Hz hot path). Two properties fall out for free:
- **no feedforward yank**: each segment starts at the measured pose, so
  even the stiff impedance law is safe at phase entry (the V-A5
  feedforward failure mode was a reference *not* starting at the robot);
- **no recompilation**: the reference is data (fixed-shape, padded
  time-indexed arrays, the existing `state.time` lookup); phase entry
  swaps array contents, never code. This must be asserted by a P-A2
  timing gate (no solve-time spike at transitions), not assumed.

**Cost and solver: unchanged.** The pregrasp tracking cost (saturated
q/v/u errors, weights 1.0/0.1/0.01) against the active segment; frozen
solver config (T=0.007, H=0.32 s, it=3, 1024 samples, gain batch 128,
linear spline, σ=0.03·τ_max) as the starting point, re-gated.

**Deployment law per phase — a P2 measurement, one existing switch.**
Default is `exact_feedback` end-to-end (what the robot ran). The known
watch-item: at stationary dwells (CLOSE/OPEN) the exact_feedback hold
wanders 1–2 cm, which is too much at the grasp instant. Two remedies,
both already certified and both inside the existing bridge contract:
1. **impedance dwell blend**: during dwells the adapter publishes the
   fixed impedance anchored on the (stationary) reference — sub-mm hold,
   K and `initial_state` are per-cycle planner outputs so no bridge
   change;
2. plain `planner_mode: feedforward` for the whole sequence — viable
   because per-phase replanning removes the reference-start mismatch
   (see above), at the cost of sidelining the F-MPPI gains.
P-A2 measures the dwell precision in exact_feedback first; the remedy
(likely 1) is adopted only if the measurement demands it, and the choice
is recorded here.

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
K/anchor contract per phase, including the impedance dwell blend if
adopted; gripper action client unit-tested against a fake action server.

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

**P0 status (2026-07-09): COMPLETE, including the model consolidation**
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
| P2 | P-A3; dwell-precision remedy decided (recorded here) and re-gated; config frozen |
| P3 | P-B1, P-B2, P-B3 |
| P4 | P-B4 stages 1→4 |

## Risks

- **Dwell precision at CLOSE/OPEN** (the exact_feedback stationary-hold
  wander, 1–2 cm known): two certified remedies inside the existing
  contract (impedance dwell blend / feedforward mode); measured and
  decided in P-A2/P2 — a choice between working options, not an open
  problem.
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
