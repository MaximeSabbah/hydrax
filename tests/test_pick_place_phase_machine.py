"""P-A1 — the pick-and-place phase machine and reference timeline.

Pure CPU checks of the task-layer logic added for the pick-and-place
(doc/pick_place_plan.md): the chained min-jerk timeline, the IK'd phase
goals, the yaml loader, and the event-gated plan clock. The controller
itself is the certified pregrasp machinery and is gated by P-A2.
"""

import json
from dataclasses import asdict

import mujoco
import numpy as np
import pytest

from hydrax.configs import load_pick_place_config
from hydrax.tasks.panda_pick_place import (
    PandaPickPlace,
    PandaPickPlaceOptions,
    Phase,
    PickPlacePhaseMachine,
)

MOTION_PHASES = [p for p in Phase if p not in (Phase.CLOSE, Phase.OPEN, Phase.DONE)]


@pytest.fixture(scope="module")
def task():
    options, _ = load_pick_place_config()
    return PandaPickPlace(options=options)


def test_yaml_loads_into_the_typed_schema():
    options, config = load_pick_place_config()
    assert config.temperature == 0.007
    assert config.spline_type == "linear"
    assert config.plan_horizon == 0.32
    assert options.transition_q_tolerance == 0.05
    assert options.cube_pos == [0.5, 0.0, 0.105]
    assert options.target_pos == [0.65, 0.0, 0.105]


def test_timeline_chains_the_phase_goals(task):
    ref = np.asarray(task.reference_qpos)
    vel = np.asarray(task.reference_qvel)
    fps = task.reference_fps
    assert np.allclose(ref[0], task.start_q, atol=1e-6)
    for phase in Phase:
        if phase == Phase.DONE:
            continue
        i = int(round(task.segment_end_times[phase] * fps))
        # each segment ends exactly on its goal, at rest
        assert np.allclose(ref[i], task.phase_goal_q[phase], atol=1e-5), phase
        assert np.abs(vel[i]).max() < 1e-4, phase
    # dwell segments are constant
    for phase in (Phase.CLOSE, Phase.OPEN):
        i0 = int(round(task.segment_end_times[phase - 1] * fps))
        i1 = int(round(task.segment_end_times[phase] * fps))
        assert np.ptp(ref[i0 : i1 + 1], axis=0).max() < 1e-6, phase


def test_timeline_respects_the_velocity_cap(task):
    vel = np.abs(np.asarray(task.reference_qvel))
    cap = task.options.max_velocity_fraction * np.asarray(task.options.vel_max)
    assert np.all(vel <= cap * (1.0 + 1e-6))


def test_phase_goals_reach_the_configured_poses(task):
    data = mujoco.MjData(task.mj_model)
    sid = mujoco.mj_name2id(task.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")

    def ee(phase):
        data.qpos[:] = task.phase_goal_q[phase]
        mujoco.mj_forward(task.mj_model, data)
        return data.site_xpos[sid].copy()

    cube = np.asarray(task.options.cube_pos)
    target = np.asarray(task.options.target_pos)
    assert np.linalg.norm(ee(Phase.DESCEND) - cube) < 1e-3
    assert np.linalg.norm(ee(Phase.PREGRASP) - (cube + task.options.pregrasp_offset)) < 1e-3
    assert np.linalg.norm(ee(Phase.PLACE) - (target + task.options.place_offset)) < 1e-3
    assert np.linalg.norm(ee(Phase.RETREAT) - (target + task.options.retreat_offset)) < 1e-3


def test_phase_goals_follow_the_measured_cube_pose():
    shifted = PandaPickPlace(
        options=PandaPickPlaceOptions(cube_pos=(0.52, -0.03, 0.105))
    )
    data = mujoco.MjData(shifted.mj_model)
    sid = mujoco.mj_name2id(shifted.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    data.qpos[:] = shifted.phase_goal_q[Phase.DESCEND]
    mujoco.mj_forward(shifted.mj_model, data)
    assert np.linalg.norm(data.site_xpos[sid] - [0.52, -0.03, 0.105]) < 1e-3


def test_clock_walks_the_sequence_when_converged(task):
    pm = PickPlacePhaseMachine(task)
    seen = []
    for _ in range(int(task.duration / task.dt) + 10):
        goal = task.phase_goal_q[min(pm.phase, Phase.RETREAT)]
        pm.update(goal, np.zeros(7))
        if not seen or seen[-1] != pm.phase:
            seen.append(pm.phase)
    assert seen == list(Phase)[list(Phase).index(seen[0]) :]
    assert seen[-1] == Phase.DONE


def test_clock_stalls_at_a_motion_boundary_until_converged(task):
    pm = PickPlacePhaseMachine(task)
    boundary = task.segment_end_times[Phase.PREGRASP]
    q_far = task.phase_goal_q[Phase.PREGRASP] + 0.2
    for _ in range(2 * int(boundary / task.dt)):
        pm.update(q_far, np.zeros(7))
    assert pm.phase == Phase.PREGRASP
    assert pm.plan_time == pytest.approx(boundary)
    # an ordinary (non-dwell-entry) stall stays on exact_feedback
    assert not pm.precision_hold
    # high velocity also blocks the crossing
    pm.update(task.phase_goal_q[Phase.PREGRASP], np.full(7, 1.0))
    assert pm.phase == Phase.PREGRASP
    # converged: the clock crosses into DESCEND
    pm.update(task.phase_goal_q[Phase.PREGRASP], np.zeros(7))
    pm.update(task.phase_goal_q[Phase.PREGRASP], np.zeros(7))
    assert pm.phase == Phase.DESCEND
    # the clock must stay a python float (an np.float64 leaking into
    # state.time changes the traced dtype and recompiles the solver)
    assert type(pm.plan_time) is float


def test_dwell_entries_use_the_precision_tolerance(task):
    # 0.03 rad crosses an ordinary boundary (tol 0.05) but must stall the
    # DESCEND->CLOSE entry (precision tol 0.02): that crossing IS the grasp
    pm = PickPlacePhaseMachine(task)
    q_off = 0.03
    for _ in range(2 * int(task.segment_end_times[Phase.DESCEND] / task.dt)):
        goal = task.phase_goal_q[min(pm.phase, Phase.RETREAT)]
        offset = q_off if pm.phase == Phase.DESCEND else 0.0
        pm.update(goal + offset, np.zeros(7))
    assert pm.phase == Phase.DESCEND  # stalled at the precision gate
    # the wait at a dwell entry IS a precision window: the impedance must
    # engage here or a model-mismatch sag > the gate deadlocks the clock
    # (P-A3 mass 1.1)
    assert pm.precision_hold
    pm.update(task.phase_goal_q[Phase.DESCEND] + 0.01, np.zeros(7))
    pm.update(task.phase_goal_q[Phase.DESCEND] + 0.01, np.zeros(7))
    assert pm.phase == Phase.CLOSE


def test_diagnostics_identifies_the_pregrasp_joint_position_blocker(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP]).copy()
    q[3] += 0.06
    v = np.zeros(7)

    state_before = (pm.plan_time, pm.precision_hold, pm.gripper_closed)
    snapshot = pm.diagnostics_snapshot(q, v)

    assert snapshot.phase == "PREGRASP"
    assert snapshot.next_phase == "DESCEND"
    assert snapshot.gate_type == "state"
    assert snapshot.at_boundary
    assert snapshot.time_to_boundary_sec == 0.0
    assert snapshot.transition_status == "blocked_q"
    assert snapshot.transition_blocked
    assert snapshot.q_error_max_rad == pytest.approx(0.06)
    assert snapshot.q_error_joint_index == 3
    assert snapshot.q_error_signed_by_joint_rad[3] == pytest.approx(0.06)
    assert snapshot.q_tolerance_rad == pytest.approx(0.05)
    assert snapshot.q_ok is False
    assert snapshot.velocity_abs_max_rad_s == 0.0
    assert snapshot.velocity_ok is True
    assert not snapshot.precision_hold
    assert snapshot.gripper_command == "open"

    # A snapshot is observational: it neither advances the event clock nor
    # changes the controller/gripper schedule. It must also serialize with
    # strict JSON (no NumPy scalar or NaN leakage into ROS diagnostics).
    assert (pm.plan_time, pm.precision_hold, pm.gripper_closed) == state_before
    payload = json.loads(json.dumps(asdict(snapshot), allow_nan=False))
    assert payload["q_error_joint_index"] == 3
    assert payload["q_error_signed_by_joint_rad"][3] == pytest.approx(0.06)


def test_diagnostics_identifies_the_pregrasp_velocity_blocker(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP]).copy()
    v = np.zeros(7)
    v[5] = -0.16

    snapshot = pm.diagnostics_snapshot(q, v)

    assert snapshot.transition_status == "blocked_v"
    assert snapshot.gate_type == "state"
    assert snapshot.transition_blocked
    assert snapshot.q_error_max_rad == pytest.approx(0.0)
    assert snapshot.q_ok is True
    assert snapshot.velocity_by_joint_rad_s[5] == pytest.approx(-0.16)
    assert snapshot.velocity_abs_max_rad_s == pytest.approx(0.16)
    assert snapshot.velocity_joint_index == 5
    assert snapshot.velocity_tolerance_rad_s == pytest.approx(0.15)
    assert snapshot.velocity_ok is False


def test_diagnostics_reports_the_tighter_descend_precision_gate(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.DESCEND])
    q = np.asarray(task.phase_goal_q[Phase.DESCEND]).copy()
    q[2] += 0.03

    # One update at the boundary records the precision-hold schedule used to
    # close the remaining 0.03 -> 0.02 rad leg on the real controller.
    pm.update(q, np.zeros(7))
    snapshot = pm.diagnostics_snapshot(q, np.zeros(7))

    assert snapshot.phase == "DESCEND"
    assert snapshot.next_phase == "CLOSE"
    assert snapshot.gate_type == "state"
    assert snapshot.at_boundary
    assert snapshot.transition_status == "blocked_q"
    assert snapshot.q_error_max_rad == pytest.approx(0.03)
    assert snapshot.q_error_joint_index == 2
    assert snapshot.q_tolerance_rad == pytest.approx(0.02)
    assert snapshot.q_ok is False
    assert snapshot.velocity_ok is True
    assert snapshot.precision_hold
    assert snapshot.gripper_command == "open"


def test_diagnostics_marks_dwell_boundaries_as_time_only(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.CLOSE])
    q_far = np.asarray(task.phase_goal_q[Phase.CLOSE]) + 0.5
    v_fast = np.full(7, 1.0)

    snapshot = pm.diagnostics_snapshot(q_far, v_fast)

    assert snapshot.phase == "CLOSE"
    assert snapshot.next_phase == "LIFT"
    assert snapshot.gate_type == "time_only"
    assert snapshot.at_boundary
    assert snapshot.transition_status == "time_only"
    assert not snapshot.transition_blocked
    assert snapshot.q_tolerance_rad is None
    assert snapshot.velocity_tolerance_rad_s is None
    assert snapshot.q_ok is None
    assert snapshot.velocity_ok is None
    assert snapshot.precision_hold
    assert snapshot.gripper_command == "close"


def test_gripper_schedule_settles_then_actuates(task):
    # dwells are settle-then-actuate: the stiff impedance pins the arm for
    # dwell_settle_sec before the gripper moves (P2 law schedule); OPEN
    # keeps holding the cube through its settle, then releases
    pm = PickPlacePhaseMachine(task)
    settle = task.options.dwell_settle_sec
    log = []
    for _ in range(int(task.duration / task.dt) + 10):
        goal = task.phase_goal_q[min(pm.phase, Phase.RETREAT)]
        pm.update(goal, np.zeros(7))
        log.append((pm.phase, pm.plan_time, pm.gripper_closed, pm.precision_hold))

    def in_phase(phase):
        return [row for row in log if row[0] == phase]

    close_start = task.segment_end_times[Phase.DESCEND]
    for phase, t, closed, hold in in_phase(Phase.CLOSE):
        assert hold  # impedance throughout the dwell
        assert closed == (t - close_start >= settle)
    open_start = task.segment_end_times[Phase.PLACE]
    for phase, t, closed, hold in in_phase(Phase.OPEN):
        assert hold
        assert closed == (t - open_start < settle)
    for phase in (Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
        assert all(row[2] and not row[3] for row in in_phase(phase))
    for phase in (Phase.PREGRASP, Phase.DESCEND, Phase.RETREAT):
        assert all(not row[2] and not row[3] for row in in_phase(phase))
