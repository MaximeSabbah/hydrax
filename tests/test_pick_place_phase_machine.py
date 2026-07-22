"""P-A1 — the pick-and-place phase machine and reference timeline.

Pure CPU checks of the task-layer logic added for the pick-and-place
(doc/pick_place_plan.md): the chained min-jerk timeline, the IK'd phase
goals, the yaml loader, and the event-gated plan clock. The controller
itself is the certified pregrasp machinery and is gated by P-A2.
"""

import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from hydrax.configs import PICK_PLACE_CONFIG_YAML, load_pick_place_config
from hydrax.tasks.panda_pick_place import (
    PandaPickPlace,
    PandaPickPlaceOptions,
    Phase,
    PickPlacePhaseMachine,
)

MOTION_PHASES = [p for p in Phase if p not in (Phase.CLOSE, Phase.OPEN, Phase.DONE)]


def _pose_with_error(
    task,
    phase: Phase,
    *,
    position_error=(0.0, 0.0, 0.0),
    orientation_error=0.0,
):
    """Measured EE pose offset from one phase goal by known gate errors."""
    goal_phase = min(phase, Phase.RETREAT)
    position = np.asarray(task.phase_goal_ee_positions[goal_phase]) + np.asarray(
        position_error
    )
    c, s = np.cos(orientation_error), np.sin(orientation_error)
    delta = np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
    rotation = np.asarray(task.phase_goal_ee_rotations[goal_phase]) @ delta
    return position, rotation


def _update(
    pm,
    task,
    q,
    v,
    *,
    position_error=(0.0, 0.0, 0.0),
    orientation_error=0.0,
    linear_velocity=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
):
    position, rotation = _pose_with_error(
        task,
        pm.phase,
        position_error=position_error,
        orientation_error=orientation_error,
    )
    return pm.update(
        q,
        v,
        position,
        rotation,
        np.asarray(linear_velocity),
        np.asarray(angular_velocity),
    )


def _snapshot(
    pm,
    task,
    q,
    v,
    *,
    position_error=(0.0, 0.0, 0.0),
    orientation_error=0.0,
    linear_velocity=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
):
    position, rotation = _pose_with_error(
        task,
        pm.phase,
        position_error=position_error,
        orientation_error=orientation_error,
    )
    return pm.diagnostics_snapshot(
        q,
        v,
        position,
        rotation,
        np.asarray(linear_velocity),
        np.asarray(angular_velocity),
    )


def _fk_pose(task, q):
    data = mujoco.MjData(task.mj_model)
    site = mujoco.mj_name2id(
        task.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripper"
    )
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(task.mj_model, data)
    return (
        data.site_xpos[site].copy(),
        data.site_xmat[site].reshape(3, 3).copy(),
    )


@pytest.fixture(scope="module")
def task():
    options, _ = load_pick_place_config()
    return PandaPickPlace(options=options)


def test_yaml_loads_into_the_typed_schema():
    options, config = load_pick_place_config()
    assert config.temperature == 0.007
    assert config.spline_type == "linear"
    assert config.plan_horizon == 0.32
    assert options.cube_pos == [0.5, 0.0, 0.105]
    assert options.target_pos == [0.65, 0.0, 0.105]
    # Phase acceptance is a fixed task-success contract, not OCP tuning.
    assert not hasattr(options, "transition_ee_position_tolerance")
    assert not hasattr(options, "transition_q_tolerance")


@pytest.mark.parametrize(
    "key",
    [
        "transition_q_tolerance",
        "precision_q_tolerance",
        "transition_v_tolerance",
        "transition_ee_position_tolerance",
        "transition_consecutive_cycles",
    ],
)
def test_yaml_rejects_phase_acceptance_knobs(tmp_path, key):
    raw = yaml.safe_load(Path(PICK_PLACE_CONFIG_YAML).read_text())
    raw["plan"][key] = 0.050
    path = tmp_path / "pick_place_with_gate_knob.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="unknown pick_place yaml keys"):
        load_pick_place_config(str(path))


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
    for _ in range(int(task.duration / task.dt) + 100):
        goal = task.phase_goal_q[min(pm.phase, Phase.RETREAT)]
        _update(pm, task, goal, np.zeros(7))
        if not seen or seen[-1] != pm.phase:
            seen.append(pm.phase)
    assert seen == list(Phase)[list(Phase).index(seen[0]) :]
    assert seen[-1] == Phase.DONE


def test_clock_stalls_at_a_motion_boundary_until_converged(task):
    pm = PickPlacePhaseMachine(task)
    boundary = task.segment_end_times[Phase.PREGRASP]
    q_far = task.phase_goal_q[Phase.PREGRASP] + 0.2
    for _ in range(2 * int(boundary / task.dt)):
        _update(
            pm,
            task,
            q_far,
            np.zeros(7),
            position_error=(0.04, 0.0, 0.0),
        )
    assert pm.phase == Phase.PREGRASP
    assert pm.plan_time == pytest.approx(boundary)
    # an ordinary (non-dwell-entry) stall is not a precision window
    assert not pm.precision_hold

    snapshot = _snapshot(
        pm, task, task.phase_goal_q[Phase.PREGRASP], np.zeros(7)
    )
    required = snapshot.consecutive_required_cycles
    assert required == 5
    # The fixed pose/stillness contract must remain true for the whole
    # acceptance window before the clock crosses into DESCEND.
    for _ in range(required):
        _update(
            pm,
            task,
            task.phase_goal_q[Phase.PREGRASP],
            np.zeros(7),
        )
    assert pm.phase == Phase.DESCEND
    # the clock must stay a python float (an np.float64 leaking into
    # state.time changes the traced dtype and recompiles the solver)
    assert type(pm.plan_time) is float


@pytest.mark.parametrize("phase", [Phase.DESCEND, Phase.PLACE])
def test_dwell_entries_share_the_fixed_action_pose_contract(task, phase):
    # A 15 mm EE error is inside the ordinary arrival envelope but outside
    # the same fixed 10 mm action-pose contract before CLOSE and OPEN.
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[phase])
    q = task.phase_goal_q[phase]
    _update(pm, task, q, np.zeros(7), position_error=(0.015, 0.0, 0.0))
    snapshot = _snapshot(
        pm, task, q, np.zeros(7), position_error=(0.015, 0.0, 0.0)
    )

    assert pm.phase == phase
    assert snapshot.ee_position_tolerance_m == pytest.approx(0.010)
    assert snapshot.transition_blockers == ("ee_position",)
    assert pm.precision_hold


def test_redundant_real_q5_offset_does_not_block_task_space_gate(task):
    """Regression for the 2026-07-21 real PREGRASP deadlock."""
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP]).copy()
    q[4] += 0.0528  # real run: just outside the obsolete 0.05 rad q gate
    position, rotation = _fk_pose(task, q)

    zero_twist = np.zeros(3)
    snapshot = pm.diagnostics_snapshot(
        q,
        np.zeros(7),
        position,
        rotation,
        zero_twist,
        zero_twist,
    )
    assert snapshot.q_error_max_rad == pytest.approx(0.0528)
    assert snapshot.ee_position_error_norm_m < 0.004
    assert snapshot.ee_orientation_error_rad == pytest.approx(0.0528)
    assert snapshot.transition_blockers == ()

    required = snapshot.consecutive_required_cycles
    for _ in range(required - 1):
        pm.update(q, np.zeros(7), position, rotation, zero_twist, zero_twist)
        assert pm.phase == Phase.PREGRASP
    pm.update(q, np.zeros(7), position, rotation, zero_twist, zero_twist)
    assert pm.phase == Phase.DESCEND


def test_gate_streak_resets_after_one_bad_orientation_sample(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP])
    v = np.zeros(7)

    for _ in range(2):
        _update(pm, task, q, v)
    snapshot = _snapshot(pm, task, q, v)
    assert snapshot.consecutive_eligible_cycles == 2

    _update(pm, task, q, v, orientation_error=0.11)
    blocked = _snapshot(pm, task, q, v, orientation_error=0.11)
    assert blocked.transition_blockers == ("ee_orientation",)
    assert blocked.consecutive_eligible_cycles == 0

    required = blocked.consecutive_required_cycles
    for _ in range(required - 1):
        _update(pm, task, q, v)
        assert pm.phase == Phase.PREGRASP
    _update(pm, task, q, v)
    assert pm.phase == Phase.DESCEND


@pytest.mark.parametrize(
    ("sample", "blocker"),
    [
        ({"position_error": (np.nan, 0.0, 0.0)}, "ee_position"),
        ({"orientation_error": np.nan}, "ee_orientation"),
        ({"linear_velocity": (np.nan, 0.0, 0.0)}, "ee_linear_speed"),
        ({"angular_velocity": (np.inf, 0.0, 0.0)}, "ee_angular_speed"),
    ],
)
def test_nonfinite_ee_samples_fail_closed_and_reset_the_streak(
    task, sample, blocker
):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP])
    v = np.zeros(7)

    _update(pm, task, q, v)
    _update(pm, task, q, v)
    assert _snapshot(pm, task, q, v).consecutive_eligible_cycles == 2

    _update(pm, task, q, v, **sample)
    snapshot = _snapshot(pm, task, q, v, **sample)
    assert pm.phase == Phase.PREGRASP
    assert snapshot.transition_blockers == (blocker,)
    assert snapshot.consecutive_eligible_cycles == 0


def test_joint_position_and_velocity_are_diagnostic_only(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP]).copy()
    q[4] += 0.36
    v = np.zeros(7)
    v[5] = -1.2

    snapshot = _snapshot(pm, task, q, v)
    required = snapshot.consecutive_required_cycles
    for _ in range(required):
        _update(pm, task, q, v)

    assert snapshot.q_error_max_rad == pytest.approx(0.36)
    assert snapshot.velocity_abs_max_rad_s == pytest.approx(1.2)
    assert snapshot.transition_blockers == ()
    assert pm.phase == Phase.DESCEND


def test_diagnostics_identifies_the_pregrasp_ee_position_blocker(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP]).copy()
    q[3] += 0.06
    v = np.zeros(7)

    state_before = (pm.plan_time, pm.precision_hold, pm.gripper_closed)
    snapshot = _snapshot(
        pm,
        task,
        q,
        v,
        position_error=(0.04, 0.0, 0.0),
    )

    assert snapshot.phase == "PREGRASP"
    assert snapshot.next_phase == "DESCEND"
    assert snapshot.gate_type == "task_space"
    assert snapshot.at_boundary
    assert snapshot.time_to_boundary_sec == 0.0
    assert snapshot.transition_status == "blocked_ee_position"
    assert snapshot.transition_blockers == ("ee_position",)
    assert snapshot.transition_blocked
    assert snapshot.q_error_max_rad == pytest.approx(0.06)
    assert snapshot.q_error_joint_index == 3
    assert snapshot.q_error_signed_by_joint_rad[3] == pytest.approx(0.06)
    assert snapshot.velocity_abs_max_rad_s == 0.0
    assert snapshot.ee_position_error_norm_m == pytest.approx(0.04)
    assert snapshot.ee_position_tolerance_m == pytest.approx(0.03)
    assert snapshot.ee_position_ok is False
    assert snapshot.ee_orientation_error_rad < 1e-6
    assert snapshot.ee_orientation_ok is True
    assert snapshot.ee_linear_speed_m_s == 0.0
    assert snapshot.ee_linear_speed_tolerance_m_s == pytest.approx(0.060)
    assert snapshot.ee_linear_speed_ok is True
    assert snapshot.ee_angular_speed_rad_s == 0.0
    assert snapshot.ee_angular_speed_tolerance_rad_s == pytest.approx(0.20)
    assert snapshot.ee_angular_speed_ok is True
    assert snapshot.consecutive_eligible_cycles == 0
    assert snapshot.consecutive_required_cycles == 5
    assert snapshot.consecutive_ok is False
    assert not snapshot.precision_hold
    assert snapshot.gripper_command == "open"

    # A snapshot is observational: it neither advances the event clock nor
    # changes the controller/gripper schedule. It must also serialize with
    # strict JSON (no NumPy scalar or NaN leakage into ROS diagnostics).
    assert (pm.plan_time, pm.precision_hold, pm.gripper_closed) == state_before
    payload = json.loads(json.dumps(asdict(snapshot), allow_nan=False))
    assert payload["q_error_joint_index"] == 3
    assert payload["q_error_signed_by_joint_rad"][3] == pytest.approx(0.06)
    assert payload["transition_blockers"] == ["ee_position"]


@pytest.mark.parametrize(
    ("linear_velocity", "angular_velocity", "blocker"),
    [
        ((0.061, 0.0, 0.0), (0.0, 0.0, 0.0), "ee_linear_speed"),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.201), "ee_angular_speed"),
    ],
)
def test_diagnostics_identifies_cartesian_speed_blockers(
    task, linear_velocity, angular_velocity, blocker
):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.PREGRASP])
    q = np.asarray(task.phase_goal_q[Phase.PREGRASP]).copy()
    snapshot = _snapshot(
        pm,
        task,
        q,
        np.zeros(7),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )

    assert snapshot.transition_status == f"blocked_{blocker}"
    assert snapshot.transition_blockers == (blocker,)
    assert snapshot.gate_type == "task_space"
    assert snapshot.transition_blocked
    assert snapshot.q_error_max_rad == pytest.approx(0.0)
    if blocker == "ee_linear_speed":
        assert snapshot.ee_linear_speed_ok is False
        assert snapshot.ee_angular_speed_ok is True
    else:
        assert snapshot.ee_linear_speed_ok is True
        assert snapshot.ee_angular_speed_ok is False


def test_diagnostics_reports_the_tighter_descend_precision_gate(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.DESCEND])
    q = np.asarray(task.phase_goal_q[Phase.DESCEND]).copy()
    q[2] += 0.03

    # One update at the boundary records the precision window while the
    # remaining 15 -> 10 mm EE leg is still blocked.
    _update(
        pm,
        task,
        q,
        np.zeros(7),
        position_error=(0.015, 0.0, 0.0),
    )
    snapshot = _snapshot(
        pm,
        task,
        q,
        np.zeros(7),
        position_error=(0.015, 0.0, 0.0),
    )

    assert snapshot.phase == "DESCEND"
    assert snapshot.next_phase == "CLOSE"
    assert snapshot.gate_type == "task_space"
    assert snapshot.at_boundary
    assert snapshot.transition_status == "blocked_ee_position"
    assert snapshot.q_error_max_rad == pytest.approx(0.03)
    assert snapshot.q_error_joint_index == 2
    assert snapshot.ee_position_error_norm_m == pytest.approx(0.015)
    assert snapshot.ee_position_tolerance_m == pytest.approx(0.010)
    assert snapshot.ee_position_ok is False
    assert snapshot.precision_hold
    assert snapshot.gripper_command == "open"


def test_diagnostics_marks_dwell_boundaries_as_time_only(task):
    pm = PickPlacePhaseMachine(task)
    pm.plan_time = float(task.segment_end_times[Phase.CLOSE])
    q_far = np.asarray(task.phase_goal_q[Phase.CLOSE]) + 0.5
    v_fast = np.full(7, 1.0)

    snapshot = _snapshot(pm, task, q_far, v_fast)

    assert snapshot.phase == "CLOSE"
    assert snapshot.next_phase == "LIFT"
    assert snapshot.gate_type == "time_only"
    assert snapshot.at_boundary
    assert snapshot.transition_status == "time_only"
    assert not snapshot.transition_blocked
    assert snapshot.ee_position_tolerance_m is None
    assert snapshot.ee_orientation_tolerance_rad is None
    assert snapshot.ee_linear_speed_tolerance_m_s is None
    assert snapshot.ee_angular_speed_tolerance_rad_s is None
    assert snapshot.consecutive_required_cycles is None
    assert snapshot.precision_hold
    assert snapshot.gripper_command == "close"


def test_gripper_schedule_settles_then_actuates(task):
    # Dwells are settle-then-actuate: the reference stays stationary for
    # dwell_settle_sec before the gripper moves; OPEN keeps holding the
    # cube through its settle, then releases.
    pm = PickPlacePhaseMachine(task)
    settle = task.options.dwell_settle_sec
    log = []
    for _ in range(int(task.duration / task.dt) + 100):
        goal = task.phase_goal_q[min(pm.phase, Phase.RETREAT)]
        _update(pm, task, goal, np.zeros(7))
        log.append((pm.phase, pm.plan_time, pm.gripper_closed, pm.precision_hold))

    def in_phase(phase):
        return [row for row in log if row[0] == phase]

    close_start = task.segment_end_times[Phase.DESCEND]
    for phase, t, closed, hold in in_phase(Phase.CLOSE):
        assert hold  # the whole dwell is a reported precision window
        assert closed == (t - close_start >= settle)
    open_start = task.segment_end_times[Phase.PLACE]
    for phase, t, closed, hold in in_phase(Phase.OPEN):
        assert hold
        assert closed == (t - open_start < settle)
    for phase in (Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
        assert all(row[2] for row in in_phase(phase))
    for phase in (Phase.PREGRASP, Phase.DESCEND, Phase.RETREAT):
        assert all(not row[2] for row in in_phase(phase))
    # Only the action-pose waits and their dwells carry this diagnostic marker.
    assert all(not row[3] for row in in_phase(Phase.PREGRASP))
    assert all(not row[3] for row in in_phase(Phase.LIFT))
    assert all(not row[3] for row in in_phase(Phase.TRANSPORT))
    assert all(not row[3] for row in in_phase(Phase.RETREAT))
