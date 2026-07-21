"""The single-source Panda model layout (2026-07-09 consolidation).

One robot description (models/panda/panda.xml) and one environment
(models/panda/scene.xml). The contact-free 7-DoF planning variant is
derived from panda.xml at task-load time; these tests pin the
derivation's invariants and its parity with the plant, replacing the
committed pregrasp.xml it was proven structurally identical to.
"""

import mujoco
import numpy as np
import pytest

from hydrax import ROOT
from hydrax.tasks.panda_pregrasp import PandaPregrasp, PandaPregraspOptions

HOME_Q = PandaPregraspOptions.start_q


@pytest.fixture(scope="module")
def planning():
    return PandaPregrasp._derive_arm_planning_model(HOME_Q)


@pytest.fixture(scope="module")
def plant():
    return mujoco.MjModel.from_xml_path(ROOT + "/models/panda/scene.xml")


def test_planning_model_is_the_contact_free_arm(planning) -> None:
    assert planning.nq == planning.nv == planning.nu == 7
    assert planning.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
    assert planning.opt.timestep == 0.04  # one physics step per 25 Hz step
    assert planning.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # torque motors at the Franka limits
    expected = [87.0] * 4 + [12.0] * 3
    assert np.allclose(planning.actuator_ctrlrange[:, 1], expected)
    # Menagerie-standard joint dynamics
    assert np.allclose(planning.dof_armature, 0.1)
    assert np.allclose(planning.dof_damping, 1.0)
    assert np.allclose(planning.key("home").qpos, HOME_Q)


def test_planning_arm_is_identical_to_the_plant_arm(planning, plant) -> None:
    for i in range(7):
        assert plant.joint(i).name == planning.joint(i).name == f"joint{i+1}"
    assert np.allclose(plant.jnt_range[:7], planning.jnt_range)
    assert np.allclose(plant.dof_damping[:7], planning.dof_damping)
    assert np.allclose(plant.dof_armature[:7], planning.dof_armature)
    assert np.allclose(plant.actuator_ctrlrange[:7], planning.actuator_ctrlrange)
    for i in range(8):
        assert np.isclose(
            plant.body(f"link{i}").mass, planning.body(f"link{i}").mass
        )
    # frozen fingers keep their inertia: the planned hand subtree weighs
    # the same as the plant's articulated one
    assert np.isclose(
        plant.body("hand").subtreemass, planning.body("hand").subtreemass
    )
    assert np.allclose(
        plant.site("gripper").pos, planning.site("gripper").pos
    )


def test_plant_scene_grasps_and_holds(plant) -> None:
    assert plant.nq == 16 and plant.nu == 8  # arm + fingers + cube
    assert plant.opt.timestep == 0.001  # the 1 kHz deployment rate
    assert plant.body("target").mocapid[0] >= 0
    object_body_id = mujoco.mj_name2id(
        plant, mujoco.mjtObj.mjOBJ_BODY, "object"
    )
    assert plant.body_gravcomp[object_body_id] == pytest.approx(1.0)
    assert plant.body("object").pos == pytest.approx([0.5, 0.0, 0.105])
    assert plant.body("target").pos == pytest.approx([0.65, 0.0, 0.105])
    data = mujoco.MjData(plant)
    mujoco.mj_resetDataKeyframe(plant, data, 0)
    for _ in range(1000):
        mujoco.mj_step(plant, data)
    # Gravity compensation holds both the arm and the deliberately floating
    # dynamic cube at their keyframe poses without a simulated table.
    assert np.abs(data.qpos[:7] - plant.key("home").qpos[:7]).max() < 1e-3
    assert np.abs(data.qpos[9:12] - plant.key("home").qpos[9:12]).max() < 1e-3
    assert data.qpos[9:12] == pytest.approx([0.5, 0.0, 0.105], abs=1e-6)
    # the gripper actuates: ctrl[7] is the finger width servo
    data.ctrl[7] = 0.0
    for _ in range(1000):
        mujoco.mj_step(plant, data)
    assert data.qpos[7] + data.qpos[8] < 0.005  # closed
