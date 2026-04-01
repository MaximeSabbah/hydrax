import jax.numpy as jnp
import mujoco

from hydrax.algs import MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pick_and_place import PickAndPlace

"""
Pick-and-place with the Franka Emika Panda using MPPI.

The robot picks up a green box from the table and places it at a red
translucent target. Double-click the target mocap body, then drag it
with [ctrl + left click] to change the goal position.
"""

# Define the task (cost and dynamics)
task = PickAndPlace()

# MPPI controller: moderate samples, low noise for position-controlled robot
ctrl = MPPI(
    task,
    num_samples=128,
    noise_level=0.1,
    temperature=0.1,
    num_randomizations=2,
    plan_horizon=0.5,
    spline_type="linear",
    num_knots=6,
    iterations=1,
)

# Set up the simulation from the "home" keyframe
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

# Warm-start MPPI mean at the home control configuration
home_ctrl = jnp.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853, 0.04])
initial_knots = jnp.tile(home_ctrl, (ctrl.num_knots, 1))

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=25,
    initial_knots=initial_knots,
    fixed_camera_id=None,
    show_traces=True,
    max_traces=3,
    trace_color=[0.2, 0.8, 0.2, 0.15],
)
