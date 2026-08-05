# Panda Feedback-MPPI vs Crocoddyl gains

This benchmark compares the feedback gain computed by the Hydrax
`FeedbackMPPI` controller on the Panda pregrasp reach with the finite-horizon
Riccati gain computed by Crocoddyl.

It deliberately separates two questions:

1. **Gain-algorithm comparison:** use the same MJX dynamics, the same Hydrax
   bounded tracking cost, the same nominal MPPI control trajectory, the same
   0.04 s discretization and the same eight-step horizon. Only the gain
   computation changes. This is what the two scripts currently implement.
2. **Model-stack comparison:** compare a native Pinocchio/Crocoddyl controller
   against the MuJoCo/MJX controller. This is a different experiment because
   model, integration and friction conventions no longer match.

## Why the benchmark is split into two processes

The Hydrax branch uses a Python 3.12 JAX/MJX environment. The Agimus Humble
control container uses the Crocoddyl/Pinocchio stack under its ROS Python
environment. An NPZ boundary keeps both existing environments intact.

## 1. Export the local Hydrax OCP

From the `feedback-mppi-panda` Hydrax environment:

```bash
python validation/export_panda_feedback_lq.py \
  --fractions 0,0.25,0.5,0.75,1 \
  --output /tmp/panda_feedback_local_lq.npz
```

For the exact all-rollout Feedback-MPPI estimator rather than the configured
256-sample gain subset:

```bash
python validation/export_panda_feedback_lq.py \
  --num-gain-samples 1024 \
  --output /tmp/panda_feedback_local_lq_all_samples.npz
```

At each requested reach time, the exporter:

- initializes the same Panda pregrasp task and Feedback-MPPI controller;
- warm-starts the control knots from the inverse-dynamics reference torques;
- runs the configured MPPI update;
- records `K_feedback_mppi = du*/dx0`;
- rolls out the updated MPPI mean;
- differentiates every discrete MJX step to obtain `A_t` and `B_t`;
- differentiates Hydrax's exact discretized cost to obtain `Q_t`, `R_t`,
  `N_t` and `Q_f`.

Hydrax accumulates the running cost after stepping the dynamics. The exporter
matches that convention exactly:

```text
x[t+1] = MJX_step(x[t], u[t])
l[t]   = dt * running_cost(x[t+1], u[t])
```

## 2. Run Crocoddyl's Riccati backward pass

Make the NPZ visible inside the Agimus control container, source the workspace,
and run:

```bash
source ~/ros2_ws/install/setup.bash
python ~/ros2_ws/src/agimus-demos/tools/compare_hydrax_feedback_gains_crocoddyl.py \
  /tmp/panda_feedback_local_lq.npz \
  --output /tmp/panda_feedback_crocoddyl.json \
  --matrices-output /tmp/panda_feedback_crocoddyl_matrices.npz \
  --plot /tmp/panda_feedback_crocoddyl.png
```

The consumer constructs one time-varying `ActionModelLQR` per Hydrax rollout
node and runs Crocoddyl's FDDP backward pass. It also runs an independent NumPy
Riccati recursion and refuses to produce a report if the two Riccati results
disagree.

## Sign convention

Hydrax stores and applies

```text
delta_u = K_feedback_mppi @ delta_x
```

Crocoddyl stores `K` but applies

```text
delta_u = -K_crocoddyl @ delta_x
```

The report therefore compares `K_feedback_mppi` with `-K_crocoddyl`.

## Reported quantities

For each trajectory snapshot, the JSON report contains:

- relative Frobenius error for the complete gain;
- separate position and velocity block errors;
- cosine similarity;
- both gain norms;
- first-node closed-loop spectral radii;
- Feedback-MPPI gain ESS and nominal-sample weight;
- the numerical agreement between Crocoddyl and the independent Riccati
  recursion;
- any Hessian regularization applied before the backward pass.

The ESS and nominal weight are important when interpreting a mismatch: a gain
estimated from an effectively one-sample softmax should not be expected to
match a local Riccati gain.

## Interpretation boundary

A close result demonstrates that the Feedback-MPPI sensitivity approximates
the Riccati feedback of the local quadratic Hydrax OCP. A mismatch can then be
studied against sample count, gain-batch selection, temperature and nominal
weight without any MuJoCo-versus-Pinocchio ambiguity.

A native nonlinear Crocoddyl controller should be evaluated only after this
baseline. It requires a deliberate decision about whether to reproduce
MuJoCo's damping/friction and implicit integration or to accept those as model
mismatch variables.
