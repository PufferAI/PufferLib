# Ten pendulum environment

Swing up ten passive serial pendulums using one cart force, then balance every
link for the final five seconds of a twenty-second episode. The live demo uses
a frozen JEPA target encoder to recover the current state, followed by
classical trajectory tracking and upright LQR. The predictor does not select
actions. The included demonstration is a hybrid controller; the Puffer training
interface supports learning an additional policy.

## Body and task

The cart has mass 1 kg. Each massless link is 0.5 m long with a 0.1 kg tip mass.
Gravity is 9.81 m/s². State coordinates are cart position, ten absolute angles,
cart velocity and ten absolute angular velocities. Coupled dynamics come from
the point-mass mass matrix and centrifugal terms. Viscous damping is 0.05 in
these absolute coordinates. The declared analytic mass matrix includes a
constant `1e-6 I`; no solver fallback changes this value.

Commands arrive at 50 Hz, are capped at ±10 N, and enter a real 20 ms queue
initially containing zero. RK4 uses twenty 1 ms substeps per command. Only the
cart is actuated. Coordinates, angular speeds and accelerations are never
projected onto the reference trajectory or clipped to keep a trial alive.

Every reset is near hanging: cart position uniform within ±0.08 m, angles
π ±0.04 rad and velocities within ±0.02. Native and WASM resets use the same
SplitMix64 implementation. Evaluation compares identical explicit starts.

Success requires the cart to stay strictly within ±2.4 m for all recorded
control ticks. At every one of the 251 endpoints bounding the final five
seconds, all wrapped angles must be below 0.22 rad and all angular speeds below
1 rad/s. Rail departures and nonfinite dynamics fail. Training reward shaping
uses link heights; `perf` reports the strict episode success criterion.

## Build and run

Clone the website repository beside PufferLib to obtain the frozen bundle:

```bash
git clone --branch 5.0 https://github.com/PufferAI/puffer.ai ../puffer.ai
./build.sh npole --cpu
./npole --model ../puffer.ai/docs/assets/models/npole_controller.bin
```

Headless evaluation records every control tick and reports every outcome:

```bash
./npole --headless --episodes 32 --seed 2026101600 \
  --model ../puffer.ai/docs/assets/models/npole_controller.bin \
  --record npole-trials.csv
```

For the browser, install Emscripten and activate its environment, then:

```bash
PUFFER_WEBSITE_DIR=../puffer.ai ./build.sh npole --web
python3 -m http.server --directory build/web/npole 8000
```

Open `http://localhost:8000/game.html`. Space pauses, R resets, and Shift plus
an arrow key or a held direction button takes control of the cart. Manual
inputs use the same force cap and actuator queue. The demo calculates its
controller and advances its physics live. It displays failed trials before
resetting, without a raw-state fallback after encoder inversion failure.

The standard training build uses the Puffer environment API:

```bash
./build.sh npole
./puffer train
```

The demonstrated bundle is not a PufferNet checkpoint. Puffer's training
observations/actions use its usual float32 buffers; the standalone frozen
controller keeps state, normalization, inference, inversion, gains and commands
in float64. Its `NP10v001` bundle has a checked header and 69731 little-endian
doubles. Weights and their manifest live in the website repository.

## Observation and controller contracts

The environment exposes 34 features: cart position/velocity, ten
sin/cos/angular-speed triplets, last applied command and pending command.
Angular speeds are divided by 8. The frozen encoder's original 33-feature
input omits pending; delay compensation receives pending separately.

The encoder is a 33 → 128 → 128 → 64 SiLU MLP. A central-difference Jacobian
and a one-sided Jacobi SVD recover a state from the latent, known last command
and fixed nominal reference. Inversion uses twenty iterations, `1e-5` Jacobian
offsets and a strict `1e-12` latent residual limit. A failed inverse fails the
trial. Feedback receives only the decoded state.

The 18-second reference, 900 tracking gains and delay-aware terminal LQR gain
were frozen offline. Extended precision was used for offline Riccati costs;
executed gains are float64. The trajectory optimizer timed out, and its
returned iterate separately passed dynamics/endpoint feasibility checks.
Objective optimality is not established. The browser does not optimize or
train during a trial. Source and original protocol:
[Rage Mill](https://github.com/marctheshark3/rage-mill),
[ten-pole control protocol](https://github.com/marctheshark3/rage-mill/blob/main/docs/POLES.md).

## Validate

The tests compare acceleration to independent Cartesian endpoint Jacobians
and MuJoCo, check energy/work consistency, RK4 convergence, the complete reset
range, final-link goal limits and the Puffer API's queue/reset logging.

```bash
python3 -m pip install numpy pytest mujoco
python3 -m pytest tests/test_npole.py
python3 tests/validate_npole.py \
  --model ../puffer.ai/docs/assets/models/npole_controller.bin \
  --seed 2026101600 --episodes 32 --workers 4 --final \
  --out npole-validation
```

The validation tool freezes source, controller and loaded-library hashes before
using the evaluation starts. It saves every successful or failed recording and
checks all final-window endpoints. The independent MuJoCo model uses tiny
`1e-6` tip rotational inertias. Its cart inertia differs slightly from the
analytic regularization, so the acceleration comparison explicitly allows the
existing `3e-5` relative / `8e-6` absolute envelope rather than bitwise equality.

The frozen Python controller's historical result was 32/32 on high-rate Euler
and 32/32 on independent MuJoCo. Those historical scores alone do not qualify
this C/WASM port; the website's validation receipt records the new evaluations.

MIT, Copyright (c) 2026 Marc / Rage Industries. This package contains the
pendulum implementation and its controller, with no robot geometry or assets.
