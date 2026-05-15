# Orbital Dock

Spacecraft rendezvous and docking in the LVLH (Hill) reference frame. A chaser spacecraft must navigate 400–1200m to reach a docking port located 60m along the station's V-bar (prograde) axis. The approach is constrained to a 60° line-of-sight cone centered on the docking axis — the chaser cannot dock from arbitrary directions but must align with the V-bar corridor. A hard speed gate (2.0 m/s) enforces a controlled final approach. The physics use Clohessy-Wiltshire linearized relative motion with RK4 integration — the standard model for proximity operations in circular orbit.

PPO achieves a **99.5% dock rate** against a 2.0 m/s speed gate at 500M training steps.

## Physics

- **Dynamics:** Clohessy-Wiltshire (CW) equations, RK4 integration, dt = 1s
- **Orbit:** GEO (42,164 km altitude), mean motion n = 7.29e-5 rad/s
- **Thrust:** 10N max per axis, 500 kg chaser, max accel = 0.02 m/s²
- **Fuel:** 100 m/s delta-v budget
- **Frame:** LVLH — R-bar (radial), V-bar (prograde), H-bar (normal)

## Observations (10D)

| Index | Description |
|-------|-------------|
| 0-2 | Position [x, y, z] in LVLH (meters) |
| 3-5 | Velocity [vx, vy, vz] in LVLH (m/s) |
| 6 | Distance to dock point (m) |
| 7 | Speed (m/s) |
| 8 | Closing velocity (m/s, positive = approaching) |
| 9 | Time remaining (fraction) |

## Actions

Continuous `Box([-1, 1], shape=(3,))` — thrust fraction per LVLH axis (R-bar, V-bar, H-bar).

## Termination

- **Dock:** in LOS cone AND dist < 10m AND speed < 2.0 m/s
- **Crash:** y < dock_y - 5m (overshoot past dock point)
- **Timeout:** 2500 steps

## Reward

Per-step:
- `+0.01 * (prev_dist - dist)` — distance progress
- `+0.1 * (exp(-dist/20) - exp(-prev_dist/20))` — potential-based proximity
- `-0.005 * exp(-dist/30) * speed²` — proximity-weighted braking
- `-0.005 * (act²)` — control cost
- `-0.005` — time penalty

Terminal: dock = +0.5 to +1.0 (soft speed bonus), crash = -1.0, timeout = -0.5

## Reward Shaping

**Exponential proximity reward.** With linear distance progress (`prev_dist - dist`) over long distances, the agent learns to approach but won't risk crashing. The crash penalty outweighs the marginal progress reward near the dock, so the agent avoids docking. An exponential proximity bonus (`exp(-dist/20)`) pulls the agent to explore through the final approach.

**Potential-based formulation.** A naive proximity bonus (`+0.1 * exp(-dist/20)`) is farmable — the agent can park at ~20m from the dock and collect +0.02/step indefinitely, earning more per episode than the +1.0 dock terminal reward. The potential-based version (`exp(-dist/20) - exp(-prev_dist/20)`) only rewards *movement toward* the dock. Hovering pays zero.

**Proximity-weighted braking** (`exp(-dist/30) * speed²`) solves the final piece: speed control. Global velocity penalties create a "50m wall" where the agent learns to stop far from the dock to avoid the per-step speed cost. Weighting by proximity focuses the braking signal where it matters — at 800m the penalty is negligible, at 30m it's moderate, at 10m it's strong. The agent cruises at full speed for most of the approach and only brakes in the final stretch.

**Velocity annealing** Annealing the velocity requirements for a successful dock (from 10 → 2 m/s over 50K per-env steps) bootstraps the dock reward signal. Without annealing, the agent never experiences docking and can't learn the goal. Short annealing gives brief exposure to successful trajectories, with the braking reward pushing gradients as the task becomes more difficult.

## Architecture

- Actor: 2-layer MLP (10 → 256 → 256 → 3), GELU, logstd init = -1.0
- Critic: separate 2-layer MLP (10 → 256 → 256 → 1)
- RunningNorm on observations, no RNN
- 664.8K parameters

Training: lr=0.001, clip=0.06, ent=0.005, epochs=2, gamma=0.998, gae=0.98, 1024 envs, batch=65536, minibatch=4096, 500M steps.

## Files

- `orbital_dock.h` — C environment (CW dynamics, reward, physics)
- `orbital_dock.py` — Python PufferEnv wrapper
- `binding.c` — C-Python binding
- `render.h` — Raylib 3D visualization