Objective: Implement a high-performance simulation of world war 2 dogfighting as an RL environment into PufferLib.

Requirements:
1. 1M+ steps per second simulation on a single CPU core. This is easily attainable by following the practice of other environments in PufferLib - no memory allocations after initialization, pass all structs by reference
2. Single-file C implementation header-only. A small, separate .c file will be used for testing.
3. Match PufferLib C API for RL environments
4. TDD Test Driven Development

Environment details:
- Must model: managing throttle, aileron, rudder, elevator, and trigger to win dogfights in a real physics simulator with real approximation of Drag Polar, aerodynamic stall, etc
- Action space: Box(5) continuous [-1, 1]: throttle, elevator, ailerons, rudder, trigger (fire if > 0.5)
- Optional future: flaps
- Not modeling: full CFD, air turbulence, structural damage

- Reasonably accurate physics, thrust, lift, drag, kinetic and potential energy, conservation of momentum, conservation of energy, lift changing with angle of attack, etc
- Try to approximately match the performance of real world war 2 aircraft
- Agents to learn to manage energy and air combat maneuvers to win dogfights

Physics (3DOF point-mass, metric units):
- ρ = 1.225 kg/m³ (fixed sea level)
- q = 0.5 * ρ * V²  (dynamic pressure, Pa)
- L = C_L * q * S  (lift, N)
- D = (C_D0 + K * C_L²) * q * S  (drag, N)
- T = T_max * throttle  (thrust, N)
- W = m * g  (weight, N)

Constraints:
- C_L ≤ 1.4 (stall)
- n = L/W ≤ 8 (structural g-limit)

Approximations (valid for WW2, Mach < 0.6):
- Incompressible flow, flat earth, ignore prop torque/weather

Instructions:
- Read pufferlib/ocean/[target, snake] for simple examples of API compatibility and code standards
- Read pufferlib/ocean/nmmo3/nmmo3.h for a much more complex environment with the same game tick system as the desired Olm environment
- The implementation will live in pufferlib/ocean/dogfight with dogfight.h being the source and dogfight.c being a tiny main file.
- Build with: python setup.py build_ext --inplace --force
- Use pufferlib/ocean/drone_race/ as a template
- Opponent will be generated programatically, very simple at first like just flying straight, adding maneuvers later

References:
Links in CLAUDE.md
PufferAI docs: https://puffer.ai/docs.html
Reference environments: pufferlib/ocean. Source code is in .h files. Ignore .pyx. The .py files only contain bindings.

Code style and optimization:
- Use the environments "squared," "target," and "template" as API references. You must implement c_step, c_reset, and c_render
- The only dependency is raylib, which is for rendering only
- Match the code style of "snake" and "nmmo3" closely: procedural C with minimal abstraction, functions mainly split out to avoid duplicating code.
- No memory allocations after initialization. Pass all structs by reference.
