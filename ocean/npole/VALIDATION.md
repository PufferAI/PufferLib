# Ten pendulum validation

The frozen C controller passes all 32 reserved starts on native RK4,
independent MuJoCo and browser WebAssembly. The same cohort also passes on
the CPU of a LAN RTX 3090 host. Each complete trial has 1000 control ticks;
all 251 endpoints bounding the final five seconds satisfy every link's angle
and angular-speed limits. Minimum final hold is 417 scored ticks (8.34 s).

| Check | Result |
| --- | --- |
| Native C RK4 | 32/32 |
| Independent MuJoCo 3.12.0 | 32/32 |
| Browser WASM RK4 | 32/32 |
| LAN x86_64 CPU, repeated cohort | 32/32 |
| Independent mechanics, energy, integration and API tests | 15 passed |
| Native CPU debug build, development trial | ASan/UBSan passed |
| Environment API compiled by NVCC for `sm_86` | Passed |
| GPU trainer runtime | Unverified: LAN kernel/NVML version mismatch |

The 32 seeds are `2026101600` through `2026101631`. Source, model and build
hashes were frozen before those starts were evaluated. All outcomes and full
recordings were retained. A separate audit checked all 96 primary recordings
and all 32 LAN repetitions for hanging reset bounds, finite states, force/rail
bounds, the exact one-tick queue from zero and every final-window endpoint.
The starts are identical across engines; these are paired evaluations.

Forty historical Python state fixtures independently check the C controller
calculation: maximum normalized-command difference is `1.18e-8`, and maximum
recovered-state difference is `5.16e-14`. This is controller arithmetic parity,
separate from the new closed-loop trial results.

## Public evidence

The companion website contribution contains
[`validation.json`](https://github.com/marctheshark3/puffer.ai/blob/contrib/ten-pendulum/docs/assets/npole/validation.json),
both complete trial receipts, the audit, hashes and one representative complete
recording per primary engine under `docs/assets/npole/evidence/`. Those three
recordings share seed `2026101600`; the receipts include every outcome and every
primary recording's hash. The complete raw cohorts remain with the submitter
and can be regenerated with the included validators.

The live browser's development trial took 20.02 wall seconds for 20 simulated
seconds. Pause/resume, reset, manual delayed force, gallery embedding and a
390-pixel viewport passed with no page errors. This was headless Chromium
153.0.8010.12 with SwiftShader on aarch64. Page initialization took 0.283 s from
a local HTTP server; this does not measure Internet loading.

## Reproduce

Clone the companion website branch while its PR is pending, then run the
commands in [README.md](README.md). `tests/validate_npole.py` constructs its
MuJoCo geometry independently of the C coefficient tables and saves every
successful or failed trial.

For the browser checker, install Node.js, Playwright and Chromium in the
PufferLib checkout. Serve `build/web/npole` on port 8000 in another terminal:

```bash
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node tests/validate_npole_web.mjs http://localhost:8000/game.html \
  npole-web-validation 2026101600 32 build/web/npole
```

The browser checker hashes the four built assets before the starts, executes
the compiled controller and plant, saves every raw trace, checks queues and
all final endpoints, and rejects page errors or any failed trial. A fresh
output directory is required by both validators. The submitted web build uses
Emscripten 3.1.64, LLVM 19.1.1, Binaryen 118 and Raylib 5.5.

The demonstrated policy is frozen encoder inversion followed by classical
trajectory tracking and LQR. Its JEPA predictor does not choose actions. No
Puffer-trained policy result is claimed. The LAN host's loaded kernel driver
is 595.91.07 while NVML is 595.99; GPU runtime validation remains open. The
declared tiny inertia difference and reference optimizer timeout are explained
in [README.md](README.md).
