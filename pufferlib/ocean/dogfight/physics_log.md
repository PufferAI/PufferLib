# Physics Sanity Log

Historical record of physics test results at specific commits.

**P-51D Reference values** (from P51d_REFERENCE_DATA.md):
- Max speed: 159 m/s (355 mph, Military power, sea level)
- Stall speed: 45 m/s (100 mph, 9000 lb, clean config)

---

## How to use

1. Run tests: `cd pufferlib/ocean/dogfight && python test_flight.py`
2. If at a clean commit worth recording, add entry below
3. Include commit hash from `git rev-parse --short HEAD`

---

## Results

| Commit | Date | max_speed | stall | climb | L/D | turn_30 | turn_60 | pitch | roll | Notes |
|--------|------|-----------|-------|-------|-----|---------|---------|-------|------|-------|
| P-51D  | ref  | 159       | 45    | 15    | 14.6| -       | -       | UP    | YES  | Reference targets |
| 0116b97c | 2026-01-13 | 86.5 | 75.5 | -4.9 | - | - | - | UP | YES | Old tests, pre-physics fix |
| 1c30c546 | 2026-01-14 | 149.6 | 50 | 16.3 | 14.7 | 2.2 | 9.4 | UP | YES | Coordinated turn tests, 97% eff |
