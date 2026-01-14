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

| Commit | Date | max_speed | cruise_50 | min_speed | dive_30 | dive_45 | climb | pitch | roll | Notes |
|--------|------|-----------|-----------|-----------|---------|---------|-------|-------|------|-------|
| | | ~159 exp | | ~45 stall | | | m/s | UP | YES | P-51D targets |
| 0116b97c | 2026-01-13 | 86.5 | 80.7 | 75.5 | 10.7 | 40.4 | -4.9 | UP | YES | +2° incidence, rate ctrl still dives |
