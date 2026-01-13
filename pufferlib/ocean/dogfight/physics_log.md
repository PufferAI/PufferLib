# Physics Sanity Log

Historical record of physics test results at specific commits.

**Theoretical values** (from dogfight.h constants):
- Max speed: 143.7 m/s (at 100% throttle, level flight)
- Stall speed: 39.5 m/s (minimum lift = weight)

---

## How to use

1. Run tests: `cd pufferlib/ocean/dogfight && python test_flight.py`
2. If at a clean commit worth recording, add entry below
3. Include commit hash from `git rev-parse --short HEAD`

---

## Results

| Commit | Date | max_speed | cruise_50 | min_speed | dive_30 | dive_45 | climb | pitch | roll | Notes |
|--------|------|-----------|-----------|-----------|---------|---------|-------|-------|------|-------|
| | | ~144 exp | | ~40 stall | | | m/s | UP | YES | expected |
