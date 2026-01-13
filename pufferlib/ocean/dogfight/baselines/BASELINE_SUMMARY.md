# Baseline Training Results

Date: 2026-01-12
Training: 100M steps each

---

## Pre-Reward Shaping (baseline)
Reward: `-dist/10000` per step (pursuit only)

| Run | Episode Return | Episode Length |
|-----|----------------|----------------|
| 1   | -31.78         | 1111           |
| 2   | -46.42         | 1247           |
| 3   | -73.12         | 1371           |
| **Mean** | **-50.44** | **1243**       |

Observations:
- High variance (-31 to -73)
- All returns negative
- Weak reward signal (~-0.03 per step)

---

## Post-Reward Shaping (Phase 3.5)
Reward: base pursuit + closing velocity + tail position + altitude/speed penalties

| Run | Episode Return | Episode Length |
|-----|----------------|----------------|
| 1   | -66.82         | 1140           |
| 2   | +5.32          | 1063           |
| 3   | +16.13         | 1050           |
| **Mean** | **-15.12** | **1084**       |

Observations:
- **2 of 3 runs achieved positive returns** (significant improvement)
- Shorter episodes (more decisive behavior)
- Still high variance (need more tuning)
- Closing velocity and tail position rewards working
