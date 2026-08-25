#!/usr/bin/env bash
# Profile native train GPU time with Nsight Systems.
#
# Usage:
#   ./profile.sh [ENV] [extra section.key=value ...]
#   ./profile.sh breakout train.total_timesteps=4000000
#
# Requires: nsys, sqlite3, a built ./puffer (builds breakout if missing).
set -euo pipefail

ENV_NAME="${1:-breakout}"
shift || true
EXTRA=("$@")

STEPS="${PROFILE_STEPS:-4000000}"
OUT="${PROFILE_OUT:-profile}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -x ./puffer ]]; then
    echo "Building native binary for $ENV_NAME..."
    ./build.sh "$ENV_NAME"
fi

echo "=== nsys profile: $ENV_NAME steps=$STEPS ==="
# Capture only between cudaProfilerStart/Stop (base.profile=1 in create/close).
# Graph node mode attributes time to kernels inside CUDA graphs.
nsys profile \
    --force-overwrite=true \
    --capture-range=cudaProfilerApi \
    --cuda-graph-trace=node \
    --sample=none \
    --trace=cuda,nvtx \
    -o "$OUT" \
    ./puffer train \
        "--base.profile=1" \
        "--train.total_timesteps=$STEPS" \
        "${EXTRA[@]}"

REP="${OUT}.nsys-rep"
SQL="${OUT}.sqlite"

echo
echo "=== Top kernels (by total time) ==="
nsys stats --report cuda_gpu_kern_sum --force-export=true "$REP" 2>/dev/null \
    | head -40 || nsys stats --report cuda_gpu_kern_sum:base --force-export=true "$REP" | head -40

echo
echo "=== NVTX ranges ==="
nsys stats --report nvtx_sum --force-export=true "$REP" 2>/dev/null | head -40 || true

echo
echo "=== Kernel groups ==="
nsys export --type=sqlite --force-overwrite=true -o "$SQL" "$REP"

# CUPTI schema: shortName -> StringIds; times in ns.
sqlite3 -header -column "$SQL" "
WITH kern AS (
  SELECT
    s.value AS name,
    (k.end - k.start) AS dur_ns
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.shortName = s.id
),
tot AS (SELECT SUM(dur_ns) AS t FROM kern)
SELECT
  CASE
    WHEN name LIKE '%gemm%' OR name LIKE '%Gemm%' OR name LIKE '%GEMM%'
      OR name LIKE '%cublasLt%' OR name LIKE '%ampere_bf16%'
      OR name LIKE '%hopper%gemm%' OR name LIKE '%xmma%'
      OR name LIKE '%Kernel2%' OR name LIKE '%splitK%' OR name LIKE '%splitk%'
      THEN 'matmul'
    WHEN name LIKE '%ppo_loss%' OR name LIKE '%ppo_var_mean%' OR name LIKE '%ppo_loss_reduce%'
      THEN 'ppo_loss'
    WHEN name LIKE '%mingru%' OR name LIKE '%scan%' OR name LIKE 'add_kernel%'
      THEN 'mingru'
    WHEN name LIKE '%muon%' OR name LIKE '%newton%' OR name LIKE '%ns_step%'
      THEN 'muon'
    WHEN name LIKE '%_step_kernel%' OR name LIKE 'gpu_%' AND name LIKE '%step%'
      OR name LIKE '%_log_kernel%'
      THEN 'env'
    WHEN name LIKE '%cast%' OR name LIKE '%select_copy%' OR name LIKE '%index_copy%'
      OR name LIKE '%transpose%' OR name LIKE '%scatter%' OR name LIKE '%Memcpy%'
      OR name LIKE '%fill_%' OR name LIKE '%clamp%'
      THEN 'copy_cast'
    WHEN name LIKE '%sample_logits%' OR name LIKE '%multinomial%' OR name LIKE '%rng%'
      OR name LIKE '%curand%' OR name LIKE '%philox%' OR name LIKE '%advance_rng%'
      THEN 'sample'
    WHEN name LIKE '%advantage%' OR name LIKE '%puff_advantage%' OR name LIKE '%gae%'
      THEN 'advantage'
    WHEN name LIKE '%prio%' OR name LIKE '%build_cdf%'
      THEN 'prio_replay'
    WHEN name LIKE '%assemble_decoder%'
      THEN 'decoder_grad'
    WHEN name LIKE '%zero_state%' OR name LIKE '%snapshot_initial%'
      OR name LIKE '%zero_frozen%'
      THEN 'state_util'
    WHEN name LIKE '%nccl%' OR name LIKE '%NCCL%'
      THEN 'nccl'
    ELSE name
  END AS group_name,
  COUNT(*) AS launches,
  ROUND(100.0 * SUM(dur_ns) / (SELECT t FROM tot), 1) AS pct,
  ROUND(SUM(dur_ns) / 1e6, 2) AS total_ms,
  ROUND(AVG(dur_ns) / 1e3, 2) AS avg_us,
  ROUND(MAX(dur_ns) / 1e3, 2) AS max_us
FROM kern
GROUP BY group_name
ORDER BY total_ms DESC
LIMIT 40;
"

echo
echo "=== Ungrouped leftovers (top 15 raw names not in buckets) ==="
sqlite3 -header -column "$SQL" "
WITH kern AS (
  SELECT s.value AS name, (k.end - k.start) AS dur_ns
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.shortName = s.id
),
tot AS (SELECT SUM(dur_ns) AS t FROM kern)
SELECT name,
  COUNT(*) AS n,
  ROUND(100.0 * SUM(dur_ns) / (SELECT t FROM tot), 2) AS pct,
  ROUND(SUM(dur_ns) / 1e6, 2) AS total_ms
FROM kern
WHERE name NOT LIKE '%gemm%' AND name NOT LIKE '%Gemm%' AND name NOT LIKE '%GEMM%'
  AND name NOT LIKE '%cublasLt%' AND name NOT LIKE '%xmma%' AND name NOT LIKE '%splitK%'
  AND name NOT LIKE '%splitk%' AND name NOT LIKE '%Kernel2%'
  AND name NOT LIKE '%ppo_%' AND name NOT LIKE '%mingru%' AND name NOT LIKE '%scan%'
  AND name NOT LIKE '%muon%' AND name NOT LIKE '%cast%' AND name NOT LIKE '%select_copy%'
  AND name NOT LIKE '%transpose%' AND name NOT LIKE '%scatter%' AND name NOT LIKE '%sample%'
  AND name NOT LIKE '%advantage%' AND name NOT LIKE '%prio%' AND name NOT LIKE '%assemble%'
  AND name NOT LIKE '%zero_state%' AND name NOT LIKE '%snapshot%' AND name NOT LIKE '%nccl%'
GROUP BY name
ORDER BY total_ms DESC
LIMIT 15;
"

echo
echo "Done. Artifacts: $REP  $SQL"
