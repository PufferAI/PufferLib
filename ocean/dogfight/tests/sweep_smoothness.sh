#!/usr/bin/env bash
#
# sweep_smoothness.sh - Brute-force grid sweep over physics damping/control
# coefficients. For each combo, runs the full 30-case smoothness matrix and
# emits one score line. Top 10 + baseline reference printed at the end.
#
# Knobs:
#   ctrl_slope   high-V control authority drop (per m/s above V_REF=100)
#   ctrl_min     authority floor [0,1]
#   damp_slope   extra rate-damping multiplier per m/s above V_REF
#   damp_mult    uniform multiplier on CM_Q, CL_P, CN_R
#
# Default grid: 11 x 9 x 9 x 9 = 8019 configs. At ~5ms each / N cores it
# completes in <10s on a 16-core machine; bump if you want more resolution.
#
# Usage:
#   bash ocean/dogfight/tests/sweep_smoothness.sh
#   bash ocean/dogfight/tests/sweep_smoothness.sh --top 25
#   bash ocean/dogfight/tests/sweep_smoothness.sh --out /tmp/x.txt --top 50
set -uo pipefail   # no -e: head/tail SIGPIPE on big sorts is harmless

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

BIN="ocean/dogfight/tests/sweep_smoothness"
OUT="/tmp/sweep_smoothness.txt"
TOP=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top) TOP="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Build harness if missing or stale.
if [[ ! -x "${BIN}" || "ocean/dogfight/tests/sweep_smoothness.c" -nt "${BIN}" ]]; then
    echo "[build] compiling ${BIN}..." >&2
    gcc -O2 -Wall \
        -I ocean/dogfight -I raylib-5.5_linux_amd64/include \
        ocean/dogfight/tests/sweep_smoothness.c \
        raylib-5.5_linux_amd64/lib/libraylib.a \
        -lm -lpthread -ldl \
        -o "${BIN}"
fi

# Parameter grid. Designed to bracket "no scaling" (default) and reasonable
# physical limits. Anything beyond these is unphysical or breaks the plant.
CTRL_VREFS="50 60 70 80 90 100 110"
CTRL_SLOPES="0.005 0.010 0.015 0.020 0.025 0.030 0.040 0.060"
CTRL_MINS="0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.70 1.00"
DAMP_SLOPES="0.000 0.005"
DAMP_MULTS="1.00 1.15 1.30 1.50 1.75"

TOTAL=$(( $(echo $CTRL_VREFS | wc -w) * $(echo $CTRL_SLOPES | wc -w) * \
          $(echo $CTRL_MINS | wc -w) * $(echo $DAMP_SLOPES | wc -w) * \
          $(echo $DAMP_MULTS | wc -w) ))

echo "[sweep] grid: $TOTAL configs across $(nproc) cores" >&2
echo "[sweep] writing to ${OUT}" >&2

START=$(date +%s)

# Emit all combos to stdin, run them in parallel via xargs -P.
# `bash -c` per row keeps args properly quoted; output is one line per run.
> "${OUT}"
for vref in $CTRL_VREFS; do
    for cs in $CTRL_SLOPES; do
        for cm in $CTRL_MINS; do
            for ds in $DAMP_SLOPES; do
                for dm in $DAMP_MULTS; do
                    echo "$vref $cs $cm $ds $dm"
                done
            done
        done
    done
done | xargs -P "$(nproc)" -L 1 sh -c \
    './ocean/dogfight/tests/sweep_smoothness --ctrl-vref "$1" --ctrl-slope "$2" --ctrl-min "$3" --damp-slope "$4" --damp-mult "$5"' \
    sh > "${OUT}"

END=$(date +%s)
N_DONE=$(wc -l < "${OUT}")
echo "[sweep] done: $N_DONE configs in $((END-START))s" >&2

# Helper to extract score and prepend for sorting. Assumes "score=N" appears
# as a single token. Lower score = better.
sort_by_score() {
    awk '{
        for (i=1; i<=NF; i++) {
            if ($i ~ /^score=/) {
                split($i, a, "=")
                print a[2] "\t" $0
                break
            }
        }
    }' "$1" | sort -k1,1g | cut -f2-
}

echo
echo "=== Top ${TOP} (lowest score, lower is better) ==="
sort_by_score "${OUT}" | head -n "${TOP}"

echo
echo "=== Baseline (no scaling, no damping boost) ==="
awk '/ctrl_slope=0\.0000/ && /ctrl_min=1\.000/ && /damp_slope=0\.0000/ && /damp_mult=1\.000/' "${OUT}" \
    || echo "(default config not in grid)"

echo
echo "=== Worst 5 ==="
sort_by_score "${OUT}" | tail -n 5

echo
echo "Output: ${OUT}"
echo "Re-sort with: bash $0 --top 50"
