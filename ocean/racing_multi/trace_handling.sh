#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

checkpoint=${1:?Usage: bash ocean/racing_multi/trace_handling.sh CHECKPOINT.bin OUTPUT.csv [FRAMES]}
output=${2:?Usage: bash ocean/racing_multi/trace_handling.sh CHECKPOINT.bin OUTPUT.csv [FRAMES]}
frames=${3:-2400}
models=()
for ((i = 0; i < 8; i++)); do models+=("$checkpoint"); done
export RACING_TRACE_PATH="$output"
exec ./puffer race "${models[@]}" --headless "--frames=$frames" \
    --policy.hidden_size=256 --policy.num_layers=3
