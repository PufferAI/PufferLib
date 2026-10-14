#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Shared latest race policy by default; explicit checkpoints retain multi-policy races.
models=()
options=()
for arg in "$@"; do
  if [[ "$arg" == --* ]]; then options+=("$arg"); else models+=("$arg"); fi
done
if [ ! -f ocean/racing/render_materials.bin ] || [ ! -f ocean/racing/car_lod.bin ] ||
   [ ocean/racing/prepare_render.py -nt ocean/racing/render_materials.bin ] ||
   [ ocean/racing/car.glb -nt ocean/racing/car_lod.bin ] ||
   [ ocean/racing/map_visual.bin -nt ocean/racing/render_materials.bin ] ||
   [ ocean/racing/assets/track/source/silverstone.glb -nt ocean/racing/render_materials.bin ]; then
  python3 ocean/racing/prepare_render.py
fi
if (( ${#models[@]} <= 1 )); then
  exec ./puffer eval "${models[0]:-latest}" \
    --policy.hidden_size=256 --policy.num_layers=3 "${options[@]}"
fi
exec ./puffer race "${models[@]}" \
  --policy.hidden_size=256 --policy.num_layers=3 "${options[@]}"
