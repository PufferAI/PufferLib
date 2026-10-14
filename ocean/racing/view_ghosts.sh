#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Latest eight snapshots from the completed ~1B-step run; names are shuffled at startup.
exec ./puffer ghosts \
  "checkpoints/racing/1790518872539/0000000999817216.bin" \
  "checkpoints/racing/1790518872539/0000000998768640.bin" \
  "checkpoints/racing/1790518872539/0000000996147200.bin" \
  "checkpoints/racing/1790518872539/0000000993525760.bin" \
  "checkpoints/racing/1790518872539/0000000990904320.bin" \
  "checkpoints/racing/1790518872539/0000000988282880.bin" \
  "checkpoints/racing/1790518872539/0000000985661440.bin" \
  "checkpoints/racing/1790518872539/0000000983040000.bin" \
  --policy.hidden_size=256 --policy.num_layers=3 "$@"
