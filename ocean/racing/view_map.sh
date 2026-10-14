#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for asset in ocean/racing/map.bin ocean/racing/map_visual.bin; do
    if [ ! -f "$asset" ]; then
        echo "Missing $asset; prepare the racing assets before launching the viewer." >&2
        exit 1
    fi
done
python3 ocean/racing/prepare_render.py --track
if [ "${1:-}" = --optix ]; then
    python3 ocean/racing/prepare_render.py
    for asset in ocean/racing/query_mesh.bin ocean/racing/route.bin ocean/racing/track.bin; do
        if [ ! -f "$asset" ]; then
            echo "Missing $asset; prepare the racing assets before launching the OptiX viewer." >&2
            exit 1
        fi
    done
    optix_include="${OPTIX_INCLUDE_DIR:-/tmp/puffysics-optix-9.1/include}"
    if [ ! -f "$optix_include/optix.h" ]; then
        echo "Set OPTIX_INCLUDE_DIR to the OptiX 9.1 include directory." >&2
        exit 1
    fi
    nvcc -std=c++17 -O2 -arch=compute_75 --ptx -I"$optix_include" \
        src/puffysics/raycast_optix.cu -o ocean/racing/raycast_optix.ptx
    cc -std=c99 -O2 -Wall -Wextra -DMAP_OPTIX -Iraylib-5.5_linux_amd64/include \
        -c ocean/racing/map_view.c -o ocean/racing/map_view.o
    nvcc -std=c++17 -O2 -I"$optix_include" ocean/racing/map_queries.cu \
        ocean/racing/map_view.o -Lraylib-5.5_linux_amd64/lib -lraylib -lm -ldl \
        -Xlinker -rpath -Xlinker '$ORIGIN/../../raylib-5.5_linux_amd64/lib' \
        -o ocean/racing/map_view
else
    cc -std=c99 -O2 -Wall -Wextra -Iraylib-5.5_linux_amd64/include \
        ocean/racing/map_view.c -Lraylib-5.5_linux_amd64/lib -lraylib -lm \
        -Wl,-rpath,'$ORIGIN/../../raylib-5.5_linux_amd64/lib' -o ocean/racing/map_view
fi
map_libs="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if command -v nix-store >/dev/null && [ -e /run/current-system/sw ]; then
    while IFS= read -r package; do
        case "$package" in
            *-libx11-*|*-libxcursor-*|*-libxi-*|*-libxinerama-*|*-libxrandr-*|*-libglvnd-*)
                map_libs="$map_libs:$package/lib"
                ;;
        esac
    done < <(nix-store --query --requisites /run/current-system/sw)
fi
export LD_LIBRARY_PATH="$map_libs"
exec ocean/racing/map_view
