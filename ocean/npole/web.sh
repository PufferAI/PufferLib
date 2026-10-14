NP_SITE="${PUFFER_WEBSITE_DIR:-../docker/puffer.ai}"
NP_MODEL="$NP_SITE/docs/assets/models/npole_controller.bin"
if [ ! -f "$NP_MODEL" ]; then
    echo "Missing controller bundle: $NP_MODEL (see ocean/npole/README.md)" >&2
    exit 1
fi

mkdir -p "build/web/$ENV"
emcc "$SRC_FILE" -o "build/web/$ENV/game.html" \
    -std=c11 -D_POSIX_C_SOURCE=200809L -O3 -fno-fast-math -ffp-contract=off \
    -Wall -Wextra -Wno-unused-function \
    -I. -Isrc -I"$SRC_DIR" -Ivendor "${INCLUDES[@]}" \
    "${LINK_ARCHIVES[@]}" -lm \
    -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
    -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sFILESYSTEM=1 -sASSERTIONS=1 \
    -sINITIAL_MEMORY=33554432 -sALLOW_MEMORY_GROWTH=1 \
    -sEXIT_RUNTIME=0 \
    --no-entry \
    -sEXPORTED_FUNCTIONS='["_np_demo_init","_np_demo_reset","_np_demo_advance","_np_demo_state","_np_demo_estimate","_np_demo_tick","_np_demo_seed_value","_np_demo_ready","_np_demo_hold","_np_demo_success","_np_demo_failed","_np_demo_command","_np_demo_applied","_np_demo_pending","_np_demo_residual","_np_demo_toggle_pause","_np_demo_paused","_np_demo_set_manual","_np_demo_clear_manual"]' \
    -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
    --shell-file ocean/npole/shell.html \
    --preload-file "$NP_MODEL@resources/npole/npole_controller.bin"

if [ -d "$NP_SITE/docs/assets" ]; then
    mkdir -p "$NP_SITE/docs/assets/$ENV"
    cp -a "build/web/$ENV/." "$NP_SITE/docs/assets/$ENV/"
fi
echo "Built live ten-pendulum demo: build/web/$ENV/game.html"
