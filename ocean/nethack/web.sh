# NetHack web build: sourced by build.sh in --web mode instead of the generic
# raylib/puffercpu path. The demo is a terminal program (ocean/nethack/nethack.c
# renders ANSI text frames), so it links the engine built for wasm and uses the
# ANSI-to-HTML shell in vendor/nh_web_shell.html. Requires an activated emsdk.
set -e
NLE_DIR="vendor/fast-nle"
WEB_LIB_DIR="$NLE_DIR/build-web"
if [ ! -d "$NLE_DIR/src" ]; then
    echo "Cloning fast-nle ..."
    git clone --depth 1 "https://github.com/FinlaySanders/fast-nle.git" "$NLE_DIR"
fi
if [ ! -f "$WEB_LIB_DIR/libnethack.a" ]; then
    echo "Building the NetHack engine for wasm (emcmake) ..."
    emcmake cmake -S "$NLE_DIR" -B "$WEB_LIB_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$WEB_LIB_DIR" --target nethack -j"$(nproc)"
fi
for lib in libnethack.a libtmt.a libbz2_static.a; do
    [ -f "$WEB_LIB_DIR/$lib" ] || { echo "Error: $WEB_LIB_DIR/$lib missing" >&2; exit 1; }
done
[ -f "$WEB_LIB_DIR/dat/nhdat" ] || { echo "Error: $WEB_LIB_DIR/dat/nhdat missing" >&2; exit 1; }
# policy weights packed into the page (override with NETHACK_WEB_WEIGHTS=path)
WEIGHTS="${NETHACK_WEB_WEIGHTS:-resources/nethack/nethack_score_weights.bin}"
[ -f "$WEIGHTS" ] || { echo "Error: weights $WEIGHTS missing" >&2; exit 1; }
mkdir -p "build/web/$ENV"
echo "Compiling $ENV for web (terminal demo) ..."
em++ -O2 -DPLATFORM_WEB -DPUFFER_NETHACK \
    -I. -Isrc -Iocean/nethack -Ivendor -I"${RAYLIB_NAME:-raylib-5.5_webassembly}/include" \
    -I"$NLE_DIR/include" -I"$WEB_LIB_DIR/include" \
    -I"$WEB_LIB_DIR/_deps/deboost_context-src/include" \
    ocean/nethack/nethack.c \
    "$WEB_LIB_DIR/libnethack.a" "$WEB_LIB_DIR/libtmt.a" "$WEB_LIB_DIR/libbz2_static.a" \
    -sASYNCIFY -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=16777216 -sASYNCIFY_STACK_SIZE=1048576 \
    -sENVIRONMENT=web,node -sEXIT_RUNTIME=0 \
    --preload-file "$WEB_LIB_DIR/dat/nhdat@/vendor/fast-nle/build/dat/nhdat" \
    --preload-file "$WEIGHTS@resources/nethack/nethack_score_weights.bin" \
    --shell-file ocean/nethack/web_shell.html \
    -o "build/web/$ENV/game.html"
echo "Built: build/web/$ENV/game.html (weights: $WEIGHTS)"
