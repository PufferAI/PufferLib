OSRS_ASSET_GROUP="${ENV#osrs_}"
if [ -z "$OSRS_ASSET_GROUP" ] || [ "$OSRS_ASSET_GROUP" = "$ENV" ]; then
    echo "Error: OSRS web hook requires an osrs_* environment" >&2
    exit 1
fi

OSRS_PRELOAD_LINES=$(python3 ocean/osrs/scripts/osrs_asset_manifest.py \
    emcc-preload-args ocean/osrs/asset_manifest.json \
    --group core \
    --group "$OSRS_ASSET_GROUP" \
    --group combat_visuals \
    --group gui \
    --group items)

OSRS_PRELOAD=()
while IFS= read -r line; do
    [ -z "$line" ] && continue
    if [[ "$line" != "--preload-file "* ]]; then
        echo "Error: invalid OSRS preload argument: $line" >&2
        exit 1
    fi
    OSRS_PRELOAD+=(--preload-file "${line#--preload-file }")
done <<< "$OSRS_PRELOAD_LINES"

if [ "${#OSRS_PRELOAD[@]}" -eq 0 ]; then
    echo "Error: OSRS asset manifest produced no web preload files" >&2
    exit 1
fi

# Inferno route graphs are optional: not in the required tarball, but the
# browser pack should ship them so WASM does not rebuild LOS at startup.
if [ "$ENV" = "osrs_inferno" ]; then
    INF_ROUTE_BAKE="ocean/osrs/data/inferno_route.bin"
    if [ -f "$INF_ROUTE_BAKE" ]; then
        OSRS_PRELOAD+=(--preload-file "$INF_ROUTE_BAKE@$INF_ROUTE_BAKE")
        echo "Preloading Inferno route bake: $INF_ROUTE_BAKE"
    else
        echo "Warning: $INF_ROUTE_BAKE missing; Inferno web will rebuild route topology at runtime" >&2
    fi
fi

mkdir -p "build/web/$ENV"
echo "Compiling $ENV for web..."
emcc \
    -o "build/web/$ENV/game.html" \
    "$SRC_FILE" \
    -O3 -Wall -Wno-narrowing \
    "${LINK_ARCHIVES[@]}" \
    -I. -Isrc -I"$SRC_DIR" -Ivendor "${INCLUDES[@]}" \
    -L. -L./"$RAYLIB_NAME"/lib \
    -sASSERTIONS=2 -gsource-map \
    -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
    -sLZ4=1 \
    --js-library vendor/puf_web_vsync.js \
    --shell-file ocean/osrs/osrs_shell.html \
    -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
    -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
    "${OSRS_PRELOAD[@]}" \
    "${EXTRA_CFLAGS[@]}"
echo "Built: build/web/$ENV/game.html"
WEBSITE_DIR="${PUFFER_WEBSITE_DIR:-../docker/puffer.ai}"
WEBSITE_ASSETS="$WEBSITE_DIR/docs/assets"
if [ -d "$WEBSITE_ASSETS" ]; then
    mkdir -p "$WEBSITE_ASSETS/$ENV"
    cp -a "build/web/$ENV/." "$WEBSITE_ASSETS/$ENV/"
    echo "Published: $WEBSITE_ASSETS/$ENV/"
fi
