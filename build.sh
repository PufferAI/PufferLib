#!/usr/bin/env bash
set -e

# Usage:
#   ./build.sh breakout              # Native train/eval -> ./puffer
#   ./build.sh path/to/my_env        # Any env directory holding my_env.h (or .cu) and my_env.ini
#   ./build.sh path/to/my-repo       # Otherwise its one NAME.ini names it; NAME.h may be in src/
#   ./build.sh breakout mybin        # Native -> ./mybin (relative to the caller)
#   ./build.sh breakout --cu         # CUDA env (ENV_HEADER=ENV_DIR/ENV.cu; exclusive vs .h)
#   ./build.sh robot_arm             # CUDA-only; implies --cu
#   ./build.sh breakout --float      # float32 precision (required for --slowly)
#   ./build.sh breakout --cpu        # Play/eval binary (optimized) -> ./breakout
#   ./build.sh osrs_inferno --cpu    # OSRS visual policy viewer (ocean/osrs_inferno/osrs_inferno.c)
#   ./build.sh nethack --cpu         # NetHack TTY demo (ocean/nethack/nethack.c)
#   ./build.sh breakout myplay --cpu # Play -> ./myplay
#   ./build.sh breakout --debug      # Debug (-O0 -g; sanitizers on --cpu)
#   ./build.sh breakout --dry-run    # Print the compile commands instead of running them
#   ./build.sh breakout --check      # Syntax-only compile with the --cpu flags; no binary
#   ./build.sh breakout --test       # Env test harness with sanitisers -> build/test_breakout
#   ./build.sh breakout --web        # Emscripten web build -> build/web/breakout
#                                    # packs website 1M-cap policy + *_web.ini
#                                    # copy build/web/ENV/* to ../docker/puffer.ai/docs/assets/ENV/
#   ./build.sh breakout --profile    # Kernel profiling binary -> ./build/profile_breakout
#   ./build.sh constellation         # Sweep dashboard -> ./seethestars
#   ./build.sh cache_data            # Sweep log cache -> ./cache_data
#   ./build.sh trailer               # 5.0 trailer -> ./resources/trailer/trailer (also exports diagrams)
#
# Paths resolve against this script's directory, so it works from anywhere.
# An env's ini may have a [build] section: src, include, archives, cflags,
# ldflags, download (archives unpacked next to this script) and run (a shell
# command in the env dir); other keys are variables. A [build.linux], [build.cpu]
# or [build.macos.cpu] section applies to those builds only. Values are
# shell-expanded. See config/impulse_wars.ini and nethack.ini.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DIR=$PWD
if [ -z "$1" ]; then
    echo "Usage: ./build.sh ENV|ENV_DIR [OUT] [--cu] [--float] [--debug] [--cpu] [--web]" \
        "[--profile] [--dry-run] [--check] [--test]"
    exit 1
fi
ENV=$1
shift
OUT=""
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    OUT=$1
    [[ $OUT == /* ]] || OUT=$PWD/$OUT
    shift
fi

# ENV: an Ocean name, an env directory (a path with a slash), or a standalone app.
ENV_DIR=""
if [[ $ENV == */* ]]; then
    ENV_DIR=$(cd "$ENV" && pwd)
    ENV=${ENV_DIR##*/}
    inis=("$ENV_DIR"/*.ini)
    if [ ! -f "$ENV_DIR/$ENV.h" ] && [ ! -f "$ENV_DIR/$ENV.cu" ] && [ -f "${inis[0]}" ] \
        && [ ${#inis[@]} -eq 1 ]; then
        ENV=$(basename "${inis[0]}" .ini)
    fi
elif [ -d "$ROOT/ocean/$ENV" ]; then
    ENV_DIR="$ROOT/ocean/$ENV"
fi
if [[ ! $ENV =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "Error: env name '$ENV' must be a C identifier (letters, digits, _)"
    exit 1
fi
cd "$ROOT"

USE_GPU_ENV=0
SNAKE_RAW=0
DRY_RUN=""
CHECK=""
TEST=""
while [ $# -gt 0 ]; do
    case $1 in
        --cu) USE_GPU_ENV=1 ;;
        --float) PRECISION="-DPRECISION_FLOAT" ;;
        --no-onehot) SNAKE_RAW=1 ;;
        --debug) DEBUG=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --check) CHECK=1 ;;
        --test)  TEST=1; DEBUG=1 ;;
        --web)   MODE=web ;;
        --profile) MODE=profile ;;
        --cpu)   MODE=cpu ;;
        *) echo "Error: unknown argument '$1'" && exit 1 ;;
    esac
    shift
done
MODE=${MODE:-native}
[ -z "$CHECK$TEST" ] || MODE=cpu

invoke() {
    if [ -n "$DRY_RUN" ]; then printf '%q ' "$@"; echo; else "$@"; fi
}

if [ "$ENV" = "robot_arm" ]; then
    USE_GPU_ENV=1
    case "$MODE" in
        cpu|web)
            echo "Error: robot_arm physics is CUDA-only; use the native trainer build" >&2
            exit 1
            ;;
    esac
fi

if [ "$ENV" = "all" ]; then
    FAILED=""
    for env_dir in ocean/*/; do
        env=$(basename "$env_dir")
        if bash "$0" "$env" && bash "$0" "$env" --float; then
            echo "OK: $env"
        else
            echo "FAIL: $env"
            FAILED="$FAILED\n  $env"
        fi
    done

    if [ -n "$FAILED" ]; then
        echo -e "\nFailed builds:$FAILED"
    fi
    exit 0
fi

# Linux/mac
PLATFORM="$(uname -s)"
if [ "$PLATFORM" = "Linux" ]; then
    OS=linux
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP_FLAGS=(-fopenmp)
    OMP_LIB=-lomp5
    STANDALONE_LDFLAGS=(-lGL)
else
    OS=macos
    RAYLIB_NAME='raylib-5.5_macos'
    OMP_PREFIX="$(brew --prefix libomp)"
    OMP_FLAGS=(-Xclang -fopenmp -I"$OMP_PREFIX/include" -L"$OMP_PREFIX/lib" -lomp)
    OMP_LIB=-lomp
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
fi
# Debug builds on Linux and every --test build (the harness relies on ASan).
SANITIZE_FLAGS=(-fsanitize=address,undefined,bounds,pointer-overflow -fno-omit-frame-pointer)
[ "$OS" = linux ] || [ -n "$TEST" ] || SANITIZE_FLAGS=()

CLANG_WARN=(
    -Wall
    -Wno-narrowing
    -Wno-unreachable-code
    -ferror-limit=3
    -Werror=incompatible-pointer-types
    -Werror=return-type
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
)

download() {
    local name=$1 url=$2
    [ -d "$name" ] && return
    if [ -n "$DRY_RUN" ]; then
        echo "Would download $name"
        return
    fi
    echo "Downloading $name..."
    case "$url" in
        *.zip) curl -sL "$url" -o "$name.zip" && unzip -q "$name.zip" && rm "$name.zip" ;;
        *)     curl -sL "$url" -o "$name.tar.gz" && tar xf "$name.tar.gz" && rm "$name.tar.gz" ;;
    esac
}

RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.zip"
else
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.tar.gz"
fi

RAYLIB_A="$RAYLIB_NAME/lib/libraylib.a"
INCLUDES=(-I./$RAYLIB_NAME/include -I./src -I./vendor)
LINK_ARCHIVES=("$RAYLIB_A")
EXTRA_SRC=()
EXTRA_LDFLAGS=()
EXTRA_CFLAGS=()
if [ -n "${NVCC_EXTRA:-}" ]; then
    read -ra _nvcc_extra <<< "$NVCC_EXTRA"
    EXTRA_CFLAGS+=("${_nvcc_extra[@]}")
fi
SRC_FILE=""

if [ "$ENV" = "constellation" ]; then
    SRC_DIR="src"
    OUTPUT_NAME="seethestars"
    STANDALONE=1
    CLANG_WARN+=(-Wno-unused-function)
elif [ "$ENV" = "cache_data" ]; then
    SRC_DIR="src"
    OUTPUT_NAME="cache_data"
    SRC_FILE="src/constellation.c"
    EXTRA_CFLAGS+=(-DPUFFER_CACHE_DATA)
    STANDALONE=1
    CLANG_WARN+=(-Wno-unused-function)
elif [ "$ENV" = "trailer" ]; then
    SRC_DIR="resources/trailer"
    OUTPUT_NAME="resources/trailer/trailer"
    SRC_FILE="resources/trailer/puffer5.c"
    EXTRA_SRC=(resources/trailer/architecture.c resources/trailer/stars.c
        resources/trailer/plot_scale.c resources/trailer/fonts.c)
    EXTRA_CFLAGS+=(-DARCHITECTURE_LIB -Iresources)
    STANDALONE=1
    CLANG_WARN+=(-Wno-unused-function)
elif [ -n "$ENV_DIR" ]; then
    SRC_DIR="$ENV_DIR"
    if [ -f "$ENV_DIR/src/$ENV.h" ] || [ -f "$ENV_DIR/src/$ENV.cu" ]; then
        SRC_DIR="$ENV_DIR/src"
    fi
else
    echo "Error: environment '$ENV' not found" && exit 1
fi

# ini.h reads config from these roots instead of the cwd; web builds keep the cwd.
ROOT_DEFINES=("-DPUFFER_ROOT=\"$ROOT\"")
ENV_INI=""
if [ -n "$ENV_DIR" ]; then
    ROOT_DEFINES+=("-DPUFFER_ENV_DIR=\"$ENV_DIR\"")
    if [ -f "$ENV_DIR/$ENV.ini" ]; then
        ENV_INI="$ENV_DIR/$ENV.ini"
    elif [ -f "config/$ENV.ini" ]; then
        ENV_INI="config/$ENV.ini"
    fi
fi

# key<TAB>value lines of the ini's [build] section.
ini_build_section() {
    awk -v sq="'" -v tags=" $2 " '
    /^[ \t]*\[/ {
        s = $0; sub(/[#;].*/, "", s); gsub(/[][ \t]/, "", s)
        n = split(s, p, "."); in_build = (p[1] == "build")
        for (i = 2; i <= n; i++) if (!index(tags, " " p[i] " ")) in_build = 0
        next
    }
    in_build && index($0, "=") {
        out = ""; q = ""
        for (i = 1; i <= length($0); i++) {
            c = substr($0, i, 1)
            if (c == "\"" || c == sq) {
                q = (q == c) ? "" : (q == "" ? c : q)
            } else if ((c == "#" || c == ";") && q == "") {
                break
            }
            out = out c
        }
        eq = index(out, "=")
        key = substr(out, 1, eq - 1); val = substr(out, eq + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", key); gsub(/^[ \t]+|[ \t]+$/, "", val)
        # List values are parsed as shell words later, so their quotes stay.
        n = length(val); c = substr(val, 1, 1)
        if (key !~ /^(src|include|archives|cflags|ldflags|download)$/ \
                && n >= 2 && (c == "\"" || c == sq) && substr(val, n, 1) == c) {
            val = substr(val, 2, n - 2)
        }
        if (key != "") printf "%s\t%s\n", key, val
    }' "$1"
}

env_path() {
    case $1 in
        /*) echo "$1" ;;
        *)  echo "$ENV_DIR/$1" ;;
    esac
}

if [ -n "$ENV_INI" ]; then
    export PUFFER_ROOT="$ROOT" ENV_DIR ENV
    [ "$MODE" = web ] && TAGS=web || TAGS="$OS $MODE"
    while IFS=$'\t' read -r key val; do
        # Lists are parsed as shell words, so a quoted entry may contain spaces
        # and relative entries resolve against the env dir after splitting.
        # run commands and variables are one string.
        case "$key" in
            src|include|archives|cflags|ldflags|download) eval "words=($val)" ;;
            *) eval "val=\"$val\"" ;;
        esac
        case "$key" in
            src)      for w in "${words[@]}"; do EXTRA_SRC+=("$(env_path "$w")"); done ;;
            include)  for w in "${words[@]}"; do INCLUDES+=(-I"$(env_path "$w")"); done ;;
            archives) for w in "${words[@]}"; do LINK_ARCHIVES+=("$(env_path "$w")"); done ;;
            cflags)   EXTRA_CFLAGS+=("${words[@]}") ;;
            ldflags)  EXTRA_LDFLAGS+=("${words[@]}") ;;
            download)
                for w in "${words[@]}"; do
                    name=${w##*/}; name=${name%.tar.gz}
                    download "${name%.zip}" "$w"
                done
                ;;
            run)
                if [ -n "$DRY_RUN" ]; then
                    echo "Would run: $val"
                else
                    (cd "$ENV_DIR" && bash -c "$val")
                fi
                ;;
            *) printf -v "$key" %s "$val" ;;
        esac
    done < <(ini_build_section "$ENV_INI" "$TAGS")
fi

# src/ocean.cu compiles only this env's custom net (PUFFER_NETHACK, PUFFER_NMMO3, …).
EXTRA_CFLAGS+=("-DPUFFER_$(echo "$ENV" | tr a-z A-Z)")

case "$ENV" in
    osrs_*)
        invoke python3 ocean/osrs/scripts/osrs_asset_manifest.py generate-c-header \
            ocean/osrs/asset_manifest.json \
            --output ocean/osrs/osrs_assets_generated.h
        invoke bash ocean/osrs/scripts/setup-data.sh
        ;;
esac

USER_OUTPUT_NAME=${OUTPUT_NAME-}
# Outputs are relative to the caller's directory, --test under build/ like
# --profile. OUT overrides.
if [ -n "$TEST" ]; then
    OUTPUT_NAME=${OUTPUT_NAME:-$START_DIR/build/test_$ENV}
fi
OUTPUT_NAME=${OUTPUT_NAME:-$START_DIR/$ENV}
if [ -n "$OUT" ]; then
    OUTPUT_NAME=$OUT
fi
invoke mkdir -p "$(dirname "$OUTPUT_NAME")"
# Header-only envs compile src/puffercpu.c. SRC_FILE is the custom standalone
# for osrs_* (visual sim) and nethack (TTY demo); see --cpu / web.sh.
SRC_FILE=${SRC_FILE:-$SRC_DIR/$ENV.c}

if [ "$(uname -m)" = "x86_64" ]; then
    SIMD_FLAGS=(-mavx2 -mfma)
else
    SIMD_FLAGS=()
fi
if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT=(-g -O0 "${CLANG_WARN[@]}" "${SANITIZE_FLAGS[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O0 -g"
    LINK_OPT="-g"
else
# No -DNDEBUG: keep assert() active (train/sweep fail-fast with messages).
    CLANG_OPT=(-O2 "${CLANG_WARN[@]}" "${SIMD_FLAGS[@]}")
    NVCC_OPT="-O2 --threads 0"
    LINK_OPT="-O2"
fi
# Dashboard / cache / trailer: compile SRC_FILE only (not puffercpu / CUDA / obs_t).
if [ "$STANDALONE" = "1" ]; then
    if [ "$MODE" = "web" ] || [ "$MODE" = "profile" ] || [ -n "$CHECK" ] || [ -n "$TEST" ]; then
        echo "Error: $ENV is a standalone app, not an env" >&2
        exit 1
    fi
    echo "Compiling $ENV..."
    invoke ${CC:-clang} "${CLANG_OPT[@]}" \
        -I. "${INCLUDES[@]}" \
        "$SRC_FILE" "${EXTRA_SRC[@]}" -o "$OUTPUT_NAME" \
        "${LINK_ARCHIVES[@]}" \
        "${EXTRA_LDFLAGS[@]}" \
        "${STANDALONE_LDFLAGS[@]}" \
        -lm -lpthread \
        -DPLATFORM_DESKTOP \
        "${ROOT_DEFINES[@]}" \
        "${EXTRA_CFLAGS[@]}"
    echo "Built: $OUTPUT_NAME"
    if [ "$ENV" = "trailer" ]; then
        echo "Compiling architecture..."
        invoke ${CC:-clang} "${CLANG_OPT[@]}" \
            -I. "${INCLUDES[@]}" \
            resources/trailer/architecture.c resources/trailer/fonts.c \
            -o resources/trailer/architecture \
            "${LINK_ARCHIVES[@]}" \
            "${EXTRA_LDFLAGS[@]}" \
            "${STANDALONE_LDFLAGS[@]}" \
            -lm -lpthread \
            -DPLATFORM_DESKTOP \
            -Iresources
        echo "Built: ./resources/trailer/architecture"
        echo "Exporting diagrams..."
        invoke mkdir -p resources/trailer/shots
        invoke ./resources/trailer/architecture --shot resources/trailer/shots/loop.png
        invoke ./"$OUTPUT_NAME" --plot resources/trailer/shots/fig_scale.png
    fi
    exit 0
fi
if [ "$MODE" = "cpu" ]; then
    STANDALONE_SOURCE="src/puffercpu.c"
    STANDALONE_DEFINES=()
    if [ -n "$TEST" ]; then
        # The harness drives the env header directly, even when the env has its own main.
        ENV_HEADER="$SRC_DIR/$ENV.h"
        if [ ! -f "$ENV_HEADER" ]; then
            echo "Error: no $ENV_HEADER; GPU-only envs have no CPU test build"
            exit 1
        fi
        STANDALONE_SOURCE="src/puffertest.c"
        STANDALONE_DEFINES=(
            "-DENV_HEADER=\"$ENV_HEADER\""
            "-DPUFFER_ENV_NAME=\"$ENV\""
        )
        # Only tests that use puf_test.h join the harness.
        TEST_FILES=()
        for f in "$ENV_DIR"/tests/*.c; do
            grep -qs 'puf_test\.h' "$f" && TEST_FILES+=("$f")
        done
        if [ ${#TEST_FILES[@]} -gt 0 ]; then
            TESTS_INC="$OUTPUT_NAME.inc"
            if [ -z "$DRY_RUN" ]; then
                printf '#include "%s"\n' "${TEST_FILES[@]}" > "$TESTS_INC"
            fi
            STANDALONE_DEFINES+=("-DPUF_TEST_INCLUDE=\"$TESTS_INC\"")
        fi
    elif [ -f "$SRC_FILE" ]; then
        # The env ships its own standalone main (osrs_* visual sim, nethack TTY demo).
        STANDALONE_SOURCE="$SRC_FILE"
    else
        ENV_HEADER="$SRC_DIR/$ENV.h"
        if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
            echo "Error: $ENV_HEADER must typedef obs_t for standalone eval"
            exit 1
        fi
        STANDALONE_DEFINES=(
            -DPUFFERCPU_EVAL_MAIN
            "-DENV_HEADER=\"$ENV_HEADER\""
            "-DPUFFER_ENV_NAME=\"$ENV\""
        )
    fi
    FLAGS=(
        -I. -Isrc -I"$SRC_DIR" -Ivendor "${INCLUDES[@]}"
        "$STANDALONE_SOURCE" "${EXTRA_SRC[@]}"
        -DPLATFORM_DESKTOP
        "${STANDALONE_DEFINES[@]}"
        "${ROOT_DEFINES[@]}"
        "${EXTRA_CFLAGS[@]}"
    )
    if [ -n "$CHECK" ]; then
        invoke ${CC:-clang} "${CLANG_OPT[@]}" -fsyntax-only "${FLAGS[@]}"
        echo "OK: $ENV"
        exit 0
    fi
    echo "Compiling $ENV..."
    invoke ${CC:-clang} "${CLANG_OPT[@]}" "${FLAGS[@]}" -o "$OUTPUT_NAME" \
        "${LINK_ARCHIVES[@]}" "${EXTRA_LDFLAGS[@]}" "${STANDALONE_LDFLAGS[@]}" \
        -lm -lpthread "${OMP_FLAGS[@]}"
    echo "Built: $OUTPUT_NAME"
    exit 0
elif [ "$MODE" = "web" ]; then
    if [ -f "$SRC_DIR/web.sh" ]; then
        if [ -n "$DRY_RUN" ]; then
            echo "Would run: $SRC_DIR/web.sh"
            exit 0
        fi
        source "$SRC_DIR/web.sh"
        exit 0
    fi
    ENV_HEADER="$SRC_DIR/$ENV.h"
    if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
        echo "Error: $ENV_HEADER must typedef obs_t for web eval"
        exit 1
    fi
    invoke mkdir -p "build/web/$ENV"
    WEBSITE_DIR="${PUFFER_WEBSITE_DIR:-../docker/puffer.ai}"
    PRELOAD_ENV=()
    if [ "$ENV" = "boxoban" ]; then
        # Do not pack generated boxoban_maps_*.bin or levels/ (hundreds of MB).
        PRELOAD_ENV=(
            --preload-file resources/boxoban/web_maps.bin@resources/boxoban/web_maps.bin
            --preload-file resources/boxoban/Wall_Black.jpg@resources/boxoban/Wall_Black.jpg
            --preload-file resources/boxoban/Crate_Black.jpg@resources/boxoban/Crate_Black.jpg
            --preload-file resources/boxoban/EndPoint_Black.jpg@resources/boxoban/EndPoint_Black.jpg
            --preload-file resources/boxoban/EndPoint_Blue.jpg@resources/boxoban/EndPoint_Blue.jpg
            --preload-file resources/boxoban/GroundGravel_Concrete.jpg@resources/boxoban/GroundGravel_Concrete.jpg
        )
    elif [ -d "resources/$ENV" ]; then
        # Env assets only. Policy weights live in the website repo.
        while IFS= read -r -d '' f; do
            rel="${f#resources/$ENV/}"
            case "$rel" in
                *weights*.bin|*_weights.bin) continue ;;
            esac
            PRELOAD_ENV+=(--preload-file "$f@resources/$ENV/$rel")
        done < <(find "resources/$ENV" -type f -print0)
    fi
    echo "Compiling $ENV for web..."
    PRELOAD=(
        --preload-file resources/shared@resources/shared
        --preload-file config/default.ini@config/default.ini
    )
    if [ -d "$SRC_DIR/generated" ]; then
        PRELOAD+=(--preload-file "$SRC_DIR/generated@ocean/$ENV/generated")
    fi
    if [ -n "$ENV_INI" ]; then
        PRELOAD+=(--preload-file "$ENV_INI@config/$ENV.ini")
    fi
    # Web overlays + 1M-cap policies live in the website repo.
    SITE_WEB_INI="$WEBSITE_DIR/config/${ENV}_web.ini"
    if [ -f "$SITE_WEB_INI" ]; then
        PRELOAD+=(--preload-file "$SITE_WEB_INI@config/${ENV}_web.ini")
    fi
    SITE_MODELS="$WEBSITE_DIR/docs/assets/models"
    if [ -f "$SITE_MODELS/${ENV}_web_weights.bin" ]; then
        PRELOAD+=(--preload-file "$SITE_MODELS/${ENV}_web_weights.bin@resources/$ENV/${ENV}_web_weights.bin")
        PRELOAD+=(--preload-file "$SITE_MODELS/${ENV}_web_weights.bin@resources/$ENV/${ENV}_weights.bin")
    elif [ -f "$SITE_MODELS/${ENV}_weights.bin" ]; then
        PRELOAD+=(--preload-file \
            "$SITE_MODELS/${ENV}_weights.bin@resources/$ENV/${ENV}_weights.bin")
    fi
    invoke emcc \
        -o "build/web/$ENV/game.html" \
        src/puffercpu.c "${EXTRA_SRC[@]}" \
        -O3 -Wall -Wno-narrowing -Wno-unreachable-code \
        "${LINK_ARCHIVES[@]}" \
        -I. -Isrc -I"$SRC_DIR" -Ivendor "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_NAME/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --js-library vendor/puf_web_vsync.js \
        --shell-file vendor/minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        -DPUFFERCPU_EVAL_MAIN \
        "-DENV_HEADER=\"$ENV_HEADER\"" \
        -DPUFFER_ENV_NAME=\"$ENV\" \
        --preload-file resources/shared@resources/shared \
        "${PRELOAD_ENV[@]}" \
        "${PRELOAD[@]}" \
        "${EXTRA_CFLAGS[@]}"
    echo "Built: build/web/$ENV/game.html"
    WEBSITE_ASSETS="$WEBSITE_DIR/docs/assets"
    if [ -d "$WEBSITE_ASSETS" ] && [ -z "$DRY_RUN" ]; then
        mkdir -p "$WEBSITE_ASSETS/$ENV"
        cp -a "build/web/$ENV/." "$WEBSITE_ASSETS/$ENV/"
        echo "Published: $WEBSITE_ASSETS/$ENV/"
    fi
    exit 0
fi

CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(which nvcc)")")}}
# NCCL include/lib fallback.
# Needed when NCCL is provided by the nvidia-nccl-cu12 wheel in the active venv.
NCCL_IFLAG=""
NCCL_LFLAG=""
for dir in /usr/include /usr/local/cuda/include; do
    if [ -f "$dir/nccl.h" ]; then NCCL_IFLAG="-I$dir"; break; fi
done
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    if [ -f "$dir/libnccl.so" ] || [ -f "$dir/libnccl.so.2" ]; then NCCL_LFLAG="-L$dir"; break; fi
done
if [ -z "$NCCL_IFLAG" ]; then
    NCCL_IFLAG=$(python -c "import nvidia.nccl, os; print('-I' + os.path.join(nvidia.nccl.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$NCCL_LFLAG" ]; then
    NCCL_LFLAG=$(python -c "import nvidia.nccl, os; print('-L' + os.path.join(nvidia.nccl.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CCACHE_BASEDIR="$ROOT"
export CCACHE_COMPILERCHECK=content
NVCC="ccache $CUDA_HOME/bin/nvcc"
CC="${CC:-$(command -v ccache >/dev/null && echo 'ccache clang' || echo 'clang')}"
ARCH=${NVCC_ARCH:-native}

# CPU and CUDA envs are separate sources. --cu selects the .cu; default is .h.
# Only one is compiled in (never both).
if [ "$USE_GPU_ENV" = "1" ]; then
    ENV_HEADER="$SRC_DIR/$ENV.cu"
    if [ ! -f "$ENV_HEADER" ]; then
        echo "Error: --cu requires $ENV_HEADER"
        exit 1
    fi
else
    ENV_HEADER="$SRC_DIR/$ENV.h"
fi
invoke mkdir -p build
if ! grep -q 'typedef[[:space:]].*obs_t' "$ENV_HEADER" 2>/dev/null; then
    echo "Error: $ENV_HEADER must typedef obs_t"
    exit 1
fi

ENV_COMPILE_FLAGS=("-DENV_HEADER=\"$ENV_HEADER\"")

# Brace-init narrowing (host -Wno-narrowing + nvcc #2361) and unreachable
# code in env headers (clang -Wunreachable-code, nvcc #111/#128).
NVCC_NARROW=(
    -Xcompiler=-Wno-narrowing
    -Xcompiler=-Wno-unreachable-code
    --diag-suppress=2361
    --diag-suppress=111
    --diag-suppress=128
)

if [ "$MODE" = "native" ]; then
    if [ -n "$OUT" ]; then
        TRAIN_BIN="$OUT"
    elif [ -n "$USER_OUTPUT_NAME" ]; then
        TRAIN_BIN="$USER_OUTPUT_NAME"
    else
        TRAIN_BIN="$START_DIR/puffer"
    fi
    if [ "$SNAKE_RAW" = "1" ]; then
        EXTRA_CFLAGS+=(-DSNAKE_ONEHOT=0)
    fi
    OSRS_RENDER_OBJECT=""
    case "$ENV" in
        osrs_*)
            OSRS_RENDER_OBJECT="build/osrs_puffer_render.o"
            ENV_COMPILE_FLAGS+=(-DOSRS_PUFFER_RENDER)
            invoke $CC $LINK_OPT "${CLANG_WARN[@]}" "${SIMD_FLAGS[@]}" -std=c11 \
                -Wno-unused-function \
                -D_POSIX_C_SOURCE=200809L \
                -I. -Isrc -I"$SRC_DIR" -Ivendor \
                "${INCLUDES[@]}" \
                -DPLATFORM_DESKTOP \
                -c ocean/osrs/osrs_puffer_render.c \
                -o "$OSRS_RENDER_OBJECT"
            ;;
    esac
    echo "Compiling $ENV_HEADER -> $TRAIN_BIN..."
    invoke $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I"$SRC_DIR" -Ivendor \
        "${INCLUDES[@]}" \
        -I$CUDA_HOME/include -I$CUDA_HOME/include/cccl $NCCL_IFLAG -I$RAYLIB_NAME/include \
	    "${ENV_COMPILE_FLAGS[@]}" \
	    -DENV_NAME=$ENV \
	    -DPUFFER_ENV_NAME=\"$ENV\" \
	    -DPUFFERLIB_BUILD_MAIN \
	    "${ROOT_DEFINES[@]}" \
	    -Xcompiler=-DPLATFORM_DESKTOP \
	    -Xcompiler=-fopenmp \
	    "${NVCC_NARROW[@]}" \
	    "${EXTRA_CFLAGS[@]}" \
	    $PRECISION \
	    src/pufferl.cu \
        "${EXTRA_SRC[@]}" \
        $OSRS_RENDER_OBJECT \
        "${LINK_ARCHIVES[@]}" \
        -L$CUDA_HOME/lib64 $NCCL_LFLAG \
        "${EXTRA_LDFLAGS[@]}" \
        -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
        -lm -Xlinker=-lpthread $OMP_LIB "${STANDALONE_LDFLAGS[@]}" \
        -o "$TRAIN_BIN"
    echo "Built: $TRAIN_BIN"

elif [ "$MODE" = "profile" ]; then
    PROFILE_BIN="$START_DIR/build/profile_$ENV"
    invoke mkdir -p "$START_DIR/build"
    echo "Compiling $ENV_HEADER -> $PROFILE_BIN..."
    invoke $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I"$SRC_DIR" -Ivendor \
        "${INCLUDES[@]}" \
        -I$CUDA_HOME/include -I$CUDA_HOME/include/cccl $NCCL_IFLAG -I$RAYLIB_NAME/include \
        "${ENV_COMPILE_FLAGS[@]}" \
        -DENV_NAME=$ENV \
	    -DPUFFER_ENV_NAME=\"$ENV\" \
        "${ROOT_DEFINES[@]}" \
        -Xcompiler=-DPLATFORM_DESKTOP \
	    "${NVCC_NARROW[@]}" \
	    "${EXTRA_CFLAGS[@]}" \
        $PRECISION \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu \
        "$RAYLIB_A" \
        -L$CUDA_HOME/lib64 \
        -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
        -lGL -lm -Xlinker=-lpthread $OMP_LIB \
        -o "$PROFILE_BIN"
    echo "Built: $PROFILE_BIN"
fi
