#!/bin/bash
#
# Usage: ./profile.sh [timing|nsys|all]
#   timing - Run all profiles without nsys (fast, just timings)
#   nsys   - Run composite operations with nsys kernel breakdown
#   all    - Run both (default)

MODE=${1:-all}
BIN="./profile_torch"

if [[ ! -x "$BIN" ]]; then
    echo "Error: $BIN not found. Build with:"
    echo "  python setup.py build_profiler"
    exit 1
fi

# Check if envspeed is compiled in
HAS_ENV=0
"$BIN" envspeed 2>&1 | grep -q "Unknown profile" || HAS_ENV=1

run_timing() {
    local name=$1
    echo "--- $name ---"
    "$BIN" "$name"
}

run_nsys() {
    local name=$1
    echo "=== $name (nsys kernel breakdown) ==="
    nsys profile \
        --capture-range=cudaProfilerApi \
        --cuda-graph-trace=node \
        --stats=false \
        --force-overwrite=true \
        -o nprof-"$name" \
        "$BIN" "$name" 2>&1 | grep -v "^Generating\|^Processing"

    nsys stats --report cuda_gpu_kern_sum:base --force-export=true nprof-"$name".nsys-rep 2>/dev/null | tail -n +4
    echo ""
}

if [[ "$MODE" == "timing" || "$MODE" == "all" ]]; then
    echo "========== TIMING ONLY =========="
    echo ""
    run_timing kernels
    run_timing forwardcall
    run_timing rolloutcopy
    run_timing trainforward
    run_timing trainstep
    if [[ $HAS_ENV -eq 1 ]]; then
        run_timing envspeed
    fi
fi

if [[ "$MODE" == "nsys" || "$MODE" == "all" ]]; then
    echo "========== NSYS KERNEL BREAKDOWN =========="
    echo ""
    run_nsys forwardcall
    run_nsys rolloutcopy
    run_nsys trainforward
    run_nsys trainstep
    if [[ $HAS_ENV -eq 1 ]]; then
        run_nsys envspeed
    fi
fi
