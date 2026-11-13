#!/usr/bin/env bash
# build.sh tests: path resolution, [build] hooks, output locations, ini.h roots,
# --check and --test. Needs clang and the raylib download (build.sh
# fetches it). CUDA paths are checked with --dry-run only.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0
LOG="$TMP/last.log"

# t NAME reads the case from stdin and runs it with -e: every line must succeed.
t() {
    if bash -e > "$LOG" 2>&1; then
        PASS=$((PASS + 1))
        echo "ok   $1"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL $1"
        sed 's/^/     /' "$LOG" | tail -20
    fi
}

# not CMD...: the command must fail.
not() {
    if "$@"; then
        echo "expected to fail: $*"
        return 1
    fi
}

# dry ENV FLAGS...: the compile commands, without running them.
dry() {
    (cd "$TMP" && "$ROOT/build.sh" "$@" --dry-run)
}

# has_arg FILE ARG: a dry run printed ARG as one shell word.
has_arg() {
    grep -qF -- "$(printf '%q' "$2")" "$1"
}

# fixture NAME SED: the template env with one edit spliced in, at $TMP/fx/NAME.
fixture() {
    mkdir -p "$TMP/fx/$1"
    sed "s/Template/Fx/g; s/template/$1/g; $2" "$ROOT/ocean/template/template.h" > "$TMP/fx/$1/$1.h"
    sed "s/template/$1/" "$ROOT/config/template.ini" > "$TMP/fx/$1/$1.ini"
}

# fails FIXTURE WORD: the fixture's test binary fails and its output names the problem.
fails() {
    (cd "$TMP" && "$ROOT/build.sh" "$TMP/fx/$1" --test)
    not "$TMP/build/test_$1" > "$TMP/fx/$1.out" 2>&1
    grep -qi "$2" "$TMP/fx/$1.out" || { echo "expected '$2' in:"; cat "$TMP/fx/$1.out"; return 1; }
}

export ROOT TMP
export -f not dry has_arg fixture fails
rm -rf "$ROOT/ocean/breakout/build"

# A bare name resolves to ocean/NAME relative to the script, not to $PWD; the
# binary lands in the current directory as always (./breakout, ./puffer); stock
# macOS bash 3.2 must do.
t "builds ./ENV in the current directory from anywhere" <<'EOF'
cd "$TMP"
/bin/bash "$ROOT/build.sh" breakout --cpu
test -x "$TMP/breakout"
test ! -e "$ROOT/ocean/breakout/build/breakout"
dry "$ROOT/ocean/breakout" out/bk --cpu | grep -q -- "-o $TMP/out/bk"
not "$ROOT/build.sh" no_such_env --cpu
not "$ROOT/build.sh" ./no_such_dir --cpu
EOF
# A path is a path even when an Ocean env has the same name; a name must be
# usable as a C identifier.
fixture breakout ''
fixture bad.name ''
t "a ./path never resolves to an Ocean env, and names must be identifiers" <<'EOF'
cd "$TMP/fx"
"$ROOT/build.sh" ./breakout --cpu --dry-run > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DENV_HEADER=\"$TMP/fx/breakout/breakout.h\""
not "$ROOT/build.sh" "$TMP/fx/bad.name" --check > "$TMP/out.txt" 2>&1
grep -q identifier "$TMP/out.txt"
EOF
# A lone ini beside the directory's own header does not rename it (ocean/nethack/depth.ini).
fixture named ''
mkdir "$TMP/fx/named/src"
mv "$TMP/fx/named/named.h" "$TMP/fx/named/src/"
mv "$TMP/fx/named" "$TMP/fx/puffer-named"
fixture stray ''
mv "$TMP/fx/stray/stray.ini" "$TMP/fx/stray/other.ini"
t "the one NAME.ini names a directory called otherwise; NAME.h may sit in src/" <<'EOF'
cd "$TMP"
"$ROOT/build.sh" fx/puffer-named --cpu --dry-run > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DENV_HEADER=\"$TMP/fx/puffer-named/src/named.h\""
"$ROOT/build.sh" fx/stray --cpu --dry-run > "$TMP/out.txt"
has_arg "$TMP/out.txt" '-DPUFFER_ENV_NAME="stray"'
EOF

# An out-of-tree env with every [build] hook kind; [build.cpu] applies to the
# cpu build only.
HOOKED="$TMP/hooked"
mkdir -p "$HOOKED/inc" "$HOOKED/extra"
sed 's/Template/Hooked/g; s/template/hooked/g' "$ROOT/ocean/template/template.h" \
    > "$HOOKED/hooked.h"
sed -i.bak 's|#include "pufferenv.h"|#include "pufferenv.h"\n#include "hook.h"\n'\
'#if HOOK_FLAG != 3\n#error HOOK_FLAG not applied\n#endif\n'\
'#ifndef HOOK_MODE_CPU\n#error [build.cpu] cflags not applied\n#endif|; '\
's|env->x = 0;|env->x = hook_value();|' \
    "$HOOKED/hooked.h"
echo 'int hook_value(void);' > "$HOOKED/inc/hook.h"
echo 'int hook_value(void) { return 0; }' > "$HOOKED/extra/hook.c"
sed 's/template/hooked/' "$ROOT/config/template.ini" > "$HOOKED/hooked.ini"
cat >> "$HOOKED/hooked.ini" <<INI

[build]
flag = 3                      # unknown keys are variables for later values
include = inc
src = extra/hook.c
cflags = -DHOOK_FLAG=\$flag -DHOOK_STR="a b"
run = touch \$ENV_DIR/ran-\$ENV.txt && touch \$ENV_DIR/ran-twice.txt

[build.cpu]
cflags = -DHOOK_MODE_CPU
INI
export HOOKED
t "out-of-tree env with [build] hooks" <<'EOF'
mkdir -p "$TMP/run"
cd "$TMP/run"
"$ROOT/build.sh" "$HOOKED" --cpu
test -x "$TMP/run/hooked"
test -f "$HOOKED/ran-hooked.txt"
test -f "$HOOKED/ran-twice.txt"
dry "$HOOKED" > "$TMP/out.txt"
not grep -q -- -DHOOK_MODE_CPU "$TMP/out.txt"
EOF

# The special cases that used to live in build.sh now come from config/*.ini.
t "per-env build steps come from the env ini" <<'EOF'
CLIFFORD_N_QUBITS=4 dry clifford --cpu | grep -Eq -- '-DCLIFFORD_N_QUBITS=4( |$)'
dry impulse_wars > "$TMP/out.txt"
grep -q impulse_wars_api.c "$TMP/out.txt"
grep -q 'box2d-.*/libbox2d.a' "$TMP/out.txt"
dry impulse_wars --cpu > "$TMP/out.txt"
not grep -q impulse_wars_api.c "$TMP/out.txt"
dry nethack | grep -q -- -lnethack
dry nethack --cpu | grep -q ocean/nethack/nethack.c
dry osrs_inferno --cpu | grep -q ocean/osrs_inferno/osrs_inferno.c
EOF

t "trainer dry-run: absolute header, output in cwd, config roots" <<'EOF'
dry breakout --cu > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DENV_HEADER=\"$ROOT/ocean/breakout/breakout.cu\""
grep -q -- "-o $TMP/puffer" "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DPUFFER_ROOT=\"$ROOT\""
has_arg "$TMP/out.txt" "-DPUFFER_ENV_DIR=\"$ROOT/ocean/breakout\""
EOF

# ini.h reads default.ini from PUFFER_ROOT and the env ini from PUFFER_ENV_DIR;
# puf_resource looks in the env's resources/ first, then the checkout's.
mkdir -p "$TMP/root/config" "$TMP/envdir/resources" "$TMP/root/resources/shared" "$TMP/elsewhere"
printf '[base]\nenv_name = default\n[env]\nsize = 1\nonly_in_default = 7\n' \
    > "$TMP/root/config/default.ini"
printf '[base]\nenv_name = oceanenv\n[env]\nsize = 2\n' > "$TMP/root/config/oceanenv.ini"
printf '[base]\nenv_name = myenv\n[env]\nsize = 3\n' > "$TMP/envdir/myenv.ini"
touch "$TMP/envdir/resources/own.png" "$TMP/root/resources/shared/x.png"
t "ini.h and puf_resource resolve from the compiled-in roots" <<'EOF'
raylib=$(ls -d "$ROOT"/raylib-5.5_*/include | head -1)
defs=(-DPUFFER_ROOT="\"$TMP/root\"" -DPUFFER_ENV_DIR="\"$TMP/envdir\"")
clang -I"$ROOT/src" "${defs[@]}" "$ROOT/tests/test_ini.c" -o "$TMP/test_ini"
clang -I"$ROOT/src" -I"$raylib" "${defs[@]}" "$ROOT/tests/test_resource.c" -o "$TMP/test_resource"
cd "$TMP/elsewhere"
"$TMP/test_ini"
"$TMP/test_resource"
EOF

# --check compiles the header with the cpu build's flags and writes nothing.
fixture broken 's/#define OBS_SIZE 1/#define OBS_SIZE 1\nint missing_semicolon/'
fixture spaced ''
mkdir -p "$TMP/with space"
mv "$TMP/fx/spaced" "$TMP/with space/spaced"
printf 'int spaced_extra(void) { return 1; }\n' > "$TMP/with space/spaced/extra.c"
printf 'int spaced_more(void) { return 2; }\n' > "$TMP/with space/spaced/more.c"
printf '\n[build]\nsrc = extra.c\n[build.cpu]\nsrc = "$ENV_DIR/more.c"\n' \
    >> "$TMP/with space/spaced/spaced.ini"
rm -f "$TMP/breakout"
t "--check passes a good env, fails a broken one, writes no binary" <<'EOF'
cd "$TMP"
"$ROOT/build.sh" breakout --check
test ! -e "$TMP/breakout"
"$ROOT/build.sh" "$TMP/with space/spaced" --check
not "$ROOT/build.sh" "$TMP/fx/broken" --check
dry "$TMP/with space/spaced" > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DENV_HEADER=\"$TMP/with space/spaced/spaced.h\""
dry "$TMP/with space/spaced" --web > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DENV_HEADER=\"$TMP/with space/spaced/spaced.h\""
"$ROOT/build.sh" "$TMP/with space/spaced" --cpu
test -x "$TMP/spaced"
EOF

# --dry-run writes nothing: no build/web, no publish into the website checkout,
# and --check on a standalone app is an error rather than a real build.
# Web builds land in the checkout's build/web whatever the cwd, so that is
# where a stray build would show up.
rm -rf "$ROOT/build/web/breakout"
t "--dry-run and --check have no side effects" <<'EOF'
mkdir -p "$TMP/site/docs/assets"
cd "$TMP"
PUFFER_WEBSITE_DIR="$TMP/site" "$ROOT/build.sh" breakout --web --dry-run > "$TMP/out.txt"
not grep -q Published "$TMP/out.txt"
test ! -e "$TMP/site/docs/assets/breakout"
test ! -e "$ROOT/build/web/breakout"
dry trailer > "$TMP/out.txt"
git -C "$ROOT" diff --quiet -- resources/trailer/shots
not "$ROOT/build.sh" constellation --check
not "$ROOT/build.sh" constellation --test
"$ROOT/build.sh" breakout --check --web
test ! -e "$ROOT/build/web/breakout"
dry nethack --web | grep -q 'Would run: .*web.sh'
EOF

# --test builds src/puffertest.c against the env with sanitisers; the binary
# runs generic checks over random actions and fails on the classic env bugs.
t "--test builds the harness with sanitisers and passes breakout" <<'EOF'
dry breakout --test > "$TMP/out.txt"
grep -q src/puffertest.c "$TMP/out.txt"
grep -q -- -fsanitize=address "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DENV_HEADER=\"$ROOT/ocean/breakout/breakout.h\""
cd "$TMP"
"$ROOT/build.sh" breakout --test
"$TMP/build/test_breakout" --test.steps=500 --test.seeds=1 | grep -q '1 seeds x 500 steps'
EOF
fixture overflow 's|env->x = 0;|env->x = 0; env->agents[0].observations[OBS_SIZE] = 1;|'
fixture noreset 's|if (env->x == env->goal) {|if (0) {|; '\
's|} else if (env->x == -env->goal) {|} else if (0) {|'
fixture hiddenstate 's|env->size = dict_get(kwargs, "size");|static int inits; '\
'env->size = dict_get(kwargs, "size") + inits++;|'
fixture signedub 's|env->x = 0;|env->x = 0; { volatile int big = 2147483647; big += 1; }|'
fixture intlog 's|    float perf;|    int perf;|'
t "test catches an overflow, a missing reset, hidden state, UB and an int Log field" <<'EOF'
fails overflow overflow
fails noreset episode
fails hiddenstate determin
fails signedub 'runtime error'
fails intlog 'an int in Log'
EOF
# An env whose puf_close writes a buffer (impulse_wars) and frees Agent* agents (nmmo3).
fixture ptragents 's|Agent agents\[1\];|Agent* agents;|; '\
's|env->num_agents = 1;|env->num_agents = 1; env->agents = calloc(1, sizeof(Agent));|; '\
's|void puf_close(Fx\* env) {|void puf_close(Fx* env) { '\
'env->agents[0].terminals[0] = 0; free(env->agents);|'
t "test survives a puf_close that writes the buffers and frees Agent* agents" <<'EOF'
cd "$TMP"
"$ROOT/build.sh" "$TMP/fx/ptragents" --test
"$TMP/build/test_ptragents" --test.seeds=1 --test.steps=50 | grep -q passed
EOF

# Env-specific tests: tests/*.c next to the header, compiled in with puf_test.h
# and run after the generic checks. Standalone programs there are left alone.
fixture withtests 's|#include "pufferenv.h"|#include "pufferenv.h"\nstatic int fail = 3; '\
'static void warn(void) {} char* read_file(const char* p);|'
mkdir -p "$TMP/fx/withtests/tests"
printf '#include "withtests.h"\nint main(void) { return 0; }\n' \
    > "$TMP/fx/withtests/tests/own_main.c"
cat > "$TMP/fx/withtests/tests/goal.c" <<'C'
#include "puf_test.h"

PUF_TEST(goal_is_reached_by_walking_toward_it) {
    Env* env = puf_test_env(7);
    for (int i = 0; i < env->size; i++) {
        env->agents[0].actions[0] = env->goal > 0 ? 1 : 0;
        puf_test_step(env);
    }
    PUF_CHECK(env->agents[0].terminals[0] == 1);
    PUF_CHECK(env->agents[0].rewards[0] == 1.0f);
    puf_test_close(env);
}
C
t "env tests run after the generic checks and a failing check is named" <<'EOF'
cd "$TMP"
"$ROOT/build.sh" "$TMP/fx/withtests" --test
"$TMP/build/test_withtests" --test.seeds=1 --test.steps=50 > "$TMP/out.txt"
grep -q '^ok   goal_is_reached_by_walking_toward_it' "$TMP/out.txt"
grep -q '1 env tests' "$TMP/out.txt"
printf '#include "puf_test.h"\nPUF_TEST(wrong) { Env* env = puf_test_env(1); '\
'PUF_CHECK(env->size == -1); puf_test_close(env); }\n' > "$TMP/fx/withtests/tests/wrong.c"
fails withtests 'wrong.c:2: env->size == -1'
rm "$TMP/build/test_withtests"
"$ROOT/build.sh" "$TMP/fx/withtests" --check --test
test ! -e "$TMP/build/test_withtests"
mkdir -p "$TMP/with space/spaced/tests"
cp "$TMP/fx/withtests/tests/goal.c" "$TMP/with space/spaced/tests/goal.c"
"$ROOT/build.sh" "$TMP/with space/spaced" --test
"$TMP/build/test_spaced" --test.seeds=1 --test.steps=50 | grep -q '1 env tests'
EOF

rm -rf "$ROOT/ocean/breakout/build" "$ROOT/ocean/template/build"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
