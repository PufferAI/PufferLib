#!/usr/bin/env bash
# bin/puffer launcher tests. Uses this checkout as the toolchain and a scratch
# PUFFER_HOME, so nothing under ~/.puffer is touched. CUDA paths use --dry-run.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export PUFFER_HOME="$TMP/home"
# file:// so a shallow checkout can still be cloned as a toolchain.
export PUFFER_REPO="file://$ROOT"
PUFFER="$ROOT/bin/puffer"
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

# has_arg FILE ARG: a dry run printed ARG as one shell word.
has_arg() {
    grep -qF -- "$(printf '%q' "$2")" "$1"
}

# scaffold NAME DIR: the template env renamed, at DIR/NAME.h and DIR/NAME.ini.
scaffold() {
    local camel
    camel="$(echo "${1:0:1}" | tr a-z A-Z)${1:1}"
    mkdir -p "$2"
    sed "s/Template/$camel/g; s/template/$1/g" "$ROOT/ocean/template/template.h" > "$2/$1.h"
    sed "s/template/$1/" "$ROOT/config/template.ini" > "$2/$1.ini"
}

export ROOT TMP PUFFER
export -f not has_arg
rm -rf "$ROOT/ocean/breakout/build"
mkdir -p "$TMP/empty"

# From an empty folder: the --cpu binary goes to the env's build/, the folder
# records the env, later commands need no env argument, and a build is only
# redone when something changed.
t "build, record and eval --cpu an Ocean env from an empty folder" <<'EOF'
cd "$TMP/empty"
"$PUFFER" build breakout --cpu
test -x "$ROOT/ocean/breakout/build/breakout"
grep -q "env = $ROOT/ocean/breakout" build/puffer.env
"$PUFFER" eval --cpu --headless --eval_episodes=1 | grep -q CPU_EVAL
"$PUFFER" build breakout --cpu | grep -qi 'up to date'
touch "$ROOT/ocean/breakout/build/breakout"
"$PUFFER" build breakout --cpu | grep -q Compiling
cp "$ROOT/config/breakout.ini" "$TMP/breakout.ini"
trap 'cp "$TMP/breakout.ini" "$ROOT/config/breakout.ini"' EXIT
echo '# edited' >> "$ROOT/config/breakout.ini"
"$PUFFER" build breakout --cpu | grep -q Compiling
"$PUFFER" build breakout --cu --dry-run | grep -q -- "-o $ROOT/ocean/breakout/build/puffer"
EOF
t "a failed build leaves the folder's record alone" <<'EOF'
cd "$TMP/empty"
not "$PUFFER" build template --cu
grep -q "env = $ROOT/ocean/breakout" build/puffer.env
EOF
t "eval --cpu runs the standalone and takes an env or a weights file" <<'EOF'
mkdir -p "$TMP/bare"
cd "$TMP/bare"
"$PUFFER" eval breakout --cpu --headless --eval_episodes=1 | grep -q CPU_EVAL
"$PUFFER" build breakout --cpu --debug > "$TMP/out.txt"
grep -q Compiling "$TMP/out.txt"
"$PUFFER" build breakout --cpu --dry-run > /dev/null
grep -q '^flags = --debug$' build/puffer.env
"$PUFFER" eval --cpu --headless --eval_episodes=1 | grep -q 'up to date'
"$PUFFER" eval latest --cpu --headless --eval_episodes=1 > "$TMP/out.txt" 2>&1 || true
not grep -q "no env 'latest'" "$TMP/out.txt"
EOF

# An out-of-tree env: build from inside it with no env argument, rebuild on change.
MY="$TMP/myenv"
scaffold myenv "$MY"
export MY
t "build, rebuild and eval --cpu inside an env directory" <<'EOF'
cd "$MY"
"$PUFFER" build --cpu
test -x build/myenv
echo '// touched' >> myenv.h
"$PUFFER" build --cpu | grep -q Compiling
"$PUFFER" eval --headless --cpu --eval_episodes=1 | grep -q CPU_EVAL
EOF
t "a directory named otherwise is an env through its NAME.ini" <<'EOF'
mkdir -p "$TMP/puffer-flat"
cd "$TMP/puffer-flat"
"$PUFFER" new flat .
"$PUFFER" build --cpu
test -x build/flat
"$PUFFER" eval --headless --cpu --eval_episodes=1 | grep -q 'up to date'
EOF

# Toolchains: a pinned rev this checkout is not at gets cloned into PUFFER_HOME;
# PUFFER_TOOLCHAIN overrides the pin; a checkout already at the rev is used as is.
OLD=$(git -C "$ROOT" rev-parse HEAD)
PIN="$TMP/pinned"
scaffold pinned "$PIN"
printf '\n[puffer]\nrev = %s\n' "$OLD" >> "$PIN/pinned.ini"
cp "$PUFFER" "$TMP/puffer-outside"
export OLD PIN
# A blobless partial clone cannot be cloned over file://.
if [ "$(git -C "$ROOT" config --get remote.origin.promisor 2>/dev/null)" = true ]; then
    echo "skip pinned rev installs a toolchain on first use (partial clone)"
else
t "pinned rev installs a toolchain on first use" <<'EOF'
cd "$PIN"
"$TMP/puffer-outside" build --dry-run > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DPUFFER_ROOT=\"$TMP/home/toolchains/$OLD\""
test -f "$TMP/home/toolchains/$OLD/build.sh"
BAD=0000000000000000000000000000000000000000
sed -i.bak "s/^rev = .*/rev = $BAD/" pinned.ini
not "$TMP/puffer-outside" build --dry-run
not "$TMP/puffer-outside" build --dry-run
test ! -e "$TMP/home/toolchains/$BAD"
mv pinned.ini.bak pinned.ini
EOF
fi
t "PUFFER_TOOLCHAIN and a checkout at the pinned rev take precedence" <<'EOF'
cd "$PIN"
PUFFER_TOOLCHAIN="$ROOT" "$PUFFER" build --dry-run > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DPUFFER_ROOT=\"$ROOT\""
sed -i.bak "s/^rev = .*/rev = $(git -C "$ROOT" rev-parse HEAD)/" pinned.ini
"$PUFFER" build --dry-run > "$TMP/out.txt"
has_arg "$TMP/out.txt" "-DPUFFER_ROOT=\"$ROOT\""
EOF

t "usage and missing env are errors with a message" <<'EOF'
not "$PUFFER" > "$TMP/out.txt" 2>&1
grep -qi usage "$TMP/out.txt"
"$PUFFER" help > /dev/null 2>&1
mkdir -p "$TMP/nothing"
cd "$TMP/nothing"
not "$PUFFER" train > "$TMP/out.txt" 2>&1
grep -qi 'no env' "$TMP/out.txt"
cp "$ROOT/config/breakout.ini" .
not "$PUFFER" train > "$TMP/out.txt" 2>&1
grep -qi 'no env' "$TMP/out.txt"
not "$PUFFER" build no_such_env --cpu
mkdir -p "$TMP/old/ocean" "$TMP/old/src"
touch "$TMP/old/build.sh" "$TMP/old/src/pufferenv.h"
PUFFER_TOOLCHAIN="$TMP/old" "$PUFFER" build breakout --dry-run > "$TMP/out.txt" 2>&1 || true
grep -q predates "$TMP/out.txt"
EOF

# new scaffolds an env from ocean/template; check compiles its header.
t "new scaffolds a renamed template that check accepts" <<'EOF'
cd "$TMP"
"$PUFFER" new fresh
grep -q 'Fresh\*' fresh/fresh.h
not grep -qi template fresh/fresh.h
grep -Eq '^rev = [0-9a-f]{40}$' fresh/fresh.ini
grep -q '^build/' fresh/.gitignore
not "$PUFFER" new fresh
cd fresh
"$PUFFER" check
cp fresh.h "$TMP/orig.h"
trap 'cp "$TMP/orig.h" fresh.h' EXIT
perl -ni -e 'print unless /#define OBS_SIZE/' fresh.h
not "$PUFFER" check > "$TMP/out.txt" 2>&1
grep -q OBS_SIZE "$TMP/out.txt"
cp "$TMP/orig.h" fresh.h
perl -pi -e 's/    float n;/    int n;/' fresh.h
not "$PUFFER" test --test.steps=10 > "$TMP/out.txt" 2>&1
grep -q 'Log.n must be a float' "$TMP/out.txt"
EOF

# A stub toolchain whose build.sh only writes a no-op binary, so the trainer
# path can be exercised without nvcc: one build, then train and eval reuse it.
STUB="$TMP/stub"
mkdir -p "$STUB/ocean/breakout"
cat > "$STUB/build.sh" <<'SH'
#!/usr/bin/env bash
# The launcher accepts a toolchain whose build.sh mentions --dry-run.
echo "Compiling $*"
mkdir -p "$(dirname "$2")"
printf '#!/bin/sh\nexit 0\n' > "$2"
chmod +x "$2"
SH
chmod +x "$STUB/build.sh"
export STUB
t "build, train and eval share one trainer binary" <<'EOF'
mkdir -p "$TMP/stubrun"
cd "$TMP/stubrun"
export PUFFER_TOOLCHAIN="$STUB"
"$PUFFER" build breakout | grep -q Compiling
"$PUFFER" train | grep -q 'up to date'
"$PUFFER" eval | grep -q 'up to date'
"$PUFFER" build | grep -q 'up to date'
EOF

# test: sanitiser build of src/puffertest.c in the env's build/, arguments
# passed through; 'all' sweeps the toolchain's ocean/ and summarises.
t "test ENV builds the harness and passes arguments through" <<'EOF'
cd "$TMP/empty"
"$PUFFER" test breakout --test.steps=200 | grep -q passed
test -x "$ROOT/ocean/breakout/build/test"
cd "$MY"
"$PUFFER" test --test.steps=50 --test.seeds=1 | grep -q '1 seeds x 50 steps'
EOF
# A fake toolchain: the template, an env that never ends an episode, and one
# that opts out with [test] skip = 1.
TC="$TMP/tc"
mkdir -p "$TC/ocean" "$TC/config"
cp -R "$ROOT/src" "$TC/src"
cp "$ROOT/build.sh" "$TC/build.sh"
cp "$ROOT/config/default.ini" "$ROOT/config/template.ini" "$TC/config/"
ln -s "$(ls -d "$ROOT"/raylib-5.5_* | head -1)" "$TC/"
cp -R "$ROOT/ocean/template" "$TC/ocean/template"
scaffold stuck "$TC/ocean/stuck"
sed -i.bak 's/if (env->x == env->goal) {/if (0) {/; '\
's/} else if (env->x == -env->goal) {/} else if (0) {/' "$TC/ocean/stuck/stuck.h"
mv "$TC/ocean/stuck/stuck.ini" "$TC/config/stuck.ini"
scaffold skipped "$TC/ocean/skipped"
printf '\n[test]\nskip = 1\n' >> "$TC/ocean/skipped/skipped.ini"
mv "$TC/ocean/skipped/skipped.ini" "$TC/config/skipped.ini"
export TC
t "test all sweeps the toolchain, skips opted-out envs and fails on the broken one" <<'EOF'
cd "$TMP/empty"
not env PUFFER_TOOLCHAIN="$TC" "$PUFFER" test all --test.steps=100 > "$TMP/all.out" 2>&1
grep -q '^ok   template' "$TMP/all.out"
grep -q '^FAIL stuck' "$TMP/all.out"
grep -q '^skip skipped' "$TMP/all.out"
grep -q '1 of 2 envs failed' "$TMP/all.out"
EOF

rm -rf "$ROOT/ocean/breakout/build"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
