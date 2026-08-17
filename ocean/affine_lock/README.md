# affine_lock

`affine_lock` is a single-agent 16-bit state-matching environment. Each episode
starts from a current bit state and a target bit state. The agent applies one of
eight reversible bit transforms until the current state equals the target.

The committed training path uses the generated visible-target table:

```text
ocean/affine_lock/generated/affine_lock_8action_visible_targets.bin
```

That table is loaded at reset time and provides exact start/target pairs for the
curriculum depths configured in `config/affine_lock.ini`.

## Runtime Action Set

The runtime environment uses the committed 8-action set. The generator and
manifest identify this exact transform set as `affine_lock_8action_v1`:

| Id | Name | Effect |
| ---: | --- | --- |
| `0` | `shift_left` | rotate bit positions left |
| `1` | `shift_right` | rotate bit positions right |
| `2` | `invert_right_7` | flip bits `9..15` |
| `3` | `swap_adjacent_bits` | swap each adjacent bit pair |
| `4` | `swap_adjacent_pairs` | swap each adjacent two-bit pair |
| `5` | `swap_nibbles_each_byte` | swap low/high nibbles within each byte |
| `6` | `reverse_each_nibble` | reverse bit order within each nibble |
| `7` | `reverse_each_byte` | reverse bit order within each byte |

The Puffer binding exposes one discrete action slot with
`NUM_ACTIONS = 8`.

## Resets

Resets always sample from the committed visible-target table. To train or test
on different target distributions, generate a new table with the tool below and
point `VISIBLE_TARGET_TABLE_PATH` at it when building.

## Committed Target Table

The committed table stores sampled visible start/target pairs at depths `2`,
`4`, `5`, `6`, and `8`, plus every known true depth-16 pair for this action
set.

| Depth | True visible pairs | Stored records |
| ---: | ---: | ---: |
| `2` | `2,216,496` | `65,536` |
| `4` | `34,379,722` | `65,536` |
| `5` | `115,388,932` | `65,536` |
| `6` | `331,789,220` | `65,536` |
| `8` | `1,125,374,770` | `65,536` |
| `16` | `100,548` | `100,548` |

The table format can store any depth sections, but this generator currently
targets the fixed depth list `{2, 4, 5, 6, 8, 16}`. The runtime `seed` controls
the episode sequence sampled from a loaded table. The generator's
`--sample-seed` controls which sampled depth-2/4/5/6/8 records are written into
a custom table. Depth 16 is stored in full for the committed 8-action set, so
changing `--sample-seed` does not change the depth-16 records.

## Regenerating the Target Table

If the generated binary artifact is omitted from a checkout, regenerate the
default table from the repo root:

```bash
gcc -std=c11 -O3 -DNDEBUG -fopenmp \
  -I. -Iocean/affine_lock \
  ocean/affine_lock/tools/generate_8action_visible_targets.c \
  -lm -o /tmp/affine_lock_generate_visible_targets

/tmp/affine_lock_generate_visible_targets
```

The no-argument generator run writes the default `.bin` and `.json` files under
`ocean/affine_lock/generated/`. The default sample seed is `0`, which preserves
the committed benchmark table. Changing the committed `.bin` changes the
training data and can change full-run `perf`, so regenerate and benchmark before
committing a replacement table.

### Using a Custom 8-Action Table

The same generator can create larger or seed-varied tables for the committed
8-action environment without changing the runtime action set:

```bash
/tmp/affine_lock_generate_visible_targets \
  --sample-seed 42 \
  --sample-per-depth 131072 \
  --store-all-depth 16 \
  --output-bin /tmp/affine_lock_8action_visible_targets_seed42.bin \
  --output-json /tmp/affine_lock_8action_visible_targets_seed42.json
```

Increasing `--sample-per-depth` raises the number of stored records for sampled
depths. `--store-all-depth D` stores every exact pair for a supported target
depth. For the committed 8-action set, depth 16 is stored in full by default.
Using the same `--sample-seed` and options produces the same table; using a
different seed produces a different sampled d2/d4/d5/d6/d8 table while leaving
stored-all depths unchanged.

To train against a custom 8-action table, either write it to the default path or
build with an explicit table path:

```bash
EXTRA_CFLAGS='-DVISIBLE_TARGET_TABLE_PATH="/tmp/affine_lock_8action_visible_targets_seed42.bin"' \
  ./build.sh affine_lock
```

The loader checks that the table action-set hash matches the runtime action
set. For seed-varied or larger 8-action tables, no runtime code changes are
needed as long as the table contains the curriculum depths requested by the
runtime.

The generator currently uses one `--sample-per-depth` value for all sampled
depths. If a future benchmark wants asymmetric budgets such as fewer d2/d4
records and more d6/d8 records, update the generator sampling options and
manifest/tests together, then regenerate and benchmark the replacement table.

To generate train/test table variants, keep the same depth/count settings and
change only `--sample-seed` and the output paths:

```bash
/tmp/affine_lock_generate_visible_targets \
  --sample-seed 42 \
  --sample-per-depth 65536 \
  --store-all-depth 16 \
  --output-bin /tmp/affine_lock_train_seed42.bin \
  --output-json /tmp/affine_lock_train_seed42.json

/tmp/affine_lock_generate_visible_targets \
  --sample-seed 69 \
  --sample-per-depth 65536 \
  --store-all-depth 16 \
  --output-bin /tmp/affine_lock_test_seed69.bin \
  --output-json /tmp/affine_lock_test_seed69.json
```

### Dropping the Committed Binary

The `.bin` is committed so the env works immediately and benchmark runs are
byte-for-byte reproducible. If the binary is removed from a branch, users must
run the no-argument generator before building/training:

```bash
/tmp/affine_lock_generate_visible_targets
./build.sh affine_lock
python -m pufferlib.pufferl train affine_lock
```

This recreates the default table at the path expected by the runtime. The
matching `.json` manifest records the depth counts, checksum, action-set hash,
and generator options.

## Experimental 4-Action Generator Set

The generator also includes an experimental `affine_lock_4action_v1` action set:

```text
shift_right
mirror
invert_right_7
swap_adjacent_bits
```

This is generator-only. The committed runtime environment does not train on this
action set. It is kept as a small, explicit alternate because a four-action
policy can be easier to learn, and this graph has far more unique depth-16
pairs than the committed 8-action table. To make it a runtime environment,
update the env action table, `NUM_ACTIONS`, the visible-table
action-set hash/path, generated table artifact, and any policy/config
expectations that assume eight actions.

The current true visible-pair counts for this generator action set are:

| Depth | True visible pairs |
| ---: | ---: |
| `2` | `772,080` |
| `4` | `6,055,652` |
| `5` | `16,234,512` |
| `6` | `42,176,998` |
| `8` | `234,409,780` |
| `16` | `2,434,606` |

Example generation command:

```bash
/tmp/affine_lock_generate_visible_targets \
  --action-set affine_lock_4action_v1 \
  --sample-per-depth 65536 \
  --store-all-depth 16 \
  --output-bin /tmp/affine_lock_4action_visible_targets.bin \
  --output-json /tmp/affine_lock_4action_visible_targets.json
```

### Making 4-Action a Runtime Env

The 4-action table is not plug-compatible with the committed 8-action runtime.
To make a real 4-action runtime variant:

1. Change `NUM_ACTIONS` to `4`.
2. Change the runtime action enum/table in `affine_lock.h` to match the
   generator's `affine_lock_4action_v1` order.
3. Point `VISIBLE_TARGET_TABLE_PATH` at a 4-action table.
4. Update the expected action-set hash in `affine_lock_visible_targets.h` to
   the 4-action manifest's `action_set_hash`.
5. Remove runtime helpers and render labels that only exist for the old
   8-action table.
6. Update policy/config/test assumptions that expect eight actions. In
   particular, the old all-actions-have-one-step-inverses test is
   8-action-specific because `shift_right` no longer has `shift_left` as an
   action. Replace it with checks that match the new action cycles and refresh
   the deterministic golden checksum.
7. Rebuild, run `ocean/affine_lock/tests/run_all.sh`, and rerun a full
   benchmark train.

## Adding New Depths Later

Adding another depth such as `7`, `10`, or `12` is intentionally not part of the
committed runtime path, but the file format can represent it. A future change
would need to:

1. Add the depth to `TARGET_DEPTHS` in
   `tools/generate_8action_visible_targets.c`.
2. Regenerate the `.bin` and `.json`.
3. Add the depth to `CURRICULUM_DEPTHS` and update
   `CURRICULUM_DEPTH_COUNT`.
4. Add matching `Log.depth_D_rate` and `Log.depth_D_solve_rate` fields plus
   `my_log` exports if the depth should appear in training logs.
5. Update config/docs/tests to expect the new depth and record count.
6. Rerun the affine tests and a full training benchmark.

The loader itself does not require a format change for additional depth
sections. If a new table omits a runtime-requested curriculum depth, reset will
abort because there is no valid record pool for that depth.
