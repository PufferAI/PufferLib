# Pathfinder Bloat Reduction TODO

Track simplifications to reduce unnecessary complexity and keep the env lean.

## High-priority refactors

- [x] Fix random action generation in player mode
  - `pathfinder_rand` now returns `unsigned int`, and action sampling uses unsigned modulo.
  - This removes potential negative casts before indexing actions.

- [ ] Remove curriculum helper asymmetry in `pathfinder.h`
  - [curriculum_min_solution_len](ocean/pathfinder/pathfinder.h:139) still does fallback/clamp sequencing while max length logic is already direct.
  - [next_curriculum_max_solution_len](ocean/pathfinder/pathfinder.h:582) duplicates curriculum clamping logic.
  - Simplify to direct env-invariant behavior where possible.

- [ ] Remove clamp/safety fallback patterns that are no longer needed
  - [pathfinder_clamp_int](ocean/pathfinder/pathfinder.h:15) and related call sites were added for defensive fault handling.
  - Decide if we trust validated INI/env invariants and drop this extra layer where possible.

- [ ] Simplify config initialization default fallback chains
  - [init defaults](ocean/pathfinder/pathfinder.h:627) has many `value == 0 ? default : value` branches for reward and tuning parameters.
  - Make defaults explicit and avoid repeated defensive defaulting.

- [ ] Simplify maze generation fallback flow
  - [generate_maze](ocean/pathfinder/pathfinder.h:544) includes multi-step rollback behavior (try A, then B, then hard reset).
  - Collapse to a single, deterministic path length selection approach.

- [ ] Reduce duplication in random edge opening
  - [open_random_edges](ocean/pathfinder/pathfinder.h:492) has duplicated horizontal/vertical loops with repeated shortest-path checks.
  - Consolidate edge-try logic into a shared path.

- [ ] Refactor `c_step` control flow
  - [c_step](ocean/pathfinder/pathfinder.h:689) has deep nested branching for action validity, bounds, wall checks, movement, penalties, and terminal handling.
  - Flatten into early-return branches for readability and lower cognitive load.

- [ ] Consolidate reset logic
  - [c_reset](ocean/pathfinder/pathfinder.h:650) and [reset_attempt](ocean/pathfinder/pathfinder.h:665) overlap heavily.
  - Extract a shared reset initializer with a small mode flag for map regeneration vs attempt retry.

## Medium-priority cleanup

- [ ] Reduce overlong/duplicated rendering code blocks in `pathfinder.h`
  - Render path in the same file is dense and mixed with gameplay-specific logic; split into clearer helpers.

- [ ] Reconcile thin “micro-wrapper” helpers
  - Some tiny wrappers (wall access/action helpers) remain after earlier cleanup; remove ones that don’t add meaningful abstraction.

## Test file cleanup

- [ ] Review `test_pathfinder_core.c` for repetitive setup/assert patterns
  - [test file is large](ocean/pathfinder/tests/test_pathfinder_core.c:1).
  - Consider shared maze builders/helpers to reduce repeated scenario boilerplate and keep tests dense but readable.
