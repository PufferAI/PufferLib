#pragma once

// Optional compile-time environment/trainer contract. Define hooks in ENV_HEADER.
// Omitting a hook preserves the standard trainer path. Environment callbacks may
// be defined in PUF_ENV_TRAINER_HEADER, included after policy/self-play APIs exist.
//
// PUF_ENV_TRAINING_MASK(env): device predicate evaluated before each action.
//   False samples have zero PPO gradients. Counts normalize by valid samples;
//   empty minibatches skip backward and optimizer updates. Sync GPU only.
// PUF_ENV_TRAINING_ROWS(vec): positive, minibatch-aligned prefix of learner rows.
//   Frozen opponents outside that prefix still act but do not enter PPO.
// PUF_ENV_LOGGED_AGENTS(vec): prefix of agents included in episode aggregation.
// PUF_ENV_POLICY_LAYOUT(vec): callback assigns policy_layout; true means handled.
// PUF_ENV_INDEPENDENT_POLICIES: whether the GPU env supplies an explicit layout.
// PUF_ENV_EXTERNAL_STEP: environment stepping cannot use full-rollout capture.
// PUF_ENV_CONFIGURE_TRAINING(kwargs, gamma): pass learner settings to the env.
// PUF_ENV_CONFIGURE_EVAL(ini, match): configure or validate evaluation.
// PUF_ENV_VALIDATE_TRAINING(ini): validate environment-specific training options.
// PUF_ENV_LOAD_INITIAL_POLICY: opt in to base.load_model_path at training start.
// PUF_ENV_RESET_OPPONENT_STATE: clear recurrent carry after replacing opponents.
//
// PUF_ENV_SELFPLAY_STATE: environment-owned, zero-initialized state type.
//   Stored as Selfplay.env_state; keep_anchor preserves the first pool entry.
// PUF_ENV_SELFPLAY_INIT(sp, ini): validate and initialize the extension.
// PUF_ENV_SELFPLAY_CHECKPOINT(sp, p, ini, path, step): handle new checkpoints.
// PUF_ENV_SELFPLAY_PINNED_POLICY(index): exclude pinned slots from random swaps.
// PUF_ENV_SELFPLAY_LOG(sp, log): append environment-owned self-play metrics.
//
// PUF_ENV_COMMAND_HEADER: late include, after trainer/evaluation APIs exist.
// PUF_ENV_COMMAND(argc, argv): return >=0 when handled, -1 for normal dispatch.
