#pragma once

// Included after the trainer's policy and self-play APIs are defined.
static void racing_configure_eval(Ini* ini, bool match) {
    assert(!match && "Use racing_multi race FILE FILE ... for competing checkpoints");
    puf_ini_put(ini, "vec.num_policies", "1");
    puf_ini_put(ini, "selfplay.enabled", "0");
}

static void racing_selfplay_init(Selfplay* sp, Ini* ini) {
    assert(sp->max_size >= 2 && "racing promotion needs an anchor and a challenger");
    assert(puf_ini_get(ini, "selfplay", "promotion_min_races") > 0);
    assert(puf_ini_get(ini, "selfplay", "promotion_min_per_grid") > 0);
    float win_rate = puf_ini_get(ini, "selfplay", "promotion_win_rate");
    assert(win_rate > 0.5f && win_rate <= 1.0f);
    sp->keep_anchor = 1;
}

static void racing_selfplay_log(Selfplay* sp, Dict* log) {
    dict_set(log, "pool/promotions", sp->env_state.promotions);
    dict_set(log, "pool/promotion_games", sp->env_state.promotion_games);
    dict_set(log, "pool/promotion_score", sp->env_state.promotion_score);
}

static void racing_validate_training(Ini* ini) {
    int cars = puf_ini_get(ini, "env", "race_cars");
    int agents = puf_ini_get(ini, "vec", "total_agents");
    int rows = puf_ini_get(ini, "train", "minibatch_size") / puf_ini_get(ini, "train", "horizon");
    assert(cars >= 2 && cars <= 8 && agents % cars == 0);
    int learners = puf_ini_get(ini, "selfplay", "enabled") ? agents / cars : agents;
    assert(learners >= rows && learners % rows == 0
        && "racing_multi learner cars must be divisible by minibatch rows");
    assert(puf_ini_get(ini, "selfplay", "eval_games") == 0
        && puf_ini_get(ini, "selfplay", "eval_bot_games") == 0
        && "Use racing_multi race for opponent evaluation");
}

static void racing_load_frozen(PuffeRL* p, int policy, const char* checkpoint) {
    pufferl_load_policy(p, policy, checkpoint);
    for (int b = 0; b < p->vec->buffers; b++) {
        Prec state = p->policies[policy].buffer_states[b];
        pf_optix_cuda(cudaMemsetAsync(state.data, 0,
            numel(state.shape) * sizeof(precision_t), p->streams[b]));
    }
}

// Slot 2 holds a fixed saved PPO checkpoint while slot 1 holds the incumbent.
// Promote only after their races cover every starting grid position.
static void racing_selfplay_promote(Selfplay* sp, PuffeRL* p, Ini* ini,
        const char* latest_checkpoint, long step) {
    if (!sp->env_state.challenger[0]) {
        racing_load_frozen(p, 2, latest_checkpoint);
        snprintf(sp->env_state.challenger, sizeof(sp->env_state.challenger), "%s", latest_checkpoint);
        sp->hist[1].opp_started_step = step;
        racing_race.promotion_active = 1;
        racing_race.promotion_generation++;
        printf("racing promotion: evaluating challenger %s\n", latest_checkpoint);
        return;
    }
    unsigned races[RACING_RACE_MAX], points[RACING_RACE_MAX];
    pf_optix_cuda(cudaDeviceSynchronize());
    pf_optix_cuda(cudaMemcpy(races, racing_race.promotion_races,
        sizeof(races), cudaMemcpyDeviceToHost));
    pf_optix_cuda(cudaMemcpy(points, racing_race.promotion_points,
        sizeof(points), cudaMemcpyDeviceToHost));
    int total = 0, min_grid = puf_ini_get(ini, "selfplay", "promotion_min_per_grid");
    float score = 0;
    bool balanced = true;
    for (int grid = 0; grid < racing_race.cars; grid++) {
        total += races[grid];
        balanced &= races[grid] >= (unsigned)min_grid;
        if (races[grid]) score += (float)points[grid] / (2.0f * races[grid]);
    }
    score /= racing_race.cars;
    if (total < puf_ini_get(ini, "selfplay", "promotion_min_races") || !balanced) return;
    sp->env_state.promotion_games = total;
    sp->env_state.promotion_score = score;
    float threshold = puf_ini_get(ini, "selfplay", "promotion_win_rate");
    bool promoted = score >= threshold;
    if (promoted) {
        racing_load_frozen(p, 1, sp->env_state.challenger);
        selfplay_add_checkpoint(sp, sp->env_state.challenger);
        sp->env_state.promotions++;
        sp->hist[0].opp_started_step = step;
    }
    printf("racing promotion: %s %s, grid-balanced score %.3f from %d races, threshold %.3f\n",
        promoted ? "accepted" : "rejected", sp->env_state.challenger, score, total, threshold);
    racing_load_frozen(p, 2, latest_checkpoint);
    snprintf(sp->env_state.challenger, sizeof(sp->env_state.challenger), "%s", latest_checkpoint);
    sp->hist[1].opp_started_step = step;
    racing_race.promotion_generation++;
    pf_optix_cuda(cudaMemset(racing_race.promotion_races, 0, sizeof(races)));
    pf_optix_cuda(cudaMemset(racing_race.promotion_points, 0, sizeof(points)));
    printf("racing promotion: evaluating challenger %s\n", latest_checkpoint);
}
