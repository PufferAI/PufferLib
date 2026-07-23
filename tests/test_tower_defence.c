#define TD_TEST
#define main tower_defence_demo_main
#include "../ocean/tower_defence/tower_defence.c"
#undef main

#define CHECK(condition)                                                                           \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition);        \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

static void test_env_init(TowerDefence *env) {
    memset(env, 0, sizeof(*env));
    allocate(env);
    c_reset(env);
}

static void test_env_close(TowerDefence *env) {
    free_allocated(env);
}

static void check_mask_and_observation(const TowerDefence *env) {
    for (int action = 0; action < TD_NUM_ACTIONS; action++) {
        CHECK(env->action_mask[action] == 0 || env->action_mask[action] == 1);
    }
    for (int i = 0; i < TD_OBS_SIZE; i++) {
        CHECK(isfinite(env->observations[i]));
    }
}

static unsigned int test_random(unsigned int *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static int select_valid_action(const TowerDefence *env, unsigned int *state) {
    if (env->action_mask[TD_ACTION_TRIGGER_NEXT_ROUND] && test_random(state) % 4u == 0u) {
        return TD_ACTION_TRIGGER_NEXT_ROUND;
    }
    int count = 0;
    for (int action = 1; action < TD_NUM_ACTIONS; action++) {
        count += env->action_mask[action] != 0;
    }
    if (count == 0) {
        return TD_ACTION_NOOP;
    }
    int selected = (int)(test_random(state) % (unsigned int)count);
    for (int action = 1; action < TD_NUM_ACTIONS; action++) {
        if (!env->action_mask[action]) {
            continue;
        }
        if (selected-- == 0) {
            return action;
        }
    }
    return TD_ACTION_NOOP;
}

static void check_equivalent(const TowerDefence *a, const TowerDefence *b) {
    CHECK(a->round == b->round);
    CHECK(a->status_code == b->status_code);
    CHECK(a->episode_index == b->episode_index);
    CHECK(a->step_count == b->step_count);
    CHECK(a->invalid_action_count == b->invalid_action_count);
    CHECK(a->enemy_high_water == b->enemy_high_water);
    CHECK(a->projectile_count == b->projectile_count);
    CHECK(a->shot_serial == b->shot_serial);
    CHECK(a->impact_serial == b->impact_serial);
    CHECK(a->time == b->time);
    CHECK(a->lives == b->lives);
    CHECK(a->cash == b->cash);
    CHECK(a->score == b->score);
    CHECK(a->rewards[0] == b->rewards[0]);
    CHECK(a->terminals[0] == b->terminals[0]);
    CHECK(memcmp(a->observations, b->observations, TD_OBS_SIZE * sizeof(*a->observations)) == 0);
    CHECK(memcmp(a->action_mask, b->action_mask, TD_NUM_ACTIONS * sizeof(*a->action_mask)) == 0);
    CHECK(memcmp(a->spawns, b->spawns, sizeof(a->spawns)) == 0);
    CHECK(memcmp(a->towers, b->towers, sizeof(a->towers)) == 0);
    CHECK(memcmp(a->enemies, b->enemies, sizeof(a->enemies)) == 0);
    CHECK(memcmp(a->projectiles, b->projectiles,
                 (size_t)a->projectile_count * sizeof(*a->projectiles)) == 0);
    CHECK(memcmp(a->shot_events, b->shot_events, sizeof(a->shot_events)) == 0);
    CHECK(memcmp(a->impact_events, b->impact_events, sizeof(a->impact_events)) == 0);
}

static void test_abi_and_geometry(void) {
    CHECK(TD_NUM_PLACEMENT_SLOTS == 510);
    CHECK(TD_NUM_ACTIONS == 3572);
    CHECK(TD_OBS_SIZE == 5686);
    CHECK(TD_ACTION_TRIGGER_NEXT_ROUND == 3571);

    TowerDefence env;
    test_env_init(&env);
    for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
        float clearance = TD_BUILD_PATH_CLEARANCE + (float)TD_TOWER_RADIUS[kind];
        for (int site = 0; site < TD_NUM_PLACEMENT_SLOTS; site++) {
            CHECK(td_site_buildable_for_kind(site, kind) ==
                  td_site_clear_with_clearance(site, clearance));
        }
    }
    check_mask_and_observation(&env);
    CHECK(env.action_mask[TD_ACTION_NOOP]);
    CHECK(env.action_mask[TD_ACTION_TRIGGER_NEXT_ROUND]);
    test_env_close(&env);
}

static void test_seed_wraparound(void) {
    TowerDefence env = {0};
    allocate(&env);
    env.base_seed = INT_MAX;
    env.rng = INT_MAX;
    env.episode_index = UINT_MAX;
    unsigned int expected = (unsigned int)INT_MAX + (unsigned int)INT_MAX * 100003u + UINT_MAX;
    c_reset(&env);
    CHECK(env.rng_state == expected);
    CHECK(env.episode_index == 0u);
    test_env_close(&env);
}

static void test_seed_zero_is_distinct(void) {
    CHECK(td_derive_seed(0u, 21u) != td_derive_seed(1u, 21u));

    TowerDefence zero = {0};
    TowerDefence one = {0};
    allocate(&zero);
    allocate(&one);
    zero.base_seed = 0;
    one.base_seed = 1;
    c_reset(&zero);
    c_reset(&one);
    zero.round = 21;
    one.round = 21;
    td_prepare_wave(&zero);
    td_prepare_wave(&one);
    CHECK(zero.active_spawns == one.active_spawns);
    CHECK(memcmp(zero.spawns, one.spawns,
                 (size_t)zero.active_spawns * sizeof(*zero.spawns)) != 0);
    test_env_close(&zero);
    test_env_close(&one);
}

static void test_invalid_actions(void) {
    static const float invalid_actions[] = {
        NAN, INFINITY, -INFINITY, -1.0f, (float)TD_NUM_ACTIONS, 0.5f, 1.5f, FLT_MAX,
    };
    TowerDefence env;
    test_env_init(&env);
    for (int i = 0; i < (int)(sizeof(invalid_actions) / sizeof(invalid_actions[0])); i++) {
        env.actions[0] = invalid_actions[i];
        c_step(&env);
        CHECK(env.invalid_action_count == i + 1);
        CHECK(env.rewards[0] == env.invalid_action_reward);
        CHECK(env.terminals[0] == 0.0f);
    }
    env.actions[0] = TD_ACTION_NOOP;
    c_step(&env);
    CHECK(env.invalid_action_count == (int)(sizeof(invalid_actions) / sizeof(invalid_actions[0])));
    test_env_close(&env);
}

static void test_one_mask_build_per_step(void) {
    TowerDefence env;
    test_env_init(&env);
    td_test_mask_update_count = 0;
    c_reset(&env);
    CHECK(td_test_mask_update_count == 1);
    env.actions[0] = TD_ACTION_NOOP;
    c_step(&env);
    CHECK(td_test_mask_update_count == 2);
    test_env_close(&env);
}

static void test_deterministic_rollout(void) {
    TowerDefence a;
    TowerDefence b;
    test_env_init(&a);
    test_env_init(&b);
    unsigned int rng = 7u;
    for (int step = 0; step < 2000; step++) {
        int action = select_valid_action(&a, &rng);
        CHECK(a.action_mask[action] == b.action_mask[action]);
        a.actions[0] = (float)action;
        b.actions[0] = (float)action;
        c_step(&a);
        c_step(&b);
        check_equivalent(&a, &b);
    }
    test_env_close(&a);
    test_env_close(&b);
}

static void test_split_conservation(void) {
    TowerDefence env;
    test_env_init(&env);
    for (int i = 0; i < 255; i++) {
        CHECK(td_add_enemy(&env, 0, 0, 0, 0, 0.2) >= 0);
    }
    int ceramic = td_add_enemy(&env, 9, 0, 0, 0, 0.2);
    CHECK(ceramic >= 0);
    CHECK(td_enemy_count(&env) == 256);
    float reward = 0.0f;
    td_kill_enemy(&env, ceramic, &reward);
    CHECK(td_enemy_count(&env) == 257);
    int zebra_children = 0;
    for (int i = 0; i < env.enemy_high_water; i++) {
        zebra_children += env.enemies[i].alive && env.enemies[i].type == 8;
    }
    CHECK(zebra_children == 2);
    test_env_close(&env);
}

static void test_split_children_wait_one_tick(void) {
    TowerDefence env;
    test_env_init(&env);
    env.status_code = TD_STATUS_ACTIVE;
    env.active_spawns = 1;
    env.spawns[0].count = 1;
    int decoy = td_add_enemy(&env, 0, 0, 0, 0, 0.1);
    int parent = td_add_enemy(&env, 6, 0, 0, 0, 0.4);
    CHECK(decoy == 0 && parent == 1);
    env.enemies[decoy].alive = 0;
    double expected_distance = env.enemies[parent].distance - fmax(2.0, TD_ENEMY_RADIUS[6] * 0.35);
    env.enemies[parent].hp = 0.0f;

    float reward = 0.0f;
    td_advance_world(&env, &reward);
    CHECK(env.enemies[0].alive && env.enemies[0].type == TD_ENEMY_CHILD_A[6]);
    CHECK(env.enemies[0].distance == expected_distance);

    td_advance_world(&env, &reward);
    CHECK(env.enemies[0].distance ==
          expected_distance + TD_ENEMY_SPEED[TD_ENEMY_CHILD_A[6]] * TD_DT);
    test_env_close(&env);
}

static void test_failed_spawn_preserves_rng(void) {
    TowerDefence env;
    test_env_init(&env);
    env.enemy_high_water = TD_MAX_ENEMIES;
    for (int i = 0; i < TD_MAX_ENEMIES; i++) {
        env.enemies[i] = (TdEnemy){
            .alive = 1,
            .id = i + 1,
            .type = 0,
            .hp = 1.0f,
            .max_hp = 1.0f,
            .slow_mult = 1.0f,
        };
    }
    env.status_code = TD_STATUS_SPAWNING;
    env.active_spawns = 1;
    env.spawns[0] = (TdSpawn){
        .type = 0,
        .count = 1,
        .interval = 1.0,
        .use_modifier_chances = 1,
        .camo_chance = 0.5f,
        .fortified_chance = 0.5f,
        .regrow_chance = 0.5f,
        .modifier_rng_state = 123u,
    };
    float reward = 0.0f;
    td_advance_world(&env, &reward);
    CHECK(env.spawns[0].emitted == 0);
    CHECK(env.spawns[0].modifier_rng_state == 123u);
    test_env_close(&env);
}

static void test_projectile_growth(void) {
    TowerDefence env;
    test_env_init(&env);
    int enemy = td_add_enemy(&env, 0, 0, 0, 0, 0.2);
    CHECK(enemy >= 0);
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        env.towers[slot].alive = 1;
        env.towers[slot].kind = slot % TD_NUM_TOWER_TYPES;
    }
    for (int i = 0; i < 2048; i++) {
        td_add_projectile(&env, i % TD_NUM_PLACEMENT_SLOTS, enemy, 1.0f, 0, 200.0f, 0.0f, 0.0f,
                          1.0f, 0.0f);
    }
    CHECK(env.projectile_count == 2048);
    CHECK(env.projectile_capacity >= 2048);
    int expected = 0;
    for (int i = 0; i < env.projectile_count; i++) {
        if (i % 3 == 0) {
            env.projectiles[i].alive = 0;
        } else {
            expected += 1;
        }
    }
    td_compact_projectiles(&env);
    CHECK(env.projectile_count == expected);
    for (int i = 0; i < env.projectile_count; i++) {
        CHECK(env.projectiles[i].alive);
    }
    test_env_close(&env);
}

static int test_dart_shot_count(int fire_rate_tier) {
    TowerDefence env;
    test_env_init(&env);
    int enemy = td_add_enemy(&env, 0, 0, 0, 0, 0.02);
    CHECK(enemy >= 0);
    env.enemies[enemy].speed = 0.0f;
    env.enemies[enemy].hp = 1000000.0f;
    env.enemies[enemy].max_hp = env.enemies[enemy].hp;
    env.towers[0].alive = 1;
    env.towers[0].kind = 0;
    env.towers[0].upgrades[2] = fire_rate_tier;
    for (int tick = 0; tick < 400; tick++) {
        td_apply_tower_damage(&env);
    }
    int shots = env.projectile_count;
    test_env_close(&env);
    return shots;
}

static void test_fire_rate_tiers_are_monotonic(void) {
    int previous = test_dart_shot_count(0);
    CHECK(previous > 0);
    for (int tier = 1; tier <= TD_MAX_TIER; tier++) {
        int shots = test_dart_shot_count(tier);
        CHECK(shots > previous);
        previous = shots;
    }
}

static void test_idle_tower_discards_cooldown_debt(void) {
    TowerDefence env;
    test_env_init(&env);
    env.towers[0].alive = 1;
    env.towers[0].kind = 0;
    env.towers[0].cooldown = 0.1f;

    td_apply_tower_damage(&env);
    CHECK(env.towers[0].cooldown == 0.0f);
    CHECK(env.projectile_count == 0);

    int enemy = td_add_enemy(&env, 0, 0, 0, 0, 0.02);
    CHECK(enemy >= 0);
    env.enemies[enemy].speed = 0.0f;
    float range, damage, fire_rate, projectile_speed, burn_dps, burn_time, slow_mult, slow_time;
    int damage_type, detect_camo;
    td_tower_stats(&env.towers[0], &range, &damage, &fire_rate, &projectile_speed, &damage_type,
                   &detect_camo, &burn_dps, &burn_time, &slow_mult, &slow_time);
    td_apply_tower_damage(&env);
    CHECK(env.projectile_count == 1);
    CHECK(env.towers[0].cooldown == fire_rate);
    test_env_close(&env);
}

static void test_projectile_travel_segments(void) {
    TowerDefence env;
    test_env_init(&env);
    int slot = 4 * TD_BUILD_GRID_COLS + 13;
    env.towers[slot].alive = 1;
    env.towers[slot].kind = 2;
    int enemy = td_add_enemy(&env, 0, 0, 0, 0, 526.0 / TD_PATH_LENGTH);
    CHECK(enemy >= 0);
    env.enemies[enemy].speed = 0.0f;
    env.enemies[enemy].hp = 100.0f;
    env.enemies[enemy].max_hp = 100.0f;
    td_add_projectile(&env, slot, enemy, 1.0f, 0, 180.0f, 0.0f, 0.0f, 1.0f, 0.0f);

    float reward = 0.0f;
    td_update_projectiles(&env, &reward);
    CHECK(env.projectile_count == 1);
    TdProjectile *projectile = &env.projectiles[0];
    CHECK(projectile->previous_x == td_site_x(slot));
    CHECK(projectile->previous_y == td_site_y(slot));
    CHECK(fabsf(hypotf(projectile->x - projectile->previous_x,
                       projectile->y - projectile->previous_y) -
                 180.0f * TD_DT) < 0.001f);
    Vector2 tail = td_render_projectile_tail(projectile, 0.0f);
    Vector2 head = td_render_projectile_point(&env, projectile, 0.0f);
    CHECK(tail.x == td_site_x(slot) && tail.y == td_site_y(slot));
    CHECK(head.x == projectile->x && head.y == projectile->y);

    float prior_x = projectile->x;
    float prior_y = projectile->y;
    td_update_projectiles(&env, &reward);
    CHECK(env.projectile_count == 1);
    projectile = &env.projectiles[0];
    CHECK(projectile->previous_x == prior_x && projectile->previous_y == prior_y);

    env.projectile_count = 0;
    int close_slot = 4 * TD_BUILD_GRID_COLS + 12;
    env.towers[close_slot].alive = 1;
    env.towers[close_slot].kind = 2;
    env.enemies[enemy].distance = 419.0;
    td_add_projectile(&env, close_slot, enemy, 1.0f, 0, 180.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    td_update_projectiles(&env, &reward);
    CHECK(env.projectile_count == 0);
    CHECK(env.impact_serial == 1);
    CHECK(env.impact_events[0].from_x == td_site_x(close_slot));
    CHECK(env.impact_events[0].from_y == td_site_y(close_slot));
    CHECK(env.impact_events[0].x == 384.0f && env.impact_events[0].y == 143.0f);
    test_env_close(&env);
}

static void test_exact_endpoint_leaks(void) {
    TowerDefence env;
    test_env_init(&env);
    env.status_code = TD_STATUS_ACTIVE;
    env.active_spawns = 1;
    env.spawns[0].count = 1;
    int enemy = td_add_enemy(&env, 0, 0, 0, 0, 1.0);
    CHECK(enemy >= 0);
    env.enemies[enemy].speed = 0.0f;
    float reward = 0.0f;
    td_advance_world(&env, &reward);
    CHECK(env.lives == 199.0f);
    CHECK(td_enemy_count(&env) == 0);
    test_env_close(&env);
}

static void test_terminal_reset_and_log(void) {
    TowerDefence env;
    test_env_init(&env);
    env.max_episode_steps = 1;
    unsigned int episode = env.episode_index;
    env.actions[0] = TD_ACTION_NOOP;
    c_step(&env);
    CHECK(env.terminals[0] == 1.0f);
    CHECK(env.log.n == 1.0f);
    CHECK(env.episode_index == episode + 1);
    CHECK(env.step_count == 0);
    check_mask_and_observation(&env);
    test_env_close(&env);
}

static void test_policy_and_animation_state(void) {
    float state[12];
    for (int i = 0; i < 12; i++) {
        state[i] = (float)(i + 1);
    }
    MinGRU mingru = {
        .state = state,
        .batch_size = 2,
        .hidden_size = 3,
        .num_layers = 2,
    };
    PufferNet net = {.mingru = &mingru};
    td_reset_policy_state(&net);
    for (int i = 0; i < 12; i++) {
        CHECK(state[i] == 0.0f);
    }

    float logits[4] = {0.0f, 1000.0f, 0.0f, -1000.0f};
    unsigned char mask[4] = {1, 0, 1, 0};
    uint32_t rng_a = 123u;
    uint32_t rng_b = 123u;
    int saw_zero = 0;
    int saw_two = 0;
    for (int i = 0; i < 64; i++) {
        int action_a = td_demo_sample_masked_action(logits, mask, 4, &rng_a);
        int action_b = td_demo_sample_masked_action(logits, mask, 4, &rng_b);
        CHECK(action_a == action_b);
        CHECK(action_a == 0 || action_a == 2);
        saw_zero |= action_a == 0;
        saw_two |= action_a == 2;
    }
    CHECK(saw_zero && saw_two);
    logits[0] = NAN;
    logits[2] = INFINITY;
    CHECK(td_demo_sample_masked_action(logits, mask, 4, &rng_a) == 0);

    TdClient client = {0};
    TdTower tower = {.kind = 0};
    CHECK(td_tower_frame(&client, &tower, 3, 0.0) == 0);
    CHECK(td_tower_frame(&client, &tower, 3, 0.26) == 1);
    client.fire_until[3] = 1.0;
    CHECK(td_tower_frame(&client, &tower, 3, 0.5) == 2);
    Vector2 aligned = td_tower_sprite_center((Vector2){10.0f, 20.0f}, 0, 2, 54.0f);
    CHECK(aligned.x == 10.0f);
    CHECK(aligned.y > 25.9f && aligned.y < 26.1f);
    Texture2D texture = {.width = 100, .height = 80};
    Rectangle frame = td_sheet_source(texture, 3);
    CHECK(frame.x == 50.0f && frame.y == 40.0f);
    CHECK(frame.width == 50.0f && frame.height == 40.0f);

    client.observed_sim_time_at = 10.0;
    CHECK(td_render_phase(&client, 10.0) == 0.0f);
    CHECK(td_render_phase(&client, 10.125) == 0.5f);
    CHECK(td_render_phase(&client, 10.25) == 1.0f);
    TdEnemy enemy = {
        .alive = 1,
        .id = 7,
        .distance = 100.0,
        .speed = 60.0f,
        .slow_mult = 1.0f,
    };
    Vector2 enemy_point = td_render_enemy_point(&enemy, 1.0f);
    CHECK(enemy_point.x == 115.0f && enemy_point.y == 108.0f);
    TowerDefence render_env = {.enemy_high_water = 1};
    render_env.enemies[0] = enemy;
    TdProjectile projectile = {
        .alive = 1,
        .target_enemy = 0,
        .target_enemy_id = 7,
        .x = 0.0f,
        .y = 108.0f,
        .previous_x = -10.0f,
        .previous_y = 108.0f,
        .speed = 40.0f,
    };
    Vector2 projectile_point = td_render_projectile_point(&render_env, &projectile, 1.0f);
    CHECK(projectile_point.x == 10.0f && projectile_point.y == 108.0f);
    Vector2 projectile_tail = td_render_projectile_tail(&projectile, 0.5f);
    CHECK(projectile_tail.x == -5.0f && projectile_tail.y == 108.0f);
}

static void test_impact_lifecycle(void) {
    TowerDefence env;
    test_env_init(&env);
    TdProjectile projectile = {
        .kind = 2,
        .previous_x = 100.0f,
        .previous_y = 200.0f,
        .x = 123.0f,
        .y = 234.0f,
    };
    td_record_impact(&env, &projectile);
    CHECK(env.impact_serial == 1);
    CHECK(env.impact_events[0].serial == 1);

    TdClient client = {
        .episode_index = env.episode_index,
        .observed_sim_time_at = 10.0,
    };
    env.client = &client;
    td_sync_client_animation(&env, 10.0);
    CHECK(client.impact_serial == 1);
    CHECK(client.impacts[0].kind == 2);
    CHECK(client.impacts[0].from_x == 100.0f && client.impacts[0].from_y == 200.0f);
    CHECK(client.impacts[0].x == 123.0f && client.impacts[0].y == 234.0f);
    CHECK(client.impacts[0].until > 10.13 && client.impacts[0].until < 10.15);
    double until = client.impacts[0].until;
    td_sync_client_animation(&env, 10.05);
    CHECK(client.impacts[0].until == until);

    env.episode_index += 1;
    td_sync_client_animation(&env, 10.1);
    CHECK(client.impacts[0].until == 0.0);
    env.client = NULL;
    test_env_close(&env);
}

static void test_animation_event_catchup(void) {
    TowerDefence env;
    test_env_init(&env);
    TdClient client = {.episode_index = env.episode_index};
    env.client = &client;

    env.time = 0.25f;
    td_record_shot(&env, 1);
    TdProjectile stale_impact = {
        .kind = 0,
        .previous_x = 1.0f,
        .previous_y = 2.0f,
        .x = 3.0f,
        .y = 4.0f,
    };
    td_record_impact(&env, &stale_impact);
    env.time = 0.5f;
    td_record_shot(&env, 2);
    env.time = 1.0f;
    td_sync_client_animation(&env, 10.0);
    CHECK(client.shot_serial == env.shot_serial);
    CHECK(client.impact_serial == env.impact_serial);
    CHECK(client.fire_until[1] == 0.0 && client.fire_until[2] == 0.0);
    CHECK(client.impacts[0].until == 0.0);

    td_record_shot(&env, 3);
    td_record_shot(&env, TD_NUM_PLACEMENT_SLOTS - 1);
    CHECK(env.shot_serial == 3);
    td_sync_client_animation(&env, 10.01);
    CHECK(client.fire_until[3] > 10.11 && client.fire_until[3] < 10.13);
    CHECK(client.fire_until[TD_NUM_PLACEMENT_SLOTS - 1] > 10.11 &&
          client.fire_until[TD_NUM_PLACEMENT_SLOTS - 1] < 10.13);

    env.client = NULL;
    test_env_close(&env);
}

static void test_shot_event_ring_wraparound(void) {
    TowerDefence env;
    test_env_init(&env);
    TdClient client = {.episode_index = env.episode_index};
    env.client = &client;

    for (int i = 0; i < TD_MAX_SHOT_EVENTS + 2; i++) {
        env.time = (float)(i + 1) / 1000.0f;
        td_record_shot(&env, i);
    }
    CHECK(env.shot_serial == TD_MAX_SHOT_EVENTS + 2);
    for (uint64_t serial = 3; serial <= env.shot_serial; serial++) {
        int index = (int)((serial - 1) % TD_MAX_SHOT_EVENTS);
        CHECK(env.shot_events[index].serial == serial);
    }

    td_sync_client_animation(&env, 10.0);
    CHECK(client.shot_serial == env.shot_serial);
    CHECK(client.fire_until[0] == 0.0 && client.fire_until[1] == 0.0);
    for (int slot = 2; slot < TD_MAX_SHOT_EVENTS + 2; slot++) {
        CHECK(client.fire_until[slot] > 10.1);
    }

    env.client = NULL;
    test_env_close(&env);
}

static void test_random_stress(void) {
    TowerDefence env;
    test_env_init(&env);
    env.max_episode_steps = 4000;
    unsigned int rng = 12345u;
    for (int step = 0; step < 5000; step++) {
        if (step % 997 == 0) {
            env.actions[0] = NAN;
        } else {
            env.actions[0] = (float)select_valid_action(&env, &rng);
        }
        c_step(&env);
        CHECK(isfinite(env.rewards[0]));
        CHECK(isfinite(env.lives));
        CHECK(isfinite(env.cash));
        CHECK(env.enemy_high_water >= 0 && env.enemy_high_water <= TD_MAX_ENEMIES);
        CHECK(env.projectile_count >= 0 && env.projectile_count <= env.projectile_capacity);
        check_mask_and_observation(&env);
    }
    test_env_close(&env);
}

#ifdef TD_TEST_RENDER
static int nearest_render_slot(const TowerDefence *env, int kind, float x, float y) {
    int best_slot = -1;
    float best_distance = INFINITY;
    for (int slot = 0; slot < TD_NUM_PLACEMENT_SLOTS; slot++) {
        int action = TD_ACTION_PLACE_FIRST + slot * TD_NUM_TOWER_TYPES + kind;
        if (!env->action_mask[action]) {
            continue;
        }
        float dx = td_site_x(slot) - x;
        float dy = td_site_y(slot) - y;
        float distance = dx * dx + dy * dy;
        if (distance < best_distance) {
            best_slot = slot;
            best_distance = distance;
        }
    }
    return best_slot;
}

static void test_render_purity(void) {
    TowerDefence env;
    test_env_init(&env);
    TowerDefence before = env;
    c_render(&env);
    TdClient *client = env.client;
    env.client = before.client;
    CHECK(memcmp(&env, &before, sizeof(env)) == 0);
    env.client = client;

    static const float target_x[TD_NUM_TOWER_TYPES] = {250.0f, 520.0f, 750.0f};
    static const float target_y[TD_NUM_TOWER_TYPES] = {220.0f, 180.0f, 380.0f};
    int tower_slots[TD_NUM_TOWER_TYPES];
    for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
        tower_slots[kind] = nearest_render_slot(&env, kind, target_x[kind], target_y[kind]);
        CHECK(tower_slots[kind] >= 0);
        td_place(&env, tower_slots[kind], kind);
        td_write_observation(&env);
    }
    for (int kind = 0; kind < TD_NUM_TOWER_TYPES; kind++) {
        int enemy = td_add_enemy(&env, kind * 4, kind == 1, kind == 2, 0, 0.18 + kind * 0.23);
        CHECK(enemy >= 0);
        td_add_projectile(&env, tower_slots[kind], enemy, 1.0f, 0, 200.0f, 0.0f, 0.0f, 1.0f, 0.0f);
        td_record_shot(&env, tower_slots[kind]);
    }
    TdProjectile impact = {
        .kind = 2,
        .previous_x = 640.0f,
        .previous_y = 270.0f,
        .x = 670.0f,
        .y = 270.0f,
    };
    td_record_impact(&env, &impact);
    td_write_observation(&env);

    float observations[TD_OBS_SIZE];
    float actions[1];
    float rewards[1];
    float terminals[1];
    unsigned char action_mask[TD_NUM_ACTIONS];
    memcpy(observations, env.observations, sizeof(observations));
    memcpy(actions, env.actions, sizeof(actions));
    memcpy(rewards, env.rewards, sizeof(rewards));
    memcpy(terminals, env.terminals, sizeof(terminals));
    memcpy(action_mask, env.action_mask, sizeof(action_mask));
    int projectile_count = env.projectile_count;
    TdProjectile *projectiles =
        (TdProjectile *)malloc((size_t)projectile_count * sizeof(*projectiles));
    CHECK(projectiles != NULL);
    memcpy(projectiles, env.projectiles, (size_t)projectile_count * sizeof(*projectiles));
    before = env;
    c_render(&env);
    CHECK(memcmp(&env, &before, sizeof(env)) == 0);
    CHECK(memcmp(env.observations, observations, sizeof(observations)) == 0);
    CHECK(memcmp(env.actions, actions, sizeof(actions)) == 0);
    CHECK(memcmp(env.rewards, rewards, sizeof(rewards)) == 0);
    CHECK(memcmp(env.terminals, terminals, sizeof(terminals)) == 0);
    CHECK(memcmp(env.action_mask, action_mask, sizeof(action_mask)) == 0);
    CHECK(env.projectile_count == projectile_count);
    CHECK(memcmp(env.projectiles, projectiles,
                 (size_t)projectile_count * sizeof(*projectiles)) == 0);
    free(projectiles);

    c_close(&env);
    free_allocated(&env);
}

static void test_multi_env_render_close(void) {
    TowerDefence a;
    TowerDefence b;
    test_env_init(&a);
    test_env_init(&b);
    c_render(&a);
    c_render(&b);
    CHECK(IsWindowReady());
    CHECK(td_render_client_count == 2);
    c_close(&a);
    CHECK(IsWindowReady());
    CHECK(td_render_client_count == 1);
    c_close(&b);
    CHECK(!IsWindowReady());
    CHECK(td_render_client_count == 0);
    free_allocated(&a);
    free_allocated(&b);
}
#endif

int main(void) {
#ifdef TD_TEST_RENDER
    SetTraceLogLevel(LOG_WARNING);
#endif
    test_abi_and_geometry();
    test_seed_wraparound();
    test_seed_zero_is_distinct();
    test_invalid_actions();
    test_one_mask_build_per_step();
    test_deterministic_rollout();
    test_split_conservation();
    test_split_children_wait_one_tick();
    test_failed_spawn_preserves_rng();
    test_projectile_growth();
    test_fire_rate_tiers_are_monotonic();
    test_idle_tower_discards_cooldown_debt();
    test_projectile_travel_segments();
    test_exact_endpoint_leaks();
    test_terminal_reset_and_log();
    test_policy_and_animation_state();
    test_impact_lifecycle();
    test_animation_event_catchup();
    test_shot_event_ring_wraparound();
    test_random_stress();
#ifdef TD_TEST_RENDER
    test_render_purity();
    test_multi_env_render_close();
#endif
    printf("tower_defence tests passed\n");
    return 0;
}
