#include "table_hockey.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_GREEN = (Color){0, 187, 0, 255};
const Color PUFF_BLUE = (Color){0, 100, 187, 255};

void init(TableHockey* env) {
    env->tick = 0;
    memset(&env->log, 0, sizeof(Log));
    env->positioning_reward = 0.005f;
    env->power_shot_bonus = 0.02f;
    env->anticipation_reward = 0.01f;
    env->save_zone_reward = 0.015f;
    env->efficiency_penalty = 0.0002f;
    env->rally_bonus_base = 0.01f;
    init_physics_cache(env);
    reset_game(env);
}

void allocate(TableHockey* env) {
    init(env);
    env->observations = (float*)calloc(11, sizeof(float));
    env->actions = (float*)calloc(2, sizeof(float));
    env->rewards = (float*)calloc(1, sizeof(float));
    env->terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env->truncations = (unsigned char*)calloc(1, sizeof(unsigned char));
}

void free_allocated(TableHockey* env) {
    if (env->observations) free(env->observations);
    if (env->actions) free(env->actions);
    if (env->rewards) free(env->rewards);
    if (env->terminals) free(env->terminals);
    if (env->truncations) free(env->truncations);
}

void c_close(TableHockey* env) {
    if (env->client) {
        if (env->client->window_initialized) {
            CloseWindow();
        }
        free(env->client);
        env->client = NULL;
    }
}

void c_reset(TableHockey* env) {
    reset_game(env);
    update_observations(env);
    env->terminals[0] = 0;
    env->truncations[0] = 0;
    env->rewards[0] = 0.0f;
}

void c_step(TableHockey* env) {
    apply_actions(env);
    if (env->render_mode != RENDER_HEADLESS) {
        process_human_opponent_input(env);
    }
    update_physics(env);
    calculate_rewards(env);
    update_observations(env);

    env->tick++;
    env->game_state.episode_length++;
    
    bool score_limit_reached = (env->game_state.player_score >= 5 || env->game_state.opponent_score >= 5);
    bool time_limit_reached = (env->render_mode == RENDER_HEADLESS) && (env->tick >= MAX_STEPS);
    
    if (score_limit_reached || time_limit_reached) {
        env->terminals[0] = 1;
        add_log(env);
        reset_game(env);
    } else {
        env->terminals[0] = 0;
    }
    
    env->truncations[0] = time_limit_reached ? 1 : 0;
}

void c_render(TableHockey* env) {
    if (!env->client && env->render_mode != RENDER_HEADLESS) {
        env->client = make_client(env);
    }

    if (env->render_mode == RENDER_VR) {
        process_vr_input(env);
    }

    render_frame(env);
}

// PHYSICS SIMULATION

void update_physics(TableHockey* env) {
    env->game_state.prev_puck_x = env->puck.x;
    env->game_state.prev_puck_y = env->puck.y;
    float prev_player_x = env->player_paddle.x;
    float prev_player_y = env->player_paddle.y;
    float prev_opponent_x = env->opponent_paddle.x;
    float prev_opponent_y = env->opponent_paddle.y;
    
    update_paddles(env);
    update_puck(env);

    env->game_state.player_vel_x = (env->player_paddle.x - prev_player_x) / env->dt;
    env->game_state.player_vel_y = (env->player_paddle.y - prev_player_y) / env->dt;
    env->game_state.opponent_vel_x = (env->opponent_paddle.x - prev_opponent_x) / env->dt;
    env->game_state.opponent_vel_y = (env->opponent_paddle.y - prev_opponent_y) / env->dt;
    env->game_state.time_since_last_hit += env->dt;
    
    handle_collisions(env);
    handle_wall_collisions(env);
    handle_goal_scoring(env);
}

void update_puck(TableHockey* env) {
#if SIMD_AVAILABLE && defined(__SSE2__)
    __m128 puck_state = _mm_set_ps(env->puck.vy, env->puck.vx, env->puck.y, env->puck.x);
    __m128 velocity = _mm_set_ps(env->puck.vy, env->puck.vx, env->puck.vy, env->puck.vx);
    __m128 dt_vec = _mm_set1_ps(env->dt);
    __m128 position_delta = _mm_mul_ps(velocity, dt_vec);
    __m128 pos_mask = _mm_set_ps(0.0f, 0.0f, 1.0f, 1.0f);
    position_delta = _mm_mul_ps(position_delta, pos_mask);
    puck_state = _mm_add_ps(puck_state, position_delta);
    __m128 friction_vec = _mm_set1_ps(0.9995f);
    __m128 vel_mask = _mm_set_ps(1.0f, 1.0f, 0.0f, 0.0f);
    __m128 current_vel = _mm_mul_ps(puck_state, vel_mask);
    __m128 friction_applied = _mm_mul_ps(current_vel, friction_vec);
    puck_state = _mm_add_ps(_mm_mul_ps(puck_state, _mm_sub_ps(_mm_set1_ps(1.0f), vel_mask)), friction_applied);
    
    // Store results back
    float result[4];
    _mm_storeu_ps(result, puck_state);
    env->puck.x = result[0];
    env->puck.y = result[1];
    env->puck.vx = result[2];
    env->puck.vy = result[3];
    
#else
    env->puck.x += env->puck.vx * env->dt;
    env->puck.y += env->puck.vy * env->dt;
    float friction = 0.9995f;
    env->puck.vx *= friction;
    env->puck.vy *= friction;
#endif
    float speed = vector_length(env->puck.vx, env->puck.vy);
    if (speed > PUCK_MAX_SPEED) {
        env->puck.vx = (env->puck.vx / speed) * PUCK_MAX_SPEED;
        env->puck.vy = (env->puck.vy / speed) * PUCK_MAX_SPEED;
    }
}

void update_paddles(TableHockey* env) {
    float prev_player_x = env->player_paddle.x;
    float prev_player_y = env->player_paddle.y;
    float prev_opponent_x = env->opponent_paddle.x;
    float prev_opponent_y = env->opponent_paddle.y;
    float new_player_x = env->player_paddle.x + env->player_paddle.vx * env->dt;
    float new_player_y = env->player_paddle.y + env->player_paddle.vy * env->dt;
    env->player_paddle.x += (new_player_x - env->player_paddle.x) * PADDLE_SMOOTHING;
    env->player_paddle.y += (new_player_y - env->player_paddle.y) * PADDLE_SMOOTHING;
    env->player_paddle.x = clamp_value(env->player_paddle.x, 
        -TABLE_WIDTH/2 + PADDLE_RADIUS, 0 - PADDLE_RADIUS);
    env->player_paddle.y = clamp_value(env->player_paddle.y, 
        -TABLE_HEIGHT/2 + PADDLE_RADIUS, TABLE_HEIGHT/2 - PADDLE_RADIUS);
    
    if (env->render_mode == RENDER_HEADLESS) {
        float new_opponent_x = env->opponent_paddle.x + env->opponent_paddle.vx * env->dt;
        float new_opponent_y = env->opponent_paddle.y + env->opponent_paddle.vy * env->dt;
        
        env->opponent_paddle.x += (new_opponent_x - env->opponent_paddle.x) * PADDLE_SMOOTHING;
        env->opponent_paddle.y += (new_opponent_y - env->opponent_paddle.y) * PADDLE_SMOOTHING;
        env->opponent_paddle.x = clamp_value(env->opponent_paddle.x, 
            PADDLE_RADIUS, TABLE_WIDTH/2 - PADDLE_RADIUS);
        env->opponent_paddle.y = clamp_value(env->opponent_paddle.y, 
            -TABLE_HEIGHT/2 + PADDLE_RADIUS, TABLE_HEIGHT/2 - PADDLE_RADIUS);
    }
    float player_movement = distance(prev_player_x, prev_player_y, 
                                   env->player_paddle.x, env->player_paddle.y);
    float opponent_movement = distance(prev_opponent_x, prev_opponent_y, 
                                     env->opponent_paddle.x, env->opponent_paddle.y);
    
    env->game_state.player_distance_traveled += player_movement;
    env->game_state.opponent_distance_traveled += opponent_movement;
}

bool check_puck_paddle_collision(Puck* puck, Paddle* paddle, float* collision_normal_x, float* collision_normal_y) {
#if SIMD_AVAILABLE && defined(__SSE2__)
    __m128 puck_pos = _mm_set_ps(0.0f, 0.0f, puck->y, puck->x);
    __m128 paddle_pos = _mm_set_ps(0.0f, 0.0f, paddle->y, paddle->x);
    __m128 diff = _mm_sub_ps(puck_pos, paddle_pos);
    __m128 dist_sq = _mm_dp_ps(diff, diff, 0x31);
    float distance_squared = _mm_cvtss_f32(dist_sq);
    
    float collision_distance = PUCK_RADIUS + PADDLE_RADIUS;
    
    if (distance_squared <= collision_distance * collision_distance) {
        float distance = sqrtf(distance_squared);
        if (distance > 0) {
            float result[4];
            _mm_storeu_ps(result, diff);
            *collision_normal_x = result[0] / distance;
            *collision_normal_y = result[1] / distance;
        } else {
            *collision_normal_x = 1.0f;
            *collision_normal_y = 0.0f;
        }
        return true;
    }
    return false;
    
#else
    float dx = puck->x - paddle->x;
    float dy = puck->y - paddle->y;
    float distance_squared = dx*dx + dy*dy;
    float collision_distance = PUCK_RADIUS + PADDLE_RADIUS;
    
    if (distance_squared <= collision_distance * collision_distance) {
        float distance = sqrtf(distance_squared);
        if (distance > 0) {
            *collision_normal_x = dx / distance;
            *collision_normal_y = dy / distance;
        } else {
            *collision_normal_x = 1.0f;
            *collision_normal_y = 0.0f;
        }
        return true;
    }
    return false;
#endif
}

void handle_collisions(TableHockey* env) {
    float normal_x, normal_y;
    if (check_puck_paddle_collision(&env->puck, &env->player_paddle, &normal_x, &normal_y)) {
        bool is_defensive = (env->puck.vx < 0);
        float dot_product = env->puck.vx * normal_x + env->puck.vy * normal_y;
        env->puck.vx = env->puck.vx - 2.0f * dot_product * normal_x;
        env->puck.vy = env->puck.vy - 2.0f * dot_product * normal_y;
        env->puck.vx *= COLLISION_BOOST;
        env->puck.vy *= COLLISION_BOOST;
        float separation = PUCK_RADIUS + PADDLE_RADIUS;
        env->puck.x = env->player_paddle.x + normal_x * separation;
        env->puck.y = env->player_paddle.y + normal_y * separation;
        env->game_state.puck_hits++;
        env->game_state.current_rally_length++;
        env->game_state.last_hit_time = env->tick;
        env->game_state.time_since_last_hit = 0.0f;
        float puck_speed = vector_length(env->puck.vx, env->puck.vy);
        env->game_state.total_puck_speed += puck_speed;
        env->game_state.total_shots++;
        
        if (is_defensive) {
            env->game_state.defensive_hits++;
            env->game_state.total_blocks++;
            
            float reaction_time = 1.0f / (puck_speed + 0.1f);
            env->game_state.total_reaction_time += reaction_time;
            env->game_state.reaction_count++;
        }
    }
    
    if (check_puck_paddle_collision(&env->puck, &env->opponent_paddle, &normal_x, &normal_y)) {
        bool is_defensive = (env->puck.vx > 0);
        
        float dot_product = env->puck.vx * normal_x + env->puck.vy * normal_y;
        env->puck.vx = env->puck.vx - 2.0f * dot_product * normal_x;
        env->puck.vy = env->puck.vy - 2.0f * dot_product * normal_y;
        
        env->puck.vx *= COLLISION_BOOST;
        env->puck.vy *= COLLISION_BOOST;
        
        float separation = PUCK_RADIUS + PADDLE_RADIUS;
        env->puck.x = env->opponent_paddle.x + normal_x * separation;
        env->puck.y = env->opponent_paddle.y + normal_y * separation;
        
        env->game_state.puck_hits++;
        env->game_state.current_rally_length++;
        env->game_state.last_hit_time = env->tick;
        env->game_state.time_since_last_hit = 0.0f;

        float puck_speed = vector_length(env->puck.vx, env->puck.vy);
        env->game_state.total_puck_speed += puck_speed;
        env->game_state.total_shots++;

        if (!is_defensive) {
            env->game_state.opponent_shots++;
        }
    }
}

void handle_wall_collisions(TableHockey* env) {
    if (env->puck.y - PUCK_RADIUS <= -TABLE_HEIGHT/2) {
        env->puck.y = -TABLE_HEIGHT/2 + PUCK_RADIUS;
        env->puck.vy = -env->puck.vy * WALL_RESTITUTION;
        env->game_state.wall_bounces_this_rally++;
        env->game_state.puck_hit_wall_this_rally = true;
    }
    if (env->puck.y + PUCK_RADIUS >= TABLE_HEIGHT/2) {
        env->puck.y = TABLE_HEIGHT/2 - PUCK_RADIUS;
        env->puck.vy = -env->puck.vy * WALL_RESTITUTION;
        env->game_state.wall_bounces_this_rally++;
        env->game_state.puck_hit_wall_this_rally = true;
    }
    
    float goal_top = GOAL_WIDTH / 2;
    float goal_bottom = -GOAL_WIDTH / 2;
    
    if (env->puck.x - PUCK_RADIUS <= -TABLE_WIDTH/2) {
        if (env->puck.y < goal_bottom || env->puck.y > goal_top) {
            env->puck.x = -TABLE_WIDTH/2 + PUCK_RADIUS;
            env->puck.vx = -env->puck.vx * WALL_RESTITUTION;
            env->game_state.wall_bounces_this_rally++;
            env->game_state.puck_hit_wall_this_rally = true;
        }
    }
    
    if (env->puck.x + PUCK_RADIUS >= TABLE_WIDTH/2) {
        if (env->puck.y < goal_bottom || env->puck.y > goal_top) {
            env->puck.x = TABLE_WIDTH/2 - PUCK_RADIUS;
            env->puck.vx = -env->puck.vx * WALL_RESTITUTION;
            env->game_state.wall_bounces_this_rally++;
            env->game_state.puck_hit_wall_this_rally = true;
        }
    }
}

void handle_goal_scoring(TableHockey* env) {
    float goal_top = GOAL_WIDTH / 2;
    float goal_bottom = -GOAL_WIDTH / 2;
    
    if (env->puck.x <= -TABLE_WIDTH/2 && 
        env->puck.y >= goal_bottom && env->puck.y <= goal_top) {
        env->game_state.opponent_score++;
        env->game_state.goal_scored = true;

        if (env->game_state.puck_hit_wall_this_rally) {
            env->game_state.wall_bounce_goals++;
        } else {
            env->game_state.direct_goals++;
        }

        if (env->game_state.current_rally_length > env->game_state.max_rally_this_episode) {
            env->game_state.max_rally_this_episode = env->game_state.current_rally_length;
        }
        
        reset_puck_position(env);
        reset_rally_tracking(env);
    } else if (env->puck.x >= TABLE_WIDTH/2 && 
               env->puck.y >= goal_bottom && env->puck.y <= goal_top) {

        env->game_state.player_score++;
        env->game_state.goal_scored = true;

        if (env->game_state.puck_hit_wall_this_rally) {
            env->game_state.wall_bounce_goals++;
        } else {
            env->game_state.direct_goals++;
        }

        if (env->game_state.current_rally_length > env->game_state.max_rally_this_episode) {
            env->game_state.max_rally_this_episode = env->game_state.current_rally_length;
        }
        
        reset_puck_position(env);
        reset_rally_tracking(env);
    }
}

void reset_puck_position(TableHockey* env) {
    env->puck.x = 0.0f;
    env->puck.y = 0.0f;

    float angle = ((float)rand() / RAND_MAX - 0.5f) * 0.8f;
    float speed = 4.0f;
    
    env->puck.vx = -speed * cosf(angle);
    env->puck.vy = speed * sinf(angle);
}

void reset_rally_tracking(TableHockey* env) {
    env->game_state.wall_bounces_this_rally = 0;
    env->game_state.current_rally_length = 0;
    env->game_state.puck_hit_wall_this_rally = false;
    env->game_state.last_puck_vx = 0.0f;
    env->game_state.last_puck_vy = 0.0f;
}

// GAME LOGIC

void reset_game(TableHockey* env) {
    reset_puck_position(env);
    env->player_paddle.x = -TABLE_WIDTH/4;
    env->player_paddle.y = 0.0f;
    env->player_paddle.vx = 0.0f;
    env->player_paddle.vy = 0.0f;
    env->player_paddle.target_x = env->player_paddle.x;
    env->player_paddle.target_y = env->player_paddle.y;
    env->opponent_paddle.x = TABLE_WIDTH/4;
    env->opponent_paddle.y = 0.0f;
    env->opponent_paddle.vx = 0.0f;
    env->opponent_paddle.vy = 0.0f;
    env->opponent_paddle.target_x = env->opponent_paddle.x;
    env->opponent_paddle.target_y = env->opponent_paddle.y;
    env->game_state.player_score = 0;
    env->game_state.opponent_score = 0;
    env->game_state.episode_length = 0;
    env->game_state.puck_hits = 0;
    env->game_state.last_hit_time = 0;
    env->game_state.goal_scored = false;
    env->game_state.episode_done = false;
    env->game_state.wall_bounces_this_rally = 0;
    env->game_state.current_rally_length = 0;
    env->game_state.max_rally_this_episode = 0;
    env->game_state.total_puck_speed = 0.0f;
    env->game_state.total_shots = 0;
    env->game_state.defensive_hits = 0;
    env->game_state.total_blocks = 0;
    env->game_state.opponent_shots = 0;
    env->game_state.total_reaction_time = 0.0f;
    env->game_state.reaction_count = 0;
    env->game_state.player_distance_traveled = 0.0f;
    env->game_state.opponent_distance_traveled = 0.0f;
    env->game_state.player_min_x = env->game_state.player_max_x = 0.0f;
    env->game_state.player_min_y = env->game_state.player_max_y = 0.0f;
    env->game_state.wall_bounce_goals = 0;
    env->game_state.direct_goals = 0;
    env->game_state.puck_hit_wall_this_rally = false;
    env->game_state.last_puck_vx = 0.0f;
    env->game_state.last_puck_vy = 0.0f;
    env->game_state.prev_puck_x = env->puck.x;
    env->game_state.prev_puck_y = env->puck.y;
    env->game_state.player_vel_x = 0.0f;
    env->game_state.player_vel_y = 0.0f;
    env->game_state.opponent_vel_x = 0.0f;
    env->game_state.opponent_vel_y = 0.0f;
    env->game_state.time_since_last_hit = 0.0f;
    env->game_state.last_shot_power = 0.0f;
    env->game_state.consecutive_saves = 0;
    env->game_state.total_movement_this_episode = 0.0f;
    env->game_state.in_defensive_zone = false;
    env->game_state.distance_to_puck = 0.0f;
    env->game_state.prev_puck_x = env->puck.x;
    env->game_state.prev_puck_y = env->puck.y;
    env->game_state.player_vel_x = 0.0f;
    env->game_state.player_vel_y = 0.0f;
    env->game_state.opponent_vel_x = 0.0f;
    env->game_state.opponent_vel_y = 0.0f;
    env->game_state.time_since_last_hit = 0.0f;
    env->game_state.last_shot_power = 0.0f;
    env->game_state.consecutive_saves = 0;
    env->game_state.total_movement_this_episode = 0.0f;
    env->game_state.in_defensive_zone = false;
    env->game_state.distance_to_puck = 0.0f;
    
    env->episode_return = 0.0f;
    env->tick = 0;
}

void apply_actions(TableHockey* env) {
    if (env->action_mode == ACTION_CONTINUOUS) {
        env->player_paddle.vx = env->actions[0] * env->max_paddle_speed;
        env->player_paddle.vy = env->actions[1] * env->max_paddle_speed;
    } else {
        int player_action = (int)env->actions[0];
        env->player_paddle.vx = 0.0f;
        env->player_paddle.vy = 0.0f;
        
        switch (player_action) {
            case 0: break;
            case 1: env->player_paddle.vy = env->max_paddle_speed; break; // N
            case 2: env->player_paddle.vx = env->max_paddle_speed * 0.707f; env->player_paddle.vy = env->max_paddle_speed * 0.707f; break; // NE
            case 3: env->player_paddle.vx = env->max_paddle_speed; break; // E
            case 4: env->player_paddle.vx = env->max_paddle_speed * 0.707f; env->player_paddle.vy = -env->max_paddle_speed * 0.707f; break; // SE
            case 5: env->player_paddle.vy = -env->max_paddle_speed; break; // S
            case 6: env->player_paddle.vx = -env->max_paddle_speed * 0.707f; env->player_paddle.vy = -env->max_paddle_speed * 0.707f; break; // SW
            case 7: env->player_paddle.vx = -env->max_paddle_speed; break; // W
            case 8: env->player_paddle.vx = -env->max_paddle_speed * 0.707f; env->player_paddle.vy = env->max_paddle_speed * 0.707f; break; // NW
        }
    }
    
    if (env->render_mode == RENDER_HEADLESS) {
        float target_y = env->puck.y;
        float y_diff = target_y - env->opponent_paddle.y;
        env->opponent_paddle.vy = clamp_value(y_diff / env->dt, -env->max_paddle_speed, env->max_paddle_speed);
        env->opponent_paddle.vx = 0.0f;
    } else {
        env->opponent_paddle.vx = 0.0f;
        env->opponent_paddle.vy = 0.0f;
    }
}


void calculate_rewards(TableHockey* env) {
    float reward = 0.0f;
    if (env->game_state.goal_scored) {
        if (env->game_state.player_score > env->game_state.opponent_score) {
            reward = env->goal_reward;
        } else {
            reward = -env->goal_reward;
        }
        env->game_state.goal_scored = false;
    } else {
        if (env->tick - env->game_state.last_hit_time <= 1) {
            reward = env->puck_hit_reward;
            if (env->puck.x < -TABLE_WIDTH * 0.1f) {
                float defensive_bonus = env->puck_hit_reward * 0.5f;
                if (env->puck.x < -TABLE_WIDTH * 0.3f) {
                    defensive_bonus += env->puck_hit_reward * 0.5f;
                }
                reward += defensive_bonus;
            }
        }
    }

    float dist_to_puck = distance(env->player_paddle.x, env->player_paddle.y, 
                                  env->puck.x, env->puck.y);
    float max_dist = sqrtf(TABLE_WIDTH * TABLE_WIDTH + TABLE_HEIGHT * TABLE_HEIGHT);
    float distance_reward = 0.005f * (1.0f - (dist_to_puck / max_dist));
    
    if (env->puck.x < 0.0f) {
        distance_reward *= 0.3f;
    }
    
    float interception_reward = 0.0f;
    if (env->puck.vx < 0.0f) {
        float time_to_goal = (-TABLE_WIDTH/2 - env->puck.x) / env->puck.vx;
        float intercept_y = env->puck.y + env->puck.vy * time_to_goal;
        
        if (intercept_y >= -TABLE_HEIGHT/2 && intercept_y <= TABLE_HEIGHT/2 && time_to_goal > 0.0f) {
            float ideal_x = -TABLE_WIDTH/2 + PADDLE_RADIUS * 2;
            float ideal_y = clamp_value(intercept_y, -TABLE_HEIGHT/2 + PADDLE_RADIUS, TABLE_HEIGHT/2 - PADDLE_RADIUS);
            float dist_to_ideal = distance(env->player_paddle.x, env->player_paddle.y, ideal_x, ideal_y);
            float base_interception = 0.008f * (1.0f - (dist_to_ideal / max_dist));
            float urgency_multiplier = 1.0f;
            if (env->puck.x < 0.0f) {
                float proximity_to_goal = (-env->puck.x) / (TABLE_WIDTH/2);
                urgency_multiplier = 1.0f + 3.0f * proximity_to_goal;
            }
            
            interception_reward = base_interception * urgency_multiplier;
        }
    }
    
    float movement_reward = 0.0f;
    if (env->puck.vx < 0.0f) {
        float time_to_goal = (-TABLE_WIDTH/2 - env->puck.x) / env->puck.vx;
        float intercept_y = env->puck.y + env->puck.vy * time_to_goal;
        float ideal_x = -TABLE_WIDTH/2 + PADDLE_RADIUS * 2;
        float ideal_y = clamp_value(intercept_y, -TABLE_HEIGHT/2 + PADDLE_RADIUS, TABLE_HEIGHT/2 - PADDLE_RADIUS);
        
        float dx_to_ideal = ideal_x - env->player_paddle.x;
        float dy_to_ideal = ideal_y - env->player_paddle.y;
        float dist_to_ideal = sqrtf(dx_to_ideal * dx_to_ideal + dy_to_ideal * dy_to_ideal);
        
        if (dist_to_ideal > 0.01f) {
            dx_to_ideal /= dist_to_ideal;
            dy_to_ideal /= dist_to_ideal;
            float vel_toward_ideal = env->player_paddle.vx * dx_to_ideal + env->player_paddle.vy * dy_to_ideal;
            float base_movement = 0.006f * fmaxf(0.0f, vel_toward_ideal / env->max_paddle_speed);
            float urgency_multiplier = 1.0f;
            if (env->puck.x < 0.0f) {
                float proximity_to_goal = (-env->puck.x) / (TABLE_WIDTH/2);
                urgency_multiplier = 1.0f + 2.0f * proximity_to_goal;
            }
            
            movement_reward = base_movement * urgency_multiplier;
        }
    } else {
        float dx_to_puck = env->puck.x - env->player_paddle.x;
        float dy_to_puck = env->puck.y - env->player_paddle.y;
        float dist_to_puck_sq = dx_to_puck * dx_to_puck + dy_to_puck * dy_to_puck;
        
        if (dist_to_puck_sq > 0.01f) {
            dx_to_puck /= sqrtf(dist_to_puck_sq);
            dy_to_puck /= sqrtf(dist_to_puck_sq);
            float vel_toward_puck = env->player_paddle.vx * dx_to_puck + env->player_paddle.vy * dy_to_puck;
            movement_reward = 0.003f * fmaxf(0.0f, vel_toward_puck / env->max_paddle_speed);
        }
    }
    
    float penetration_penalty = 0.0f;
    if (env->puck.x < 0.0f) {
        float penetration_ratio = (-env->puck.x) / (TABLE_WIDTH/2);
        penetration_penalty = -0.02f * penetration_ratio * penetration_ratio;
        if (env->puck.x < -TABLE_WIDTH * 0.25f) {
            float critical_ratio = (-env->puck.x - TABLE_WIDTH * 0.25f) / (TABLE_WIDTH * 0.25f);
            penetration_penalty += -0.03f * critical_ratio;
        }
    }
    
    float defensive_reward = 0.0f;
    if (env->puck.x > 0.0f) {
        if (env->player_paddle.x < -TABLE_WIDTH/3) {
            defensive_reward = 0.001f;
        }
    }

    reward += distance_reward + interception_reward + movement_reward + defensive_reward + penetration_penalty;
    env->rewards[0] = reward;
    env->episode_return += reward;
}

void update_observations(TableHockey* env) {
    PhysicsCache* cache = &env->perf_cache;
    
    // 11 dimensions:
    // [0] Puck X position [-1, 1]
    // [1] Puck Y position [-1, 1]
    // [2] Puck X velocity [-1, 1]
    // [3] Puck Y velocity [-1, 1]
    // [4] Player paddle X position [-1, 1]
    // [5] Player paddle Y position [-1, 1]  
    // [6] Opponent paddle X position [-1, 1]
    // [7] Opponent paddle Y position [-1, 1]
    // [8] Player paddle X velocity [-1, 1]
    // [9] Player paddle Y velocity [-1, 1]
    // [10] dt (normalized to 1/60 = 1.0)
    
    env->observations[0] = env->puck.x * cache->inv_table_width_half;
    env->observations[1] = env->puck.y * cache->inv_table_height_half;
    env->observations[2] = env->puck.vx * cache->inv_puck_max_speed;
    env->observations[3] = env->puck.vy * cache->inv_puck_max_speed;
    env->observations[4] = env->player_paddle.x * cache->inv_table_width_half;
    env->observations[5] = env->player_paddle.y * cache->inv_table_height_half;
    env->observations[6] = env->opponent_paddle.x * cache->inv_table_width_half;
    env->observations[7] = env->opponent_paddle.y * cache->inv_table_height_half;
    env->observations[8] = env->player_paddle.vx / env->max_paddle_speed;
    env->observations[9] = env->player_paddle.vy / env->max_paddle_speed;
    env->observations[10] = env->dt * 60.0f;
}

void add_log(TableHockey* env) {
    env->log.player_goals += env->game_state.player_score;
    env->log.opponent_goals += env->game_state.opponent_score;
    env->log.episode_length += env->game_state.episode_length;
    env->log.puck_hits_per_episode += env->game_state.puck_hits;
    env->log.n += 1.0f;
    
    if (env->log.n > 0) {
     
        float wins = (env->game_state.player_score > env->game_state.opponent_score) ? 1.0f : 0.0f;
        env->log.win_rate = (env->log.win_rate * (env->log.n - 1) + wins) / env->log.n;
        env->log.avg_reward = (env->log.avg_reward * (env->log.n - 1) + env->episode_return) / env->log.n;
    }
    
    env->log.perf = env->log.win_rate;
}

// INPUT

void process_human_input(TableHockey* env) {
    if (env->render_mode == RENDER_HEADLESS) return;

    Vector2 mouse_pos = GetMousePosition();
    env->client->mouse_position = mouse_pos;
    float game_x = (mouse_pos.x - WINDOW_WIDTH/2) / SCALE;
    float game_y = (mouse_pos.y - WINDOW_HEIGHT/2) / SCALE;
    
    env->player_paddle.target_x = clamp_value(game_x, 
        -TABLE_WIDTH/2 + PADDLE_RADIUS, -PADDLE_RADIUS);
    env->player_paddle.target_y = clamp_value(game_y, 
        -TABLE_HEIGHT/2 + PADDLE_RADIUS, TABLE_HEIGHT/2 - PADDLE_RADIUS);

    float kb_speed = env->max_paddle_speed * env->dt;
    if (IsKeyDown(KEY_W)) env->player_paddle.target_y += kb_speed;
    if (IsKeyDown(KEY_S)) env->player_paddle.target_y -= kb_speed;
    if (IsKeyDown(KEY_A)) env->player_paddle.target_x -= kb_speed;
    if (IsKeyDown(KEY_D)) env->player_paddle.target_x += kb_speed;
    if (IsKeyPressed(KEY_F1)) {
        env->client->show_debug_info = !env->client->show_debug_info;
    }
}

void process_human_opponent_input(TableHockey* env) {
    if (env->render_mode == RENDER_HEADLESS) return;
    Vector2 mouse_pos = GetMousePosition();
    float game_x = (mouse_pos.x - WINDOW_WIDTH/2) / SCALE;
    float game_y = (mouse_pos.y - WINDOW_HEIGHT/2) / SCALE;
    float new_x = clamp_value(game_x, PADDLE_RADIUS, TABLE_WIDTH/2 - PADDLE_RADIUS);
    float new_y = clamp_value(game_y, -TABLE_HEIGHT/2 + PADDLE_RADIUS, TABLE_HEIGHT/2 - PADDLE_RADIUS);
    env->opponent_paddle.vx = (new_x - env->opponent_paddle.x) / env->dt;
    env->opponent_paddle.vy = (new_y - env->opponent_paddle.y) / env->dt;
    env->opponent_paddle.x = new_x;
    env->opponent_paddle.y = new_y;
}

void process_vr_input(TableHockey* env) {
    if (env->render_mode != RENDER_VR) return;
}

// RENDERING

Client* make_client(TableHockey* env) {
    if (env->render_mode == RENDER_HEADLESS) {
        return NULL;
    }
    
    Client* client = (Client*)calloc(1, sizeof(Client));
    client->render_mode = env->render_mode;
    client->show_debug_info = false;
    
    if (env->render_mode == RENDER_NORMAL) {
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Table Hockey - PufferLib");
        SetTargetFPS(60);
        client->window_initialized = true;
        
        // Setup 3D camera
        client->camera.position = (Vector3){ 0.0f, 8.0f, 6.0f };
        client->camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
        client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        client->camera.fovy = 45.0f;
        client->camera.projection = CAMERA_PERSPECTIVE;
    } else if (env->render_mode == RENDER_VR) {
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Table Hockey VR - PufferLib");
        SetTargetFPS(90);
        client->window_initialized = true;
        client->vr_initialized = false;
        
        client->camera.position = (Vector3){ 0.0f, 4.0f, 4.0f };
        client->camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
        client->camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
        client->camera.fovy = 60.0f;
        client->camera.projection = CAMERA_PERSPECTIVE;
    }
    
    return client;
}

void render_frame(TableHockey* env) {
    if (env->render_mode == RENDER_HEADLESS || !env->client) return;
    
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    
    if (env->render_mode == RENDER_NORMAL) {
        render_3d(env);
    } else if (env->render_mode == RENDER_VR) {
    }
    
    draw_ui(env);
    EndDrawing();
}

void render_2d(TableHockey* env) {
    Rectangle table = {
        WINDOW_WIDTH/2 - TABLE_WIDTH*SCALE/2,
        WINDOW_HEIGHT/2 - TABLE_HEIGHT*SCALE/2,
        TABLE_WIDTH*SCALE,
        TABLE_HEIGHT*SCALE
    };
    DrawRectangleLinesEx(table, 3, PUFF_WHITE);
    
    float goal_height = GOAL_WIDTH * SCALE;
    Rectangle left_goal = {
        table.x,
        table.y + (TABLE_HEIGHT*SCALE - goal_height)/2,
        -10,
        goal_height
    };
    Rectangle right_goal = {
        table.x + table.width,
        table.y + (TABLE_HEIGHT*SCALE - goal_height)/2,
        10,
        goal_height
    };
    DrawRectangleRec(left_goal, PUFF_RED);
    DrawRectangleRec(right_goal, PUFF_BLUE);
    
    draw_game_elements_2d(env);
}

void render_3d(TableHockey* env) {
    BeginMode3D(env->client->camera);
    DrawPlane((Vector3){0, 0, 0}, (Vector2){TABLE_WIDTH, TABLE_HEIGHT}, DARKGREEN);
    DrawCube((Vector3){0, 0.1f, TABLE_HEIGHT/2 + 0.05f}, TABLE_WIDTH, 0.2f, 0.1f, PUFF_WHITE);
    DrawCube((Vector3){0, 0.1f, -TABLE_HEIGHT/2 - 0.05f}, TABLE_WIDTH, 0.2f, 0.1f, PUFF_WHITE);
    DrawCube((Vector3){-TABLE_WIDTH/2 - 0.05f, 0.1f, 0}, 0.1f, 0.2f, TABLE_HEIGHT, PUFF_WHITE);
    DrawCube((Vector3){TABLE_WIDTH/2 + 0.05f, 0.1f, 0}, 0.1f, 0.2f, TABLE_HEIGHT, PUFF_WHITE);
    // Draw dynamic game objects (puck and paddles)
    draw_game_elements_3d(env);
    EndMode3D();
}

void draw_game_elements_2d(TableHockey* env) {
    Vector2 puck_pos = {
        WINDOW_WIDTH/2 + env->puck.x * SCALE,
        WINDOW_HEIGHT/2 - env->puck.y * SCALE
    };
    DrawCircleV(puck_pos, PUCK_RADIUS * SCALE, PUFF_WHITE);
    
    Vector2 player_pos = {
        WINDOW_WIDTH/2 + env->player_paddle.x * SCALE,
        WINDOW_HEIGHT/2 - env->player_paddle.y * SCALE
    };
    DrawCircleV(player_pos, PADDLE_RADIUS * SCALE, PUFF_RED);
    
    Vector2 opponent_pos = {
        WINDOW_WIDTH/2 + env->opponent_paddle.x * SCALE,
        WINDOW_HEIGHT/2 - env->opponent_paddle.y * SCALE
    };
    DrawCircleV(opponent_pos, PADDLE_RADIUS * SCALE, PUFF_BLUE);
}

void draw_game_elements_3d(TableHockey* env) {
    DrawCylinder((Vector3){env->puck.x, 0.02f, env->puck.y}, PUCK_RADIUS, PUCK_RADIUS, 0.04f, 16, PUFF_WHITE);
    DrawCylinder((Vector3){env->player_paddle.x, 0.05f, env->player_paddle.y}, 
                 PADDLE_RADIUS, PADDLE_RADIUS, 0.1f, 16, PUFF_RED);
    DrawCylinder((Vector3){env->opponent_paddle.x, 0.05f, env->opponent_paddle.y}, 
                 PADDLE_RADIUS, PADDLE_RADIUS, 0.1f, 16, PUFF_BLUE);
}

void draw_ui(TableHockey* env) {
    char score_text[64];
    snprintf(score_text, sizeof(score_text), "Player: %d  Opponent: %d", 
             env->game_state.player_score, env->game_state.opponent_score);
    DrawText(score_text, 10, 10, 20, PUFF_WHITE);
    
    char episode_text[64];
    snprintf(episode_text, sizeof(episode_text), " ", env->tick, MAX_STEPS);
    DrawText(episode_text, 10, 40, 16, PUFF_WHITE);
    
    if (env->client && env->client->show_debug_info) {
        char debug_text[256];
        snprintf(debug_text, sizeof(debug_text), 
                 "Puck: (%.2f, %.2f) v(%.2f, %.2f)\nPlayer: (%.2f, %.2f)\nHits: %d", 
                 env->puck.x, env->puck.y, env->puck.vx, env->puck.vy,
                 env->player_paddle.x, env->player_paddle.y,
                 env->game_state.puck_hits);
        DrawText(debug_text, 10, 70, 12, PUFF_CYAN);
    }
    
    if (env->render_mode != RENDER_HEADLESS) {
        DrawText(" ", 10, WINDOW_HEIGHT - 45, 14, PUFF_WHITE);
        DrawText(" ", 10, WINDOW_HEIGHT - 25, 12, GRAY);
    }
}

// SIMD PERFORMANCE OPTIMIZATIONS

void init_physics_cache(TableHockey* env) {
    PhysicsCache* cache = &env->perf_cache;

    cache->inv_table_width_half = 1.0f / (TABLE_WIDTH / 2);
    cache->inv_table_height_half = 1.0f / (TABLE_HEIGHT / 2);
    cache->inv_puck_max_speed = 1.0f / PUCK_MAX_SPEED;
    cache->inv_paddle_max_speed = 1.0f / MAX_SPEED;
    cache->collision_dist_squared = (PUCK_RADIUS + PADDLE_RADIUS) * (PUCK_RADIUS + PADDLE_RADIUS);

    for (int i = 0; i < 8; i++) {
        cache->wall_bounds[i] = 0.0f;
    }
    
    cache->wall_bounds[0] = -TABLE_WIDTH/2;  // left wall
    cache->wall_bounds[1] = TABLE_WIDTH/2;   // right wall
    cache->wall_bounds[2] = -TABLE_HEIGHT/2; // bottom wall
    cache->wall_bounds[3] = TABLE_HEIGHT/2;  // top wall
    cache->wall_bounds[4] = -GOAL_WIDTH/2;   // goal bottom
    cache->wall_bounds[5] = GOAL_WIDTH/2;    // goal top
}

// UTILITY FUNCTIONS

float distance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx*dx + dy*dy);
}

float clamp_value(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

void normalize_vector(float* x, float* y) {
    float length = sqrtf((*x) * (*x) + (*y) * (*y));
    if (length > 0) {
        *x /= length;
        *y /= length;
    }
}

float vector_length(float x, float y) {
    return sqrtf(x*x + y*y);
}