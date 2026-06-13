#include <time.h>
#include <ctype.h>
#include <string.h>
#include "bat.h"

#define DEMO_CONFIG_PATH "config/bat.ini"

static char* trim(char* s) {
    while (isspace((unsigned char)*s)) s++;
    char* end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void apply_env_config_value(Bat* env, const char* key, float value) {
    if (strcmp(key, "max_speed") == 0) env->max_speed = value;
    else if (strcmp(key, "min_speed") == 0) env->min_speed = value;
    else if (strcmp(key, "accel") == 0) env->accel = value;
    else if (strcmp(key, "turn_rate") == 0) env->turn_rate = value;
    else if (strcmp(key, "render_target_fps") == 0) env->render_target_fps = (int)value;
    else if (strcmp(key, "record_video") == 0) env->record_video = (int)value;
    else if (strcmp(key, "record_video_fps") == 0) env->record_video_fps = (int)value;
    else if (strcmp(key, "record_video_seconds") == 0) env->record_video_seconds = (int)value;
    else if (strcmp(key, "record_video_audio") == 0) env->record_video_audio = (int)value;
    else if (strcmp(key, "bug_echo_farther_penalty_scale") == 0) env->bug_echo_farther_penalty_scale = value;
    else if (strcmp(key, "bug_echo_reward_scale") == 0) env->bug_echo_reward_scale = value;
    else if (strcmp(key, "bug_wing_sideband_gain") == 0) env->bug_wing_sideband_gain = value;
    else if (strcmp(key, "curriculum_initial_level") == 0) env->curriculum_initial_level = (int)value;
    else if (strcmp(key, "curriculum_obstacle_step") == 0) env->curriculum_obstacle_step = (int)value;
    else if (strcmp(key, "curriculum_start_bug_distance") == 0) env->curriculum_start_bug_distance = value;
    else if (strcmp(key, "curriculum_successes_per_level") == 0) env->curriculum_successes_per_level = (int)value;
    else if (strcmp(key, "ear_separation_scale") == 0) env->ear_separation_scale = value;
    else if (strcmp(key, "ear_rear_gain") == 0) env->ear_rear_gain = value;
    else if (strcmp(key, "ear_front_gain") == 0) env->ear_front_gain = value;
    else if (strcmp(key, "ear_side_gain") == 0) env->ear_side_gain = value;
    else if (strcmp(key, "early_chirp_penalty") == 0) env->early_chirp_penalty = value;
    else if (strcmp(key, "progress_reward_scale") == 0) env->progress_reward_scale = value;
    else if (strcmp(key, "reflector_strength") == 0) env->reflector_strength = value;
    else if (strcmp(key, "sound_speed") == 0) env->sound_speed = value;
    else if (strcmp(key, "step_cost") == 0) env->step_cost = value;
    else if (strcmp(key, "valid_chirp_reward") == 0) env->valid_chirp_reward = value;
    else if (strcmp(key, "chirp_cooldown_ticks") == 0) env->chirp_cooldown_ticks = (int)value;
    else if (strcmp(key, "chirp_efficiency_reward") == 0) env->chirp_efficiency_reward = value;
    else if (strcmp(key, "chirp_overlap_penalty") == 0) env->chirp_overlap_penalty = value;
    else if (strcmp(key, "collision_penalty") == 0) env->collision_penalty = value;
}

static void load_env_config(Bat* env, const char* path) {
    FILE* file = fopen(path, "r");
    if (file == NULL) return;

    bool in_env = false;
    char line[256];
    while (fgets(line, sizeof(line), file) != NULL) {
        char* s = trim(line);
        if (*s == '\0' || *s == '#' || *s == ';') continue;
        if (*s == '[') {
            in_env = strcmp(s, "[env]") == 0;
            continue;
        }
        if (!in_env) continue;

        char* eq = strchr(s, '=');
        if (eq == NULL) continue;
        *eq = '\0';
        char* key = trim(s);
        char* raw_value = trim(eq + 1);
        apply_env_config_value(env, key, strtof(raw_value, NULL));
    }

    fclose(file);
}

void demo() {
    Bat env = {
        .num_agents = NUM_AGENTS,
        .render_target_fps = 60,
        .record_video_fps = 30,
        .record_video_seconds = 30,
        .record_video_audio = 1,
    };
    load_env_config(&env, DEMO_CONFIG_PATH);
    env.rng = (unsigned int)time(NULL);
    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);

    while (!WindowShouldClose()) {
        memset(env.actions, 0, sizeof(float) * NUM_ACTIONS);
        if (IsKeyDown(KEY_W)) env.actions[ACTION_MOVE] = THRUST_FORWARD;
        if (IsKeyDown(KEY_S)) env.actions[ACTION_MOVE] = BRAKE;
        if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) env.actions[ACTION_TURN] = TURN_LEFT;
        if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) env.actions[ACTION_TURN] = TURN_RIGHT;
        env.actions[ACTION_CHIRP_FREQ_END] = CHIRP_FREQ_BINS - 1;
        env.actions[ACTION_CHIRP_DURATION] = 1;
        env.actions[ACTION_CHIRP_EMIT] = IsKeyDown(KEY_SPACE) ? 1.0f : 0.0f;
        c_step(&env);
        c_render(&env);
    }

    close_client(env.client);
    free_allocated(&env);
}

int main() {
    demo();
    return 0;
}
