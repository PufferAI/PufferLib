#include <time.h>
#include <ctype.h>
#include <string.h>
#include "bat.h"

#define BAT_DEMO_CONFIG_PATH "config/bat.ini"

static char* trim(char* s) {
    while (isspace((unsigned char)*s)) s++;
    char* end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return s;
}

static void set_demo_defaults(Bat* env) {
    *env = (Bat){
        .num_agents = 1,
        .frameskip = 1,
        .bat_max_speed = 15.498233877318418f,
        .bat_min_speed = 2.6389946132676654f,
        .bat_accel = 53.02330161128345f,
        .bat_turn_rate = 8.371655963408276f,
        .max_steps = 512,
        .render_target_fps = 60,
        .record_video = 0,
        .record_video_fps = 30,
        .record_video_seconds = 30,
        .record_video_audio = 1,
        .bug_echo_farther_penalty_scale = 0.19351291407677712f,
        .bug_echo_reward_scale = 0.35f,
        .bug_wing_sideband_gain = 0.19056934455600955f,
        .curriculum_initial_level = 1,
        .curriculum_obstacle_step = 8,
        .curriculum_start_bug_distance = 8.438008720355143f,
        .curriculum_successes_per_level = 4,
        .ear_separation_scale = 2.0f,
        .ear_rear_gain = 0.22038613968607276f,
        .ear_front_gain = 0.6419214149115183f,
        .ear_side_gain = 0.28043867572747055f,
        .early_chirp_penalty = 0.006f,
        .inbound_bug_speed_multiplier = 1.75f,
        .inbound_heading_noise_degrees = 18.0f,
        .max_chirp_age_ticks = 30,
        .max_chirps_per_episode = 15,
        .max_echo_range = 128.0f,
        .progress_reward_scale = 0.12f,
        .reflector_strength = 0.6f,
        .sound_speed = 180.0f,
        .step_cost = 0.00010781401476030468f,
        .valid_chirp_reward = 0.00015478540834814922f,
        .chirp_cooldown_ticks = 11,
        .chirp_efficiency_reward = 2.0f,
        .chirp_overlap_penalty = 0.004278154705335052f,
        .collision_penalty = 1.950717141233687f,
    };
}

static void apply_env_config_value(Bat* env, const char* key, float value) {
    if (strcmp(key, "frameskip") == 0) env->frameskip = (int)value;
    else if (strcmp(key, "bat_max_speed") == 0) env->bat_max_speed = value;
    else if (strcmp(key, "bat_min_speed") == 0) env->bat_min_speed = value;
    else if (strcmp(key, "bat_accel") == 0) env->bat_accel = value;
    else if (strcmp(key, "bat_turn_rate") == 0) env->bat_turn_rate = value;
    else if (strcmp(key, "max_steps") == 0) env->max_steps = (int)value;
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
    else if (strcmp(key, "inbound_bug_speed_multiplier") == 0) env->inbound_bug_speed_multiplier = value;
    else if (strcmp(key, "inbound_heading_noise_degrees") == 0) env->inbound_heading_noise_degrees = value;
    else if (strcmp(key, "max_chirp_age_ticks") == 0) env->max_chirp_age_ticks = (int)value;
    else if (strcmp(key, "max_chirps_per_episode") == 0) env->max_chirps_per_episode = (int)value;
    else if (strcmp(key, "max_echo_range") == 0) env->max_echo_range = value;
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
    Bat env;
    set_demo_defaults(&env);
    load_env_config(&env, BAT_DEMO_CONFIG_PATH);
    env.rng = (unsigned int)time(NULL);
    allocate(&env);
    env.client = make_client(&env);
    c_reset(&env);

    SetTargetFPS(60);
    while (!WindowShouldClose()) {
        memset(env.actions, 0, sizeof(float) * BAT_NUM_ACTIONS);
        env.actions[0] = BAT_NOOP;
        env.actions[1] = BAT_TURN_NONE;
        if (IsKeyDown(KEY_W)) env.actions[0] = BAT_THRUST_FORWARD;
        if (IsKeyDown(KEY_S)) env.actions[0] = BAT_BRAKE;
        if (IsKeyDown(KEY_A) || IsKeyDown(KEY_LEFT)) env.actions[1] = BAT_TURN_LEFT;
        if (IsKeyDown(KEY_D) || IsKeyDown(KEY_RIGHT)) env.actions[1] = BAT_TURN_RIGHT;
        env.actions[2] = 0;
        env.actions[3] = 7;
        env.actions[4] = 1;
        env.actions[5] = IsKeyDown(KEY_SPACE) ? 1.0f : 0.0f;
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
