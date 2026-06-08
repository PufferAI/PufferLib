#include "chess.h"
// Before embedding approach
// #define OBS_SIZE 1082
#define OBS_SIZE 167
#define NUM_ATNS 1
#define ACT_SIZES {97}
#define OBS_TENSOR_T ByteTensor
#define MY_ACTION_MASK 97

#define MY_VEC_INIT
#define MY_VEC_CLOSE
#define MY_USES_PERM
#define MY_USES_TAGS
#define Env Chess
#include "vecenv.h"

void my_setup_perm(StaticVec* vec, Env* env, int slot_base) {
    size_t obs_elem_size = obs_element_size();
    for (int s = 0; s < env->num_agents; s++) {
        int phys = vec->agent_perm ? vec->agent_perm[slot_base + s] : (slot_base + s);
        env->obs_ptr[s]         = (uint8_t*)vec->observations + (size_t)phys * OBS_SIZE * obs_elem_size;
        env->action_mask_ptr[s] = vec->action_mask + (size_t)phys * MY_ACTION_MASK;
        env->action_ptr[s]      = vec->actions + (size_t)phys * NUM_ATNS;
        env->reward_ptr[s]      = vec->rewards + phys;
        env->terminal_ptr[s]    = vec->terminals + phys;
    }
}

#define DEFAULT_STARTING_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#define FEN_CURRICULUM_PATH "resources/chess/fens.txt"

static char** SHARED_FEN_CURRICULUM = NULL;
static int SHARED_NUM_FENS = 0;

static char** load_fen_file(const char* path, int* num_fens_out) {
    FILE* f = fopen(path, "r");
    if (f == NULL) {
        *num_fens_out = 0;
        return NULL;
    }

    int num_fens = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '#' && line[0] != '\n' && line[0] != '\r') {
            num_fens++;
        }
    }
    if (num_fens == 0) {
        fclose(f);
        *num_fens_out = 0;
        return NULL;
    }

    char** fens = (char**)malloc(num_fens * sizeof(char*));
    rewind(f);
    int idx = 0;
    while (fgets(line, sizeof(line), f) && idx < num_fens) {
        if (line[0] != '#' && line[0] != '\n' && line[0] != '\r') {
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
                line[--len] = '\0';
            }
            fens[idx++] = strdup(line);
        }
    }
    fclose(f);
    *num_fens_out = num_fens;
    return fens;
}

static void apply_kwargs(Env* env, Dict* kwargs) {
    env->max_moves = (int)dict_get(kwargs, "max_moves")->value;
    env->reward_draw = (float)dict_get(kwargs, "reward_draw")->value;
    env->reward_invalid_piece = (float)dict_get(kwargs, "reward_invalid_piece")->value;
    env->reward_invalid_move = (float)dict_get(kwargs, "reward_invalid_move")->value;
    env->reward_repetition = (float)dict_get(kwargs, "reward_repetition")->value;
    env->render_fps = (int)dict_get(kwargs, "render_fps")->value;
    env->mode = (int)dict_get(kwargs, "mode")->value;
    env->enable_50_move_rule = (int)dict_get(kwargs, "enable_50_move_rule")->value;
    env->enable_threefold_repetition = (int)dict_get(kwargs, "enable_threefold_repetition")->value;
    env->random_fen = (int)dict_get(kwargs, "random_fen")->value;
    env->fen_curric_pct = (float)dict_get(kwargs, "fen_curric_pct")->value;

    env->client = NULL;
    env->legal_dirty = 1;
    env->human_color = -1;
    env->log_pgn = 0;
    env->log_pgn_choice_made = 1;
    env->pgn_filename[0] = '\0';
    env->pgn_game_number = 0;
    env->maia_pid = 0;
    env->maia_stdin_fd = -1;
    env->maia_stdout_fd = -1;
    env->maia_phase = 0;
    strcpy(env->starting_fen, DEFAULT_STARTING_FEN);
    strcpy(env->last_result, "Game starting...");
}

Env* my_vec_init(int* num_envs_out, int* buffer_env_starts, int* buffer_env_counts,
                 Dict* vec_kwargs, Dict* env_kwargs) {
    int total_agents = (int)dict_get(vec_kwargs, "total_agents")->value;
    int num_buffers = (int)dict_get(vec_kwargs, "num_buffers")->value;
    int agents_per_buffer = total_agents / num_buffers;

    float curric_pct = (float)dict_get(env_kwargs, "fen_curric_pct")->value;
    if (curric_pct > 0.0f && SHARED_FEN_CURRICULUM == NULL) {
        SHARED_FEN_CURRICULUM = load_fen_file(FEN_CURRICULUM_PATH, &SHARED_NUM_FENS);
        if (SHARED_FEN_CURRICULUM != NULL) {
            printf("Loaded %d FENs from %s\n", SHARED_NUM_FENS, FEN_CURRICULUM_PATH);
        }
    }

    int mode = (int)dict_get(env_kwargs, "mode")->value;
    int agents_per_env = (mode == CHESS_MODE_SELFPLAY) ? 2 : 1;
    int num_envs = total_agents / agents_per_env;
    Env* envs = (Env*)calloc(num_envs, sizeof(Env));
    for (int i = 0; i < num_envs; i++) {
        Env* env = &envs[i];
        apply_kwargs(env, env_kwargs);
        env->num_agents = agents_per_env;
        env->rng = i;
        // In selfplay, learner_color is unused; the slot↔color mapping is per-env
        // randomized so policies in fixed slots see both colors equally.
        env->learner_color = (agents_per_env == 1) ? (i % 2) : CHESS_WHITE;
        if (agents_per_env == 2 && (i & 1)) {
            env->slot_for_color[CHESS_WHITE] = 1;
            env->slot_for_color[CHESS_BLACK] = 0;
        } else {
            env->slot_for_color[CHESS_WHITE] = 0;
            env->slot_for_color[CHESS_BLACK] = 1;
        }
        env->fen_curriculum = SHARED_FEN_CURRICULUM;
        env->num_fens = SHARED_NUM_FENS;
        init_bitboards();
    }

    int buf = 0;
    int buf_agents = 0;
    buffer_env_starts[0] = 0;
    buffer_env_counts[0] = 0;
    for (int i = 0; i < num_envs; i++) {
        buf_agents += agents_per_env;
        buffer_env_counts[buf]++;
        if (buf_agents >= agents_per_buffer && buf < num_buffers - 1) {
            buf++;
            buffer_env_starts[buf] = i + 1;
            buffer_env_counts[buf] = 0;
            buf_agents = 0;
        }
    }

    *num_envs_out = num_envs;
    return envs;
}

void my_vec_close(Env* envs) {
    if (SHARED_FEN_CURRICULUM != NULL) {
        for (int i = 0; i < SHARED_NUM_FENS; i++) {
            free(SHARED_FEN_CURRICULUM[i]);
        }
        free(SHARED_FEN_CURRICULUM);
        SHARED_FEN_CURRICULUM = NULL;
        SHARED_NUM_FENS = 0;
    }
}

void my_init(Env* env, Dict* kwargs) {
    apply_kwargs(env, kwargs);
    env->num_agents = (env->mode == CHESS_MODE_SELFPLAY) ? 2 : 1;
    env->learner_color = (env->num_agents == 1) ? CHESS_WHITE : CHESS_WHITE;
    env->slot_for_color[CHESS_WHITE] = 0;
    env->slot_for_color[CHESS_BLACK] = 1;
    env->fen_curriculum = NULL;
    env->num_fens = 0;
    init_bitboards();
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "draw_rate", log->draw_rate);
    dict_set(out, "timeout_rate", log->timeout_rate);
    dict_set(out, "chess_moves", log->chess_moves);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "invalid_action_rate", log->invalid_action_rate);
    dict_set(out, "slot_0_score", log->slot_0_score);
    dict_set(out, "slot_1_score", log->slot_1_score);
    dict_set(out, "hist_score", log->hist_score);
    dict_set(out, "hist_n", log->hist_n);
    // Per-bank historical stats for multi-bank selfplay. selfplay.py reads
    // hist_score_bank_<b> / hist_n_bank_<b> to drive each bank's swap decision.
    // dict_set stores the key pointer (vecenv.h:61), not a copy, so we MUST
    // use string literals here — a stack buffer in a loop aliases and collapses
    // all 16 entries into one. Sized to CHESS_MAX_BANKS = 8.
    dict_set(out, "hist_score_bank_0", log->hist_score_bank[0]);
    dict_set(out, "hist_score_bank_1", log->hist_score_bank[1]);
    dict_set(out, "hist_score_bank_2", log->hist_score_bank[2]);
    dict_set(out, "hist_score_bank_3", log->hist_score_bank[3]);
    dict_set(out, "hist_score_bank_4", log->hist_score_bank[4]);
    dict_set(out, "hist_score_bank_5", log->hist_score_bank[5]);
    dict_set(out, "hist_score_bank_6", log->hist_score_bank[6]);
    dict_set(out, "hist_score_bank_7", log->hist_score_bank[7]);
    dict_set(out, "hist_n_bank_0", log->hist_n_bank[0]);
    dict_set(out, "hist_n_bank_1", log->hist_n_bank[1]);
    dict_set(out, "hist_n_bank_2", log->hist_n_bank[2]);
    dict_set(out, "hist_n_bank_3", log->hist_n_bank[3]);
    dict_set(out, "hist_n_bank_4", log->hist_n_bank[4]);
    dict_set(out, "hist_n_bank_5", log->hist_n_bank[5]);
    dict_set(out, "hist_n_bank_6", log->hist_n_bank[6]);
    dict_set(out, "hist_n_bank_7", log->hist_n_bank[7]);
    dict_set(out, "wins_as_white", log->wins_as_white);
    dict_set(out, "wins_as_black", log->wins_as_black);
    dict_set(out, "games_as_white", log->games_as_white);
    dict_set(out, "games_as_black", log->games_as_black);
    dict_set(out, "maia_failures", log->maia_failures);
}
