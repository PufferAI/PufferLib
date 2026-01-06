#include "raylib.h"
#include "tetrominoes.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }

#define HALF_LINEWIDTH 1
#define SQUARE_SIZE 32
#define DECK_SIZE (2 * NUM_TETROMINOES) // To implement the 7-bag system
#define NUM_PREVIEW 2
#define NUM_FLOAT_OBS 6

#define ACTION_NO_OP 0
#define ACTION_LEFT 1
#define ACTION_RIGHT 2
#define ACTION_ROTATE 3
#define ACTION_SOFT_DROP 4
#define ACTION_HARD_DROP 5
#define ACTION_HOLD 6

#define MAX_TICKS 10000
#define PERSONAL_BEST 67890
#define INITIAL_TICKS_PER_FALL 6 // how many ticks before the tetromino naturally falls down of one square
#define GARBAGE_KICKOFF_TICK 500
#define INITIAL_TICKS_PER_GARBAGE 100

#define LINES_PER_LEVEL 10
// Revisit scoring with level. See https://tetris.wiki/Scoring
#define SCORE_SOFT_DROP 1
#define REWARD_SOFT_DROP 0.0f
#define SCORE_HARD_DROP 2
#define REWARD_HARD_DROP 0.02f
#define REWARD_ROTATE 0.01f
#define REWARD_INVALID_ACTION 0.0f

const int SCORE_COMBO[5] = {0, 100, 300, 500, 1000};
const float REWARD_COMBO[5] = {0, 0.1, 0.3, 0.5, 1.0};

typedef struct Log {
	float perf;
	float score;
	float ep_length;
	float ep_return;
	float lines_deleted;
	float avg_combo;
	float atn_frac_soft_drop;
	float atn_frac_hard_drop;
	float atn_frac_rotate;
	float atn_frac_hold;
	float game_level;
	float ticks_per_line;
	float n;
} Log;

typedef struct Client {
	int total_cols;
	int total_rows;
	int ui_rows;
	int deck_rows;
	int preview_target_rotation;
	int preview_target_col;
} Client;

typedef struct Tetris {
	Client *client;
	Log log;
	float *observations;
	int *actions;
	float *rewards;
	unsigned char *terminals;
	int dim_obs;
	int num_float_obs;

	int n_rows;
	int n_cols;
	bool use_deck_obs;
	int n_noise_obs;
	int n_init_garbage;

	int *grid;
	int tick;
	int tick_fall;
	int ticks_per_fall;
	int tick_garbage;
	int ticks_per_garbage;
	int score;
	int can_swap;

	int *tetromino_deck;
	int hold_tetromino;
	int cur_position_in_deck;
	int cur_tetromino;
	int cur_tetromino_row;
	int cur_tetromino_col;
	int cur_tetromino_rot;

	float ep_return;
	int lines_deleted;
	int count_combos;
	int game_level;
	int atn_count_hard_drop;
	int atn_count_soft_drop;
	int atn_count_rotate;
	int atn_count_hold;
	int tetromino_counts[NUM_TETROMINOES];
} Tetris;

void init(Tetris *env) {
	env->grid = (int *)calloc(env->n_rows * env->n_cols, sizeof(int));
	if (env->grid == NULL) {
		exit(1);
	}
	env->tetromino_deck = calloc(DECK_SIZE, sizeof(int));
	if (env->tetromino_deck == NULL) {
		exit(1);
	}
}

void allocate(Tetris *env) {
	init(env);
	// grid, 6 floats, 4 one-hot tetrominoes encode (current, previews, hold) + self-inflicting noisy action bits
	env->dim_obs = env->n_cols * env->n_rows + NUM_FLOAT_OBS + NUM_TETROMINOES * (NUM_PREVIEW + 2) + env->n_noise_obs;
	env->observations = (float *)calloc(env->dim_obs, sizeof(float));
	env->actions = (int *)calloc(1, sizeof(int));
	env->rewards = (float *)calloc(1, sizeof(float));
	env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
}

void c_close(Tetris *env) {
	free(env->grid);
	free(env->tetromino_deck);
}

void free_allocated(Tetris *env) {
	free(env->actions);
	free(env->observations);
	free(env->terminals);
	free(env->rewards);
	c_close(env);
}

void add_log(Tetris *env) {
	env->log.score += env->score;
	env->log.perf += env->score / ((float)PERSONAL_BEST);
	env->log.ep_length += env->tick;
	env->log.ep_return += env->ep_return;
	env->log.lines_deleted += env->lines_deleted;
	env->log.avg_combo += env->count_combos > 0 ? ((float)env->lines_deleted) / ((float)env->count_combos) : 1.0f;
	env->log.atn_frac_hard_drop += env->atn_count_hard_drop / ((float)env->tick);
	env->log.atn_frac_soft_drop += env->atn_count_soft_drop / ((float)env->tick);
	env->log.atn_frac_rotate += env->atn_count_rotate / ((float)env->tick);
	env->log.atn_frac_hold += env->atn_count_hold / ((float)env->tick);
	env->log.game_level += env->game_level;
	env->log.ticks_per_line += (env->lines_deleted > 0) ? ((float)env->tick / (float)env->lines_deleted) : (float)env->tick;
	env->log.n += 1;
}

void compute_observations(Tetris *env) {
	// content of the grid: 0 for empty, 1 for placed blocks, 2 for the current tetromino
	for (int i = 0; i < env->n_cols * env->n_rows; i++) {
		env->observations[i] = env->grid[i] != 0;
	}

	for (int r = 0; r < SIZE; r++) {
		for (int c = 0; c < SIZE; c++) {
			if (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1) {
				env->observations[(env->cur_tetromino_row + r) * env->n_cols + c + env->cur_tetromino_col] = 2;
			}
		}
	}
	int offset = env->n_cols * env->n_rows;
	env->observations[offset] = env->tick / ((float)MAX_TICKS);
	env->observations[offset + 1] = env->tick_fall / ((float)env->ticks_per_fall);
	env->observations[offset + 2] = env->cur_tetromino_row / ((float)env->n_rows);
	env->observations[offset + 3] = env->cur_tetromino_col / ((float)env->n_cols);
	env->observations[offset + 4] = env->cur_tetromino_rot;
	env->observations[offset + 5] = env->can_swap;
	offset += NUM_FLOAT_OBS;

	// Zero out the one-hot encoded part of the observations for deck and hold.
	memset(env->observations + offset, 0, NUM_TETROMINOES * (NUM_PREVIEW + 2) * sizeof(float));
	if (env->use_deck_obs) {
		// Deck, one hot encoded
		int tetromino_id;
		for (int j = 0; j < NUM_PREVIEW + 1; j++) {
			tetromino_id = env->tetromino_deck[(env->cur_position_in_deck + j) % DECK_SIZE];
			env->observations[offset + tetromino_id] = 1;
			offset += NUM_TETROMINOES;
		}
		
		// Hold, one hot encoded
		if (env->hold_tetromino > -1) {
			env->observations[offset + env->hold_tetromino] = 1;
		}
		offset += NUM_TETROMINOES;
	} else {
		offset += NUM_TETROMINOES * (NUM_PREVIEW + 2);
	}

	// Turn off noise bits, one-by-one.
	if (env->n_noise_obs > 0) {
		env->observations[offset + rand() % env->n_noise_obs] = 0;
	}
}

void restore_grid(Tetris *env) { memset(env->grid, 0, env->n_rows * env->n_cols * sizeof(int)); }

void refill_and_shuffle(int *array) {
	// Hold can change the deck distribution, so need to refill
	for (int i = 0; i < NUM_TETROMINOES; i++) {
		array[i] = i;
	}

	// Fisher-Yates shuffle
	for (int i = NUM_TETROMINOES - 1; i > 0; i--) {
		int j = rand() % (i + 1);
		int temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}
}

void initialize_deck(Tetris *env) {
	// Implements a 7-bag system. The deck is composed of two bags.
	refill_and_shuffle(env->tetromino_deck); // First bag
	refill_and_shuffle(env->tetromino_deck + NUM_TETROMINOES); // Second bag
	env->cur_position_in_deck = 0;
	env->cur_tetromino = env->tetromino_deck[env->cur_position_in_deck];
}

void spawn_new_tetromino(Tetris *env) {
	env->cur_position_in_deck = (env->cur_position_in_deck + 1) % DECK_SIZE;
	env->cur_tetromino = env->tetromino_deck[env->cur_position_in_deck];
	env->cur_tetromino_rot = 0;

	if (env->cur_position_in_deck == 0) {
		// Now using the first bag, so shuffle the second bag
		refill_and_shuffle(env->tetromino_deck + NUM_TETROMINOES);
	} else if (env->cur_position_in_deck == NUM_TETROMINOES) {
		// Now using the second bag, so shuffle the first bag
		refill_and_shuffle(env->tetromino_deck);
	}

	env->cur_tetromino_col = env->n_cols / 2;
	env->cur_tetromino_row = 0;
	env->tick_fall = 0;
	env->tetromino_counts[env->cur_tetromino]++;
}

// This is only used to check if the game is done
bool can_spawn_new_tetromino(Tetris *env) {
	int next_pos = (env->cur_position_in_deck + 1) % DECK_SIZE;
	int next_tetromino = env->tetromino_deck[next_pos];
	for (int c = 0; c < TETROMINOES_FILLS_COL[next_tetromino][0]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[next_tetromino][0]; r++) {
			if ((env->grid[r * env->n_cols + c + env->n_cols / 2] != 0) && (TETROMINOES[next_tetromino][0][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_soft_drop(Tetris *env) {
	if (env->cur_tetromino_row == (env->n_rows - TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot])) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row + 1) * env->n_cols + c + env->cur_tetromino_col] != 0) &&
			    (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_go_left(Tetris *env) {
	if (env->cur_tetromino_col == 0) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col - 1] != 0) &&
			    (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_go_right(Tetris *env) {
	if (env->cur_tetromino_col == (env->n_cols - TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot])) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col + 1] != 0) &&
			    (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_hold(Tetris *env) {
	if (env->can_swap == 0) {
		return false;
	}
	if (env->hold_tetromino == -1) {
		return true;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->hold_tetromino][env->cur_tetromino_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->hold_tetromino][env->cur_tetromino_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col] != 0) &&
			    (TETROMINOES[env->hold_tetromino][env->cur_tetromino_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool can_rotate(Tetris *env) {
	int next_rot = (env->cur_tetromino_rot + 1) % NUM_ROTATIONS;
	if (env->cur_tetromino_col > (env->n_cols - TETROMINOES_FILLS_COL[env->cur_tetromino][next_rot])) {
		return false;
	}
	if (env->cur_tetromino_row > (env->n_rows - TETROMINOES_FILLS_ROW[env->cur_tetromino][next_rot])) {
		return false;
	}
	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][next_rot]; c++) {
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][next_rot]; r++) {
			if ((env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col] != 0) &&
			    (TETROMINOES[env->cur_tetromino][next_rot][r][c] == 1)) {
				return false;
			}
		}
	}
	return true;
}

bool is_full_row(Tetris *env, int row) {
	for (int c = 0; c < env->n_cols; c++) {
		if (env->grid[row * env->n_cols + c] == 0) {
			return false;
		}
	}
	return true;
}

void clear_row(Tetris *env, int row) {
	for (int r = row; r > 0; r--) {
		for (int c = 0; c < env->n_cols; c++) {
			env->grid[r * env->n_cols + c] = env->grid[(r - 1) * env->n_cols + c];
		}
	}
	for (int c = 0; c < env->n_cols; c++) {
		env->grid[c] = 0;
	}
}

void add_garbage_lines(Tetris *env, int num_lines, int num_holes) {
	// Check if adding garbage would cause an immediate game over
	for (int r = 0; r < num_lines; r++) {
		for (int c = 0; c < env->n_cols; c++) {
			if (env->grid[r * env->n_cols + c] != 0) {
				env->terminals[0] = 1; // Game over
				return;
			}
		}
	}

	// Shift the existing grid up by num_lines
	for (int r = 0; r < env->n_rows - num_lines; r++) {
		memcpy(&env->grid[r * env->n_cols], &env->grid[(r + num_lines) * env->n_cols],
		       env->n_cols * sizeof(int));
	}

	// Add new garbage lines at the bottom
	for (int r = env->n_rows - num_lines; r < env->n_rows; r++) {
		// First, fill the entire row with garbage
		for (int c = 0; c < env->n_cols; c++) {
			env->grid[r * env->n_cols + c] = -(rand() % NUM_TETROMINOES + 1);
		}

		// Create holes by selecting distinct columns
		int cols[env->n_cols];
		for (int i = 0; i < env->n_cols; i++) {
			cols[i] = i;
		}
		// Shuffle column indices
		for (int i = env->n_cols - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			int temp = cols[i];
			cols[i] = cols[j];
			cols[j] = temp;
		}
		for (int i = 0; i < num_holes; i++) {
			env->grid[r * env->n_cols + cols[i]] = 0;
		}
	}

	// Move the current piece up as well
	env->cur_tetromino_row = max(0, env->cur_tetromino_row - num_lines);
}

void c_reset(Tetris *env) {
	env->score = 0;
	env->hold_tetromino = -1;
	env->tick = 0;
	env->game_level = 1;
	env->ticks_per_fall = INITIAL_TICKS_PER_FALL;
	env->tick_fall = 0;
	env->ticks_per_garbage = INITIAL_TICKS_PER_GARBAGE;
	env->tick_garbage = 0;
	env->can_swap = 1;

	env->ep_return = 0.0;
	env->count_combos = 0;
	env->lines_deleted = 0;
	env->atn_count_hard_drop = 0;
	env->atn_count_soft_drop = 0;
	env->atn_count_rotate = 0;
	env->atn_count_hold = 0;
	memset(env->tetromino_counts, 0, sizeof(env->tetromino_counts));

	restore_grid(env);
	// This acts as a learning curriculum, exposing agents to garbage lines later
	add_garbage_lines(env, env->n_init_garbage, 9);

	// Noise obs effectively jitters the action.
	// The agents will eventually learn to ignore these.
	for (int i = 0; i < env->n_noise_obs; i++) {
		env->observations[234 + i] = 1;
	}

	initialize_deck(env);
	spawn_new_tetromino(env);
	compute_observations(env);
}

void place_tetromino(Tetris *env) {
	int row_to_check = env->cur_tetromino_row + TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot] - 1;
	int lines_deleted = 0;
	env->can_swap = 1;

	for (int c = 0; c < TETROMINOES_FILLS_COL[env->cur_tetromino][env->cur_tetromino_rot];
	     c++) { // Fill the main grid with the tetromino
		for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot]; r++) {
			if (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1) {
				env->grid[(r + env->cur_tetromino_row) * env->n_cols + c + env->cur_tetromino_col] =
				    env->cur_tetromino + 1;
			}
		}
	}
	for (int r = 0; r < TETROMINOES_FILLS_ROW[env->cur_tetromino][env->cur_tetromino_rot];
	     r++) { // Proceed to delete the complete rows
		if (is_full_row(env, row_to_check)) {
			clear_row(env, row_to_check);
			lines_deleted += 1;
		} else {
			row_to_check -= 1;
		}
	}
	if (lines_deleted > 0) {
		env->count_combos += 1;
		env->lines_deleted += lines_deleted;
		env->score += SCORE_COMBO[lines_deleted];
		env->rewards[0] += REWARD_COMBO[lines_deleted];
		env->ep_return += REWARD_COMBO[lines_deleted];

		// These determine the game difficulty. Consider making them args.
		env->game_level = 1 + env->lines_deleted / LINES_PER_LEVEL;
		env->ticks_per_fall = max(3, INITIAL_TICKS_PER_FALL - env->game_level / 4);
		env->ticks_per_garbage = max(40, (int)(INITIAL_TICKS_PER_GARBAGE - 7 * sqrt((double)env->game_level)));
	}

	if (can_spawn_new_tetromino(env)) {
		spawn_new_tetromino(env);
	} else {
		env->terminals[0] = 1; // Game over
	}
}

void c_step(Tetris *env) {
	env->terminals[0] = 0;
	env->rewards[0] = 0.0;
	env->tick += 1;
	env->tick_fall += 1;
	env->tick_garbage += 1;
	int action = env->actions[0];

	if (action == ACTION_LEFT) {
		if (can_go_left(env)) {
			env->cur_tetromino_col -= 1;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
		}
	}
	if (action == ACTION_RIGHT) {
		if (can_go_right(env)) {
			env->cur_tetromino_col += 1;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
		}
	}
	if (action == ACTION_ROTATE) {
		env->atn_count_rotate += 1;
		if (can_rotate(env)) {
			env->cur_tetromino_rot = (env->cur_tetromino_rot + 1) % NUM_ROTATIONS;
			env->rewards[0] += REWARD_ROTATE;
			env->ep_return += REWARD_ROTATE;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
		}
	}
	if (action == ACTION_SOFT_DROP) {
		env->atn_count_soft_drop += 1;
		if (can_soft_drop(env)) {
			env->cur_tetromino_row += 1;
			env->score += SCORE_SOFT_DROP;
			// env->rewards[0] += REWARD_SOFT_DROP;
			// env->ep_return += REWARD_SOFT_DROP;
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
		}
	}
	if (action == ACTION_HOLD) {
		env->atn_count_hold += 1;
		if (can_hold(env)) {
			int t1 = env->cur_tetromino;
			int t2 = env->hold_tetromino;
			if (t2 == -1) {
				spawn_new_tetromino(env);
				env->hold_tetromino = t1;
				env->can_swap = 0;
			} else {
				env->cur_tetromino = t2;
				env->tetromino_deck[env->cur_position_in_deck] = t2;
				env->hold_tetromino = t1;
				env->can_swap = 0;
				env->cur_tetromino_rot = 0;
				env->cur_tetromino_col = env->n_cols / 2;
				env->cur_tetromino_row = 0;
				env->tick_fall = 0;
			}
		} else {
			env->rewards[0] += REWARD_INVALID_ACTION;
			env->ep_return += REWARD_INVALID_ACTION;
		}
	}
	if (action == ACTION_HARD_DROP) {
		env->atn_count_hard_drop += 1;
		while (can_soft_drop(env)) {
			env->cur_tetromino_row += 1;
			// NOTE: this seems to be a super effective reward trick
			env->rewards[0] += REWARD_HARD_DROP;
			env->ep_return += REWARD_HARD_DROP;
		}
		env->score += SCORE_HARD_DROP;
		place_tetromino(env);
	}
	if (env->tick_fall >= env->ticks_per_fall) {
		env->tick_fall = 0;
		if (can_soft_drop(env)) {
			env->cur_tetromino_row += 1;
		} else {
			place_tetromino(env);
		}
	}

	if (env->tick >= GARBAGE_KICKOFF_TICK && env->tick_garbage >= env->ticks_per_garbage) {
		env->tick_garbage = 0;
		int num_holes = min(5, max(1, env->game_level / 8));
		add_garbage_lines(env, 1, num_holes);
	}

	if (env->terminals[0] == 1 || (env->tick >= MAX_TICKS)) {
		// TraceLog(LOG_INFO, "Game reset. Score: %d", env->score);
		// TraceLog(LOG_INFO, "I:%d O:%d T:%d S:%d Z:%d J:%d L:%d",
		// 	env->tetromino_counts[1], env->tetromino_counts[0], env->tetromino_counts[4],
		// 	env->tetromino_counts[2], env->tetromino_counts[3], env->tetromino_counts[6], env->tetromino_counts[5]);

		add_log(env);
		c_reset(env);
	}

	compute_observations(env);
}

Client *make_client(Tetris *env) {
	Client *client = (Client *)calloc(1, sizeof(Client));
	client->ui_rows = 1;
	client->deck_rows = SIZE;
	client->total_rows = 1 + client->ui_rows + 1 + client->deck_rows + 1 + env->n_rows + 1;
	client->total_cols = max(1 + env->n_cols + 1, 1 + 3 * NUM_PREVIEW);
	client->preview_target_col = env->n_cols / 2;
	client->preview_target_rotation = 0;
	InitWindow(SQUARE_SIZE * client->total_cols, SQUARE_SIZE * client->total_rows, "PufferLib Tetris");
	SetTargetFPS(30);
	return client;
}

void close_client(Client *client) {
	CloseWindow();
	free(client);
}

Color BORDER_COLOR = (Color){100, 100, 100, 255};
Color DASH_COLOR = (Color){80, 80, 80, 255};
Color DASH_COLOR_BRIGHT = (Color){150, 150, 150, 255};
Color DASH_COLOR_DARK = (Color){50, 50, 50, 255};
// Color GARBAGE_COLOR = (Color){140, 140, 140, 255};

void c_render(Tetris *env) {
	if (env->client == NULL) {
		env->client = make_client(env);
	}
	Client *client = env->client;

	if (IsKeyDown(KEY_ESCAPE)) {
		exit(0);
	}
	if (IsKeyPressed(KEY_TAB)) {
		ToggleFullscreen();
	}

	BeginDrawing();
	ClearBackground(BLACK);
	int x, y;
	Color color;

	// outer grid
	for (int r = 0; r < client->total_rows; r++) {
		for (int c = 0; c < client->total_cols; c++) {
			x = c * SQUARE_SIZE;
			y = r * SQUARE_SIZE;
			if ((c == 0) || (c == client->total_cols - 1) ||
			    ((r >= 1 + client->ui_rows + 1) && (r < 1 + client->ui_rows + 1 + client->deck_rows)) ||
			    ((r >= 1 + client->ui_rows + 1 + client->deck_rows + 1) && (c >= env->n_rows)) || (r == 0) ||
			    (r == 1 + client->ui_rows) || (r == 1 + client->ui_rows + 1 + client->deck_rows) ||
			    (r == client->total_rows - 1)) {
				DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
				              SQUARE_SIZE - 2 * HALF_LINEWIDTH, BORDER_COLOR);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR_DARK);
				DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR_DARK);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_DARK);
				DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR_DARK);
			}
		}
	}
	// main grid
	for (int r = 0; r < env->n_rows; r++) {
		for (int c = 0; c < env->n_cols; c++) {
			x = (c + 1) * SQUARE_SIZE;
			y = (1 + client->ui_rows + 1 + client->deck_rows + 1 + r) * SQUARE_SIZE;
			int block_id = env->grid[r * env->n_cols + c];
			if (block_id == 0) {
				color = BLACK;
			} else if (block_id < 0) {
				color = TETROMINOES_COLORS[-block_id - 1];
			} else {
				color = TETROMINOES_COLORS[block_id - 1];
			}
			DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
			              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR);
			DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
			              DASH_COLOR);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR);
			DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
			              DASH_COLOR);
		}
	}

	// current tetromino
	for (int r = 0; r < SIZE; r++) {
		for (int c = 0; c < SIZE; c++) {
			x = (c + env->cur_tetromino_col + 1) * SQUARE_SIZE;
			y = (1 + client->ui_rows + 1 + client->deck_rows + 1 + r + env->cur_tetromino_row) * SQUARE_SIZE;

			if (TETROMINOES[env->cur_tetromino][env->cur_tetromino_rot][r][c] == 1) {
				color = TETROMINOES_COLORS[env->cur_tetromino];
				DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
				              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR);
				DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR);
				DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR);
			}
		}
	}

	// Deck grid
	int tetromino_id;
	for (int i = 0; i < NUM_PREVIEW; i++) {
		int deck_idx = (env->cur_position_in_deck + 1 + i) % DECK_SIZE;
		tetromino_id = env->tetromino_deck[deck_idx];
		for (int r = 0; r < SIZE; r++) {
			for (int c = 0; c < 2; c++) {
				x = (c + 1 + 3 * i) * SQUARE_SIZE;
				y = (1 + client->ui_rows + 1 + r) * SQUARE_SIZE;
				int r_offset = (SIZE - TETROMINOES_FILLS_ROW[tetromino_id][0]);
				if (r < r_offset) {
					color = BLACK;
				} else {
					color =
					    (TETROMINOES[tetromino_id][0][r - r_offset][c] == 0) ? BLACK : TETROMINOES_COLORS[tetromino_id];
				}
				DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
				              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR_BRIGHT);
				DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
				              DASH_COLOR_BRIGHT);
				DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR_BRIGHT);
				DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
				              DASH_COLOR_BRIGHT);
			}
		}
	}

	// hold tetromino
	for (int r = 0; r < SIZE; r++) {
		for (int c = 0; c < 2; c++) {
			x = (client->total_cols - 3 + c) * SQUARE_SIZE;
			y = (1 + client->ui_rows + 1 + r) * SQUARE_SIZE;
			if (env->hold_tetromino > -1) {
				int r_offset = (SIZE - TETROMINOES_FILLS_ROW[env->hold_tetromino][0]);
				if (r < r_offset) {
					color = BLACK;
				} else {
					color = (env->hold_tetromino > -1) && (TETROMINOES[env->hold_tetromino][0][r - r_offset][c] == 0)
					            ? BLACK
					            : TETROMINOES_COLORS[env->hold_tetromino];
				}
			} else {
				color = BLACK;
			}
			DrawRectangle(x + HALF_LINEWIDTH, y + HALF_LINEWIDTH, SQUARE_SIZE - 2 * HALF_LINEWIDTH,
			              SQUARE_SIZE - 2 * HALF_LINEWIDTH, color);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH, DASH_COLOR_BRIGHT);
			DrawRectangle(x - HALF_LINEWIDTH, y + SQUARE_SIZE - HALF_LINEWIDTH, SQUARE_SIZE, 2 * HALF_LINEWIDTH,
			              DASH_COLOR_BRIGHT);
			DrawRectangle(x - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE, DASH_COLOR_BRIGHT);
			DrawRectangle(x + SQUARE_SIZE - HALF_LINEWIDTH, y - HALF_LINEWIDTH, 2 * HALF_LINEWIDTH, SQUARE_SIZE,
			              DASH_COLOR_BRIGHT);
		}
	}
	// Draw UI
	DrawText(TextFormat("Score: %i", env->score), SQUARE_SIZE + 4, SQUARE_SIZE + 4, 28, (Color){255, 160, 160, 255});
	DrawText(TextFormat("Lvl: %i", env->game_level), (client->total_cols - 4) * SQUARE_SIZE, SQUARE_SIZE + 4, 28, (Color){160, 255, 160, 255});
	EndDrawing();
}
