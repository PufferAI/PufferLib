#pragma once

#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EMPTY 0
#define AGENT 1
#define OPPONENT 3
#define AGENT_PAWN 1
#define AGENT_KING 2
#define OPPONENT_PAWN 3
#define OPPONENT_KING 4

// Required struct. Only use floats!
typedef struct {
  float perf;
  float score;
  float episode_return;
  float episode_length;
  float winrate;
  float n;
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
  Log log;
  unsigned char *observations;
  int *actions;
  float *rewards;
  unsigned char *terminals;
  int size;
  int tick;
  int current_player;
  int agent_pieces;
  int opponent_pieces;
  int capture_available_cache;
  int capture_available_valid;
  int game_over_cache;
  int game_over_valid;
} Checkers;

typedef struct {
  int r;
  int c;
} Position;

typedef struct {
  Position from;
  Position to;
} Move;

float clamp(float val, float low, float high) {
  return fmin(fmax(val, low), high);
}

Move decode_action(Checkers *env, int action) {
  int num_move_types = 8;
  int pos = action / num_move_types;
  int move_type = action % num_move_types;

  Move m;
  m.from.r = pos / env->size;
  m.from.c = pos % env->size;
  m.to.r = m.from.r;
  m.to.c = m.from.c;

  switch (move_type) {
  case 0:
    m.to.r = m.from.r - 1;
    m.to.c = m.from.c - 1;
    break;
  case 1:
    m.to.r = m.from.r - 1;
    m.to.c = m.from.c + 1;
    break;
  case 2:
    m.to.r = m.from.r + 1;
    m.to.c = m.from.c - 1;
    break;
  case 3:
    m.to.r = m.from.r + 1;
    m.to.c = m.from.c + 1;
    break;
  case 4:
    m.to.r = m.from.r - 2;
    m.to.c = m.from.c - 2;
    break;
  case 5:
    m.to.r = m.from.r - 2;
    m.to.c = m.from.c + 2;
    break;
  case 6:
    m.to.r = m.from.r + 2;
    m.to.c = m.from.c - 2;
    break;
  case 7:
    m.to.r = m.from.r + 2;
    m.to.c = m.from.c + 2;
    break;
  }

  return m;
}

int p2i(Checkers *env, Position p) { return p.r * env->size + p.c; }

int check_in_bounds(Checkers *env, Position p) {
  return 0 <= p.r && p.r < env->size && 0 <= p.c && p.c < env->size;
}

int get_piece(Checkers *env, Position p) {
  if (!check_in_bounds(env, p)) {
    return EMPTY;
  }
  return env->observations[p2i(env, p)];
}

int get_piece_type(Checkers *env, Position p) {
  int piece = get_piece(env, p);
  if (piece == AGENT_PAWN || piece == AGENT_KING)
    return AGENT;
  if (piece == OPPONENT_PAWN || piece == OPPONENT_KING)
    return OPPONENT;
  return EMPTY;
}

int get_move_direction(Checkers *env, Move m) {
  return m.to.r > m.from.r ? 1 : -1;
}

int valid_move_direction(Checkers *env, Move m) {
  int piece = get_piece(env, m.from);
  if (piece == AGENT_PAWN)
    return get_move_direction(env, m) == 1 ? 1 : 0;
  if (piece == OPPONENT_PAWN)
    return get_move_direction(env, m) == -1 ? 1 : 0;
  return 1;
}

int is_diagonal_move(Move m) {
  int dr = m.to.r - m.from.r;
  int dc = m.to.c - m.from.c;
  return (dr == dc) || (dr == -dc);
}
int move_size(Move m) { return abs(m.from.r - m.to.r); }

int is_valid_move_no_capture(Checkers *env, Move m) {
  if (!check_in_bounds(env, m.from) || !check_in_bounds(env, m.to))
    return 0;

  if (get_piece_type(env, m.from) != env->current_player)
    return 0;

  if (get_piece(env, m.to) != EMPTY)
    return 0;

  if (!valid_move_direction(env, m))
    return 0;

  if (!is_diagonal_move(m))
    return 0;

  if (move_size(m) != 1 && move_size(m) != 2)
    return 0;

  if (move_size(m) == 2) {
    int other_player = env->current_player == AGENT ? OPPONENT : AGENT;
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    if (get_piece_type(env, between_pos) != other_player)
      return 0;
  }

  return 1;
}

int capture_available(Checkers *env) {
  if (env->capture_available_valid) {
    return env->capture_available_cache;
  }

  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[4][2] = {{-2, -2}, {-2, 2}, {2, -2}, {2, 2}};
    for (int d = 0; d < 4; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;

      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int mid_r = r + directions[d][0] / 2;
      int mid_c = c + directions[d][1] / 2;
      int mid_piece = env->observations[mid_r * env->size + mid_c];

      int opponent_pawn =
          env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
      int opponent_king =
          env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

      if (mid_piece == opponent_pawn || mid_piece == opponent_king) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;

        env->capture_available_cache = 1;
        env->capture_available_valid = 1;
        return 1;
      }
    }
  }

  env->capture_available_cache = 0;
  env->capture_available_valid = 1;
  return 0;
}

int is_valid_move(Checkers *env, Move m) {
  if (capture_available(env) && move_size(m) != 2)
    return 0;
  return is_valid_move_no_capture(env, m);
}

int num_legal_moves(Checkers *env) {
  int res = 0;
  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;
  int has_captures = capture_available(env);

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[8][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                            {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

    for (int d = 0; d < 8; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;

      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int move_size = abs(directions[d][0]);

      if (has_captures && move_size != 2)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      if (move_size == 2) {
        int mid_r = r + directions[d][0] / 2;
        int mid_c = c + directions[d][1] / 2;
        int mid_piece = env->observations[mid_r * env->size + mid_c];

        int opponent_pawn =
            env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
        int opponent_king =
            env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

        if (mid_piece != opponent_pawn && mid_piece != opponent_king)
          continue;
      }

      res++;
    }
  }

  return res;
}

int num_pieces_by_player(Checkers *env, int player) {
  if (player == AGENT) {
    return env->agent_pieces;
  } else {
    return env->opponent_pieces;
  }
}

int try_make_king(Checkers *env) {
  int promoted = 0;

  for (int i = 0; i < env->size; i++) {
    if (env->observations[i] == OPPONENT_PAWN) {
      env->observations[i] = OPPONENT_KING;
      promoted = 1;
    }
  }
  for (int i = 0; i < env->size; i++) {
    if (env->observations[env->size * (env->size - 1) + i] == AGENT_PAWN) {
      env->observations[env->size * (env->size - 1) + i] = AGENT_KING;
      promoted = 1;
    }
  }

  if (promoted) {
    env->capture_available_valid = 0;
    env->game_over_valid = 0;
  }

  return promoted;
}

int is_game_over(Checkers *env) {
  if (env->game_over_valid) {
    return env->game_over_cache;
  }

  int current_player_pieces = num_pieces_by_player(env, env->current_player);
  int other_player = env->current_player == AGENT ? OPPONENT : AGENT;
  int other_player_pieces = num_pieces_by_player(env, other_player);

  if (current_player_pieces == 0 || other_player_pieces == 0) {
    env->game_over_cache = 1;
    env->game_over_valid = 1;
    return 1;
  }

  int has_captures = capture_available(env);
  if (has_captures) {
    env->game_over_cache = 0;
    env->game_over_valid = 1;
    return 0;
  }

  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    for (int d = 0; d < 4; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;
      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      env->game_over_cache = 0;
      env->game_over_valid = 1;
      return 0;
    }
  }

  env->game_over_cache = 1;
  env->game_over_valid = 1;
  return 1;
}

int get_winner(Checkers *env) {
  if (env->agent_pieces == 0) {
    return OPPONENT;
  }

  if (env->opponent_pieces == 0) {
    return AGENT;
  }

  if (is_game_over(env)) {
    return env->current_player == AGENT ? OPPONENT : AGENT;
  }

  return EMPTY;
}

void make_move(Checkers *env, int action) {
  Move m = decode_action(env, action);
  if (!is_valid_move(env, m)) {
    env->rewards[0] = -1.0f; // reward for invalid move
    return;
  }

  int moving_piece = get_piece(env, m.from);
  env->observations[p2i(env, m.from)] = EMPTY;
  env->observations[p2i(env, m.to)] = moving_piece;

  int capture_occurred = 0;
  float reward = 0.0f;

  if (move_size(m) == 2) {
    Position between_pos =
        (Position){(m.from.r + m.to.r) / 2, (m.from.c + m.to.c) / 2};
    int captured_piece = env->observations[p2i(env, between_pos)];
    env->observations[p2i(env, between_pos)] = EMPTY;
    capture_occurred = 1;

    if (captured_piece == AGENT_PAWN || captured_piece == AGENT_KING) {
      env->agent_pieces--;
      reward -= 0.05f; // reward for losing pieces
    } else if (captured_piece == OPPONENT_PAWN ||
               captured_piece == OPPONENT_KING) {
      env->opponent_pieces--;
    }
  }

  env->capture_available_valid = 0;
  env->game_over_valid = 0;

  int promotion_occurred = try_make_king(env);

  if (capture_occurred && env->current_player == OPPONENT) {
    reward += 0.1f; // reward for capturing
  } else if (env->current_player == AGENT) {
    reward += 0.01f; // reward for successful moves
  }

  if (move_size(m) == 1 || !capture_available(env)) {
    int other_player = env->current_player == AGENT ? OPPONENT : AGENT;
    env->current_player = other_player;
  }

  if (promotion_occurred) {
    for (int i = 0; i < env->size; i++) {
      if (env->observations[env->size * (env->size - 1) + i] == AGENT_KING) {
        reward += 0.05f; // reward for promotion
        break;
      }
    }
  }

  if (is_game_over(env)) {
    env->terminals[0] = 1;
    int winner = get_winner(env);
    reward = winner == AGENT ? 1.0f : -1.0f;
  }

  env->rewards[0] = clamp(reward, -1.0f, 1.0f);
}

void scripted_first_move(Checkers *env) {
  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;
  int has_captures = capture_available(env);

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[8][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                            {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

    for (int d = 0; d < 8; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;
      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int move_size = abs(directions[d][0]);

      if (has_captures && move_size != 2)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      if (move_size == 2) {
        int mid_r = r + directions[d][0] / 2;
        int mid_c = c + directions[d][1] / 2;
        int mid_piece = env->observations[mid_r * env->size + mid_c];

        int opponent_pawn =
            env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
        int opponent_king =
            env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

        if (mid_piece != opponent_pawn && mid_piece != opponent_king)
          continue;
      }

      int action = i * 8 + d;
      make_move(env, action);
      return;
    }
  }
}

// Helper function to evaluate position value
float evaluate_position(Checkers *env) {
  float score = 0.0f;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    int r = i / env->size;

    if (piece == AGENT_PAWN) {
      score += 1.0f + (r * 0.1f); // Pawns are worth more as they advance
    } else if (piece == AGENT_KING) {
      score += 2.0f;
    } else if (piece == OPPONENT_PAWN) {
      score -= 1.0f + ((env->size - 1 - r) * 0.1f);
    } else if (piece == OPPONENT_KING) {
      score -= 2.0f;
    }
  }

  return score;
}

void scripted_random_move(Checkers *env) {
  int current_pawn = env->current_player == AGENT ? AGENT_PAWN : OPPONENT_PAWN;
  int current_king = env->current_player == AGENT ? AGENT_KING : OPPONENT_KING;
  int has_captures = capture_available(env);

  srand(time(NULL));
  int i;
  int num_positions = env->size * env->size;
  while (1) {
    i = random() % num_positions;
    int piece = env->observations[i];
    if (piece != current_pawn && piece != current_king)
      continue;

    int r = i / env->size;
    int c = i % env->size;

    int directions[8][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1},
                            {-2, -2}, {-2, 2}, {2, -2}, {2, 2}};

    for (int d = 0; d < 8; d++) {
      int new_r = r + directions[d][0];
      int new_c = c + directions[d][1];

      if (new_r < 0 || new_r >= env->size || new_c < 0 || new_c >= env->size)
        continue;
      if (env->observations[new_r * env->size + new_c] != EMPTY)
        continue;

      int move_size = abs(directions[d][0]);

      if (has_captures && move_size != 2)
        continue;

      if (piece == current_pawn) {
        int move_dir = directions[d][0] > 0 ? 1 : -1;
        int valid_dir = env->current_player == AGENT ? 1 : -1;
        if (move_dir != valid_dir)
          continue;
      }

      if (move_size == 2) {
        int mid_r = r + directions[d][0] / 2;
        int mid_c = c + directions[d][1] / 2;
        int mid_piece = env->observations[mid_r * env->size + mid_c];

        int opponent_pawn =
            env->current_player == AGENT ? OPPONENT_PAWN : AGENT_PAWN;
        int opponent_king =
            env->current_player == AGENT ? OPPONENT_KING : AGENT_KING;

        if (mid_piece != opponent_pawn && mid_piece != opponent_king)
          continue;
      }

      int action = i * 8 + d;
      make_move(env, action);
      return;
    }
  }
}

void scripted_step(Checkers *env, int difficulty) {
  switch (difficulty) {
  case 0:
    scripted_first_move(env);
    break;
  case 1:
    scripted_random_move(env);
    break;
  default:
    scripted_random_move(env);
    break;
  }
}

void update_piece_counts(Checkers *env) {
  env->agent_pieces = 0;
  env->opponent_pieces = 0;

  for (int i = 0; i < env->size * env->size; i++) {
    int piece = env->observations[i];
    if (piece == AGENT_PAWN || piece == AGENT_KING) {
      env->agent_pieces++;
    } else if (piece == OPPONENT_PAWN || piece == OPPONENT_KING) {
      env->opponent_pieces++;
    }
  }

  env->capture_available_valid = 0;
  env->game_over_valid = 0;
}

void add_log(Checkers *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += evaluate_position(env);
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  if (env->terminals[0] == 1)
    env->log.winrate += get_winner(env) == AGENT ? 1.0f : 0.0f;
  env->log.n += 1;
}

// Required function
void c_reset(Checkers *env) {
  env->tick = 0;
  env->terminals[0] = 0;
  env->rewards[0] = 0.0f;

  int tiles = env->size * env->size;
  for (int i = 0; i < tiles; i++)
    env->observations[i] = EMPTY;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < env->size; j++) {
      if ((i + j) % 2)
        env->observations[i * env->size + j] = AGENT_PAWN;
    }
  }
  for (int i = env->size - 3; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      if ((i + j) % 2)
        env->observations[i * env->size + j] = OPPONENT_PAWN;
    }
  }

  env->current_player = AGENT;

  update_piece_counts(env);
}

// Required function
void c_step(Checkers *env) {
  env->tick += 1;
  int action = env->actions[0];
  env->rewards[0] = 0.0f;
  env->terminals[0] = 0;

  make_move(env, action);

  env->rewards[0] = clamp(env->rewards[0], -1.0f, 1.0f);
  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }

  scripted_step(env, 1);
  if (env->terminals[0] == 1) {
    add_log(env);
    c_reset(env);
    return;
  }
}

// Required function. Should handle creating the client on first call
void c_render(Checkers *env) {
  const Color BG1 = (Color){27, 27, 27, 255};
  const Color BG2 = (Color){13, 13, 13, 255};

  int cell_size = 64;
  int window_width = cell_size * env->size;
  int window_height = cell_size * env->size;
  int radius = cell_size / 3;
  int king_offset = 14;

  if (!IsWindowReady()) {
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(window_width, window_height, "Puffer Checkers");
    SetTargetFPS(30);
  } else if (GetScreenWidth() != window_width ||
             GetScreenHeight() != window_height) {
    SetWindowSize(window_width, window_height);
  }

  if (IsKeyDown(KEY_ESCAPE)) {
    CloseWindow();
    exit(0);
    return;
  }

  BeginDrawing();
  ClearBackground(BG1);

  Color piece_color;
  for (int i = 0; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      int piece = env->observations[i * env->size + j];
      if ((i + j) % 2 == 0)
        DrawRectangle(j * cell_size - 1, i * cell_size - 1, cell_size + 1,
                      cell_size + 1, BG2);
      if (piece == EMPTY)
        continue;

      int center_x = j * cell_size + cell_size / 2;
      int center_y = i * cell_size + cell_size / 2;

      switch (piece) {
      case AGENT_PAWN:
        piece_color = BLUE;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x - radius / 3, center_y - radius / 3,
                           radius / 3, (Color){255, 255, 255, 80},
                           (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      case AGENT_KING:
        piece_color = BLUE;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});

        DrawCircleGradient(center_x, center_y - king_offset / 2, radius,
                           (Color){20, 20, 20, 60}, (Color){20, 20, 20, 30});
        DrawCircle(center_x, center_y - king_offset, radius, piece_color);
        DrawCircleGradient(
            center_x - radius / 3, center_y - radius / 3 - king_offset,
            radius / 3, (Color){255, 255, 255, 80}, (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y - king_offset, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      case OPPONENT_PAWN:
        piece_color = RED;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x - radius / 3, center_y - radius / 3,
                           radius / 3, (Color){255, 255, 255, 80},
                           (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      case OPPONENT_KING:
        piece_color = RED;
        DrawCircle(center_x, center_y, radius, piece_color);
        DrawCircleGradient(center_x, center_y, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});

        DrawCircleGradient(center_x, center_y - king_offset / 2, radius,
                           (Color){20, 20, 20, 60}, (Color){20, 20, 20, 30});
        DrawCircle(center_x, center_y - king_offset, radius, piece_color);
        DrawCircleGradient(
            center_x - radius / 3, center_y - radius / 3 - king_offset,
            radius / 3, (Color){255, 255, 255, 80}, (Color){255, 255, 255, 10});
        DrawCircleGradient(center_x, center_y - king_offset, radius,
                           (Color){255, 255, 255, 50},
                           (Color){255, 255, 255, 5});
        break;

      default:
        break;
      }
    }
  }

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(Checkers *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
}
