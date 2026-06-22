import os
import subprocess
import textwrap


def test_g2048_spawns_tile_after_full_board_merge(tmp_path):
    fake_raylib = tmp_path / 'raylib.h'
    fake_raylib.write_text(textwrap.dedent('''
        typedef struct Color { unsigned char r, g, b, a; } Color;
        #define KEY_ESCAPE 0
        static inline void InitWindow(int width, int height, const char* title) {}
        static inline void SetTargetFPS(int fps) {}
        static inline int IsKeyDown(int key) { return 0; }
        static inline void CloseWindow(void) {}
        static inline void BeginDrawing(void) {}
        static inline void ClearBackground(Color color) {}
        static inline void DrawRectangle(int posX, int posY, int width, int height, Color color) {}
        static inline void DrawText(const char* text, int posX, int posY, int fontSize, Color color) {}
        static inline void EndDrawing(void) {}
        static inline int IsWindowReady(void) { return 0; }
    '''))

    source = tmp_path / 'g2048_spawn_test.c'
    source.write_text(textwrap.dedent('''
        #include <stdio.h>
        #include "g2048.h"

        int main(void) {
            unsigned char observations[SIZE * SIZE] = {0};
            float actions[1] = {2.0f};  // LEFT: c_step adds 1 to action.
            float rewards[1] = {0};
            float terminals[1] = {0};

            Game game = {0};
            game.observations = observations;
            game.actions = actions;
            game.rewards = rewards;
            game.terminals = terminals;
            game.empty_count = 0;
            game.grid_changed = true;
            game.max_episode_ticks = BASE_MAX_TICKS;
            game.rng = 1;

            const unsigned char grid[SIZE][SIZE] = {
                {1, 1, 3, 4},
                {5, 6, 7, 8},
                {5, 9, 10, 11},
                {12, 13, 14, 15},
            };
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    game.grid[i][j] = grid[i][j];
                }
            }

            c_step(&game);

            int observed_empty = 0;
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    observed_empty += game.grid[i][j] == EMPTY;
                }
            }

            if (game.empty_count != 0 || observed_empty != 0) {
                fprintf(stderr, "expected replacement spawn after merge, empty_count=%d observed_empty=%d\\n",
                    game.empty_count, observed_empty);
                return 1;
            }
            if (game.terminals[0] != 0.0f) {
                fprintf(stderr, "test board should remain non-terminal after spawn\\n");
                return 1;
            }
            return 0;
        }
    '''))

    binary = tmp_path / 'g2048_spawn_test'
    repo_root = os.getcwd()
    subprocess.run([
        os.environ.get('CC', 'cc'),
        '-std=gnu11',
        '-I', str(tmp_path),
        '-I', os.path.join(repo_root, 'ocean', 'g2048'),
        str(source),
        '-lm',
        '-o', str(binary),
    ], check=True)
    subprocess.run([str(binary)], check=True)
