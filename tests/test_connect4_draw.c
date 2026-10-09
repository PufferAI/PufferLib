// Run from the repository root:
// cc -O2 -Isrc -Iraylib-5.5_linux_amd64/include tests/test_connect4_draw.c \
//   raylib-5.5_linux_amd64/lib/libraylib.a -lm -lGL -lpthread -ldl -lrt -lX11 \
//   -o /tmp/test_connect4_draw && /tmp/test_connect4_draw
#include <assert.h>
#include "../ocean/connect4/connect4.h"

static void attach(Connect4* env, float* obs, float* action,
        float* reward, float* terminal) {
    env->agents[0].observations = obs;
    env->agents[0].actions = action;
    env->agents[0].rewards = reward;
    env->agents[0].terminals = terminal;
    puf_reset(env);
}

static void test_opening(void) {
    for (int column = 5; column <= 6; column++) {
        for (unsigned int seed = 0; seed < 128; seed++) {
            Connect4 env = {0};
            float obs[42], action = column, reward = 0, terminal = 0;
            attach(&env, obs, &action, &reward, &terminal);
            env.rng = seed;
            puf_step(&env);
            assert(terminal == 0 && "two pieces cannot end a game");
            assert(reward == 0);
            assert(env.log.n == 0);
            int pieces = 0;
            for (int i = 0; i < 42; i++) {
                pieces += obs[i] != 0;
            }
            assert(pieces == 2);
        }
    }
}

static void test_full_board(void) {
    // Legal, alternating play to a draw. Rows are filled from the bottom.
    // The last two moves (player 2, opponent 4) fill the only empty cells.
    const int moves[42] = {
        1, 1, 0, 4, 2, 3, 0, 4, 3, 0, 6, 2, 0, 3,
        1, 0, 3, 2, 5, 0, 4, 2, 1, 5, 5, 5, 1, 5,
        4, 5, 6, 3, 3, 1, 4, 6, 2, 6, 6, 6, 2, 4,
    };
    Connect4 env = {0};
    float obs[42], action = moves[40], reward = 0, terminal = 0;
    attach(&env, obs, &action, &reward, &terminal);
    int heights[7] = {0};
    uint64_t pieces[2] = {0};
    for (int i = 0; i < 42; i++) {
        int c = moves[i];
        assert(heights[c] < 6);
        pieces[i % 2] |= UINT64_C(1) << (7 * c + heights[c]++);
        assert(!won(pieces[i % 2]));
        assert(draw(pieces[0] | pieces[1]) == (i == 41));
        if (i == 39) {
            env.player_pieces = pieces[0];
            env.env_pieces = pieces[1];
        }
    }
    env.tick = 20;
    puf_step(&env);
    assert(terminal == 1);
    assert(reward == 0);
    assert(env.log.n == 1 && env.log.episode_length == 21);
    assert(env.log.score == 0 && env.log.perf == 0 && env.log.invalids == 0);
    assert(env.player_pieces == 0 && env.env_pieces == 0);
    for (int i = 0; i < 42; i++) {
        assert(obs[i] == 0);
    }
}

int main(void) {
    test_opening();
    test_full_board();
    uint64_t full_board = 0;
    for (int column = 0; column < 7; column++) {
        for (int row = 0; row < 6; row++) {
            full_board |= UINT64_C(1) << (7 * column + row);
        }
    }
    assert(draw(full_board));
    // Leaving any top cell empty is a legal, non-full occupancy mask.
    for (int column = 0; column < 7; column++) {
        assert(!draw(full_board ^ (UINT64_C(1) << (7 * column + 5))));
    }
    puts("PASS: 256 openings and a complete legal draw with same-step reset");
    return 0;
}
