// Round-robin tournament among scripted robocode bots (3,4,5,6).
//
// Compile from repo root:
//   gcc -O2 -Iocean/robocode -Isrc -Iraylib-5.5_linux_amd64/include \
//       -o build/bot_tournament ocean/robocode/bot_tournament.c \
//       -Lraylib-5.5_linux_amd64/lib -lraylib -lm -lpthread -ldl \
//       -Wl,-rpath,$PWD/raylib-5.5_linux_amd64/lib
//
// Usage:
//   ./build/bot_tournament [games_per_pair] [seed]
//
// Policies: 3=wave_surfer 4=hawk_on_fire 5=raiko 6=drussgt

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "robocode.h"

static const char* policy_name(int p) {
    switch (p) {
        case 3: return "wave_surfer";
        case 4: return "hawk_on_fire";
        case 5: return "raiko";
        case 6: return "drussgt";
        default: return "unknown";
    }
}

// Returns 0 if policy_a wins, 1 if policy_b wins, -1 draw.
static int play_one(int policy_a, int policy_b, int max_ticks, unsigned int seed) {
    Robocode env;
    memset(&env, 0, sizeof(env));
    env.num_agents = 0;
    env.num_bots = 2;
    env.width = 800;
    env.height = 600;
    env.max_ticks = max_ticks;
    env.bot_policy = policy_a;
    env.bot_policy_1 = policy_b;
    env.bot_match_winner = -2;
    env.rng = seed;
    env.dr = 0.0f;
    env.client = NULL;

    init(&env);
    puf_reset(&env);

    for (int step = 0; step < max_ticks + 5; step++) {
        env.bot_match_winner = -2;
        puf_step(&env);
        if (env.bot_match_winner != -2) {
            int w = env.bot_match_winner;
            puf_close(&env);
            return w;
        }
    }
    puf_close(&env);
    return -1;
}

int main(int argc, char** argv) {
    int games = (argc > 1) ? atoi(argv[1]) : 100;
    unsigned int seed0 = (argc > 2) ? (unsigned)atoi(argv[2]) : 42u;
    if (games < 2) games = 2;
    if (games % 2) games++;

    int policies[] = {3, 4, 5, 6};
    const int N = 4;
    int wins[4][4];
    int draws[4][4];
    memset(wins, 0, sizeof(wins));
    memset(draws, 0, sizeof(draws));
    int total_wins[4] = {0};
    int total_games[4] = {0};

    printf("Bot tournament: 3,4,5,6 | games/pair=%d (half each side) | seed=%u\n",
           games, seed0);
    printf("Field 800x600 max_ticks=3000\n\n");
    fflush(stdout);

    unsigned int game_i = 0;
    int half = games / 2;
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            int pa = policies[i], pb = policies[j];
            int w_a = 0, w_b = 0, d = 0;
            for (int g = 0; g < half; g++) {
                int r = play_one(pa, pb, 3000, seed0 + 10007u * (++game_i));
                if (r == 0) w_a++;
                else if (r == 1) w_b++;
                else d++;
            }
            for (int g = 0; g < half; g++) {
                int r = play_one(pb, pa, 3000, seed0 + 10007u * (++game_i));
                if (r == 0) w_b++;
                else if (r == 1) w_a++;
                else d++;
            }
            wins[i][j] = w_a;
            wins[j][i] = w_b;
            draws[i][j] = draws[j][i] = d;
            total_wins[i] += w_a;
            total_wins[j] += w_b;
            total_games[i] += games;
            total_games[j] += games;
            printf("%-14s vs %-14s | wins %3d-%3d  draws=%3d  | score_wr=%.3f\n",
                   policy_name(pa), policy_name(pb), w_a, w_b, d,
                   (w_a + 0.5 * d) / (double)games);
            fflush(stdout);
        }
    }

    printf("\n=== Win matrix (row beat col) ===\n");
    printf("%16s", "");
    for (int j = 0; j < N; j++) printf("%14s", policy_name(policies[j]));
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("%16s", policy_name(policies[i]));
        for (int j = 0; j < N; j++) {
            if (i == j) printf("%14s", "—");
            else printf("%14d", wins[i][j]);
        }
        printf("\n");
    }

    double score[4];
    int order[4] = {0, 1, 2, 3};
    for (int i = 0; i < N; i++) {
        score[i] = (double)total_wins[i];
        for (int j = 0; j < N; j++) if (i != j) score[i] += 0.5 * draws[i][j];
    }
    for (int a = 0; a < N; a++) {
        for (int b = a + 1; b < N; b++) {
            if (score[order[b]] > score[order[a]]) {
                int t = order[a]; order[a] = order[b]; order[b] = t;
            }
        }
    }

    printf("\n=== Ranking (score = wins + 0.5*draws) ===\n");
    for (int r = 0; r < N; r++) {
        int i = order[r];
        double wr = score[i] / (double)total_games[i];
        printf("#%d  %-14s  score=%.1f / %d  wr=%.3f  pure_wins=%d\n",
               r + 1, policy_name(policies[i]), score[i], total_games[i], wr,
               total_wins[i]);
    }
    return 0;
}
