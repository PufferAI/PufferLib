#pragma once
#include <random>
#include <utility>

#define RACING_MAX_GHOSTS 8
struct RacingGhostSnapshot {
    RacingCarFrame frame;
    RacingEpisodeEnd end;
    int checkpoints;
#ifdef RACING_MULTI
    int rank;
#endif
};
// Cosmetic names from contributor history and user selections.
static const char *racing_ghost_names[RACING_MAX_GHOSTS] = {
    "Fbr", "Joseph Suarez", "Spencer Cheng", "l1onh3art88",
    "kywch", "BET", "Kinvert", "Valtteri Valo"};
static Color racing_ghost_colors[RACING_MAX_GHOSTS] = {
    {255, 100, 100, 255}, {90, 190, 255, 255}, {100, 240, 150, 255}, {255, 210, 80, 255},
    {210, 130, 255, 255}, {255, 150, 70, 255}, {90, 240, 225, 255}, {240, 140, 200, 255}};
static struct {
    int count, selected;
    char paths[RACING_MAX_GHOSTS][4096];
    RacingCarFrame frames[RACING_MAX_GHOSTS];
    RacingEpisodeEnd ends[RACING_MAX_GHOSTS];
    int checkpoints[RACING_MAX_GHOSTS], best_checkpoints[RACING_MAX_GHOSTS];
    float best_progress[RACING_MAX_GHOSTS];
    double best_lap[RACING_MAX_GHOSTS];
#ifdef RACING_MULTI
    int rank[RACING_MAX_GHOSTS];
#endif
} racing_ghosts;

static void racing_shuffle_names(int count) {
    std::mt19937 rng(std::random_device{}());
    for (int i = count - 1; i > 0; i--) {
        int j = std::uniform_int_distribution<int>(0, i)(rng);
        std::swap(racing_ghost_names[i], racing_ghost_names[j]);
        std::swap(racing_ghost_colors[i], racing_ghost_colors[j]);
    }
}

// Camera selection only accepts live cars, including in manual mode.
static bool racing_ghost_active(int i) {
    return i >= 0 && i < racing_ghosts.count
        && !racing_ghosts.frames[i].task_done && !racing_ghosts.frames[i].crashed;
}
static int racing_live_leader() {
    int best = -1;
    for (int i=0;i<racing_ghosts.count;++i) {
        if (!racing_ghost_active(i)) continue;
#ifdef RACING_MULTI
        if (best < 0 || racing_ghosts.rank[i] < racing_ghosts.rank[best]) best=i;
#else
        if (best < 0 || racing_ghosts.frames[i].progress > racing_ghosts.frames[best].progress) best=i;
#endif
    }
    return best;
}
static int racing_next_live(int selected, int direction) {
    int count=racing_ghosts.count;
    if (!count) return -1;
    for (int step=1;step<=count;++step) {
        int i=((selected < 0 ? (direction > 0 ? -1 : 0) : selected) + direction*step + count*2)%count;
        if (racing_ghost_active(i)) return i;
    }
    return -1;
}
