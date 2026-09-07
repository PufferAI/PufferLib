#include "fc_click_feedback.h"

#include "fc_pathfinding.h"

#include <string.h>

void fc_click_feedback_reset(FcClickFeedback* feedback) {
    if (!feedback) return;
    memset(feedback, 0, sizeof(*feedback));
    feedback->destination_x = -1;
    feedback->destination_y = -1;
}

static void start_cross(FcClickFeedback* feedback, FcClickCrossKind kind,
                        float screen_x, float screen_y) {
    feedback->cross_kind = kind;
    feedback->cross_screen_x = screen_x;
    feedback->cross_screen_y = screen_y;
    feedback->cross_elapsed = 0.0f;
}

void fc_click_feedback_select_move(FcClickFeedback* feedback,
                                   const FcState* state,
                                   int tile_x, int tile_y,
                                   float screen_x, float screen_y) {
    if (!feedback || !state ||
        tile_x < 0 || tile_x >= FC_ARENA_WIDTH ||
        tile_y < 0 || tile_y >= FC_ARENA_HEIGHT) {
        return;
    }

    feedback->destination_active = 1;
    feedback->destination_x = tile_x;
    feedback->destination_y = tile_y;
    feedback->preview_pending = 1;
    feedback->preview_route_len = fc_pathfind_bfs_move_near(
        state->player.x, state->player.y, tile_x, tile_y,
        state->walkable, state->movement_flags,
        feedback->preview_route_x, feedback->preview_route_y, FC_MAX_ROUTE);
    start_cross(feedback, FC_CLICK_CROSS_MOVE, screen_x, screen_y);
}

void fc_click_feedback_select_interaction(FcClickFeedback* feedback,
                                          float screen_x, float screen_y) {
    if (!feedback) return;
    feedback->destination_active = 0;
    feedback->destination_x = -1;
    feedback->destination_y = -1;
    feedback->preview_pending = 0;
    feedback->preview_route_len = 0;
    start_cross(feedback, FC_CLICK_CROSS_INTERACTION, screen_x, screen_y);
}

void fc_click_feedback_accept_move_tick(FcClickFeedback* feedback,
                                        const FcState* state) {
    if (!feedback || !state || !feedback->preview_pending) return;
    feedback->preview_pending = 0;
    feedback->preview_route_len = 0;
    fc_click_feedback_sync(feedback, state);
}

void fc_click_feedback_sync(FcClickFeedback* feedback,
                            const FcState* state) {
    if (!feedback || !state || !feedback->destination_active ||
        feedback->preview_pending) {
        return;
    }
    if (state->player.route_idx >= state->player.route_len) {
        feedback->destination_active = 0;
        feedback->destination_x = -1;
        feedback->destination_y = -1;
    }
}

void fc_click_feedback_update(FcClickFeedback* feedback,
                              float elapsed_seconds) {
    if (!feedback || feedback->cross_kind == FC_CLICK_CROSS_NONE ||
        elapsed_seconds <= 0.0f) {
        return;
    }
    feedback->cross_elapsed += elapsed_seconds;
    if (feedback->cross_elapsed >=
        FC_CLICK_CROSS_FRAME_COUNT * FC_CLICK_CROSS_FRAME_SECONDS) {
        feedback->cross_kind = FC_CLICK_CROSS_NONE;
        feedback->cross_elapsed = 0.0f;
    }
}

int fc_click_feedback_cross_frame(const FcClickFeedback* feedback) {
    if (!feedback || feedback->cross_kind == FC_CLICK_CROSS_NONE) return -1;
    int frame = (int)(feedback->cross_elapsed / FC_CLICK_CROSS_FRAME_SECONDS);
    if (frame < 0) frame = 0;
    if (frame >= FC_CLICK_CROSS_FRAME_COUNT)
        frame = FC_CLICK_CROSS_FRAME_COUNT - 1;
    return frame;
}

int fc_click_feedback_route(const FcClickFeedback* feedback,
                            const FcState* state,
                            const int** out_x, const int** out_y,
                            int* out_start, int* out_len) {
    if (!feedback || !state || !out_x || !out_y || !out_start || !out_len ||
        !feedback->destination_active) {
        return 0;
    }

    if (feedback->preview_pending) {
        *out_x = feedback->preview_route_x;
        *out_y = feedback->preview_route_y;
        *out_start = 0;
        *out_len = feedback->preview_route_len;
        return feedback->preview_route_len > 0;
    }

    *out_x = state->player.route_x;
    *out_y = state->player.route_y;
    *out_start = state->player.route_idx;
    *out_len = state->player.route_len;
    return state->player.route_idx < state->player.route_len;
}
