#ifndef FC_CLICK_FEEDBACK_H
#define FC_CLICK_FEEDBACK_H

#include "fc_types.h"

#define FC_CLICK_CROSS_FRAME_COUNT 4
#define FC_CLICK_CROSS_FRAME_SECONDS 0.10f

typedef enum {
    FC_CLICK_CROSS_NONE = 0,
    FC_CLICK_CROSS_MOVE,
    FC_CLICK_CROSS_INTERACTION,
} FcClickCrossKind;

typedef struct {
    int destination_active;
    int destination_x;
    int destination_y;

    int preview_pending;
    int preview_route_x[FC_MAX_ROUTE];
    int preview_route_y[FC_MAX_ROUTE];
    int preview_route_len;

    FcClickCrossKind cross_kind;
    float cross_screen_x;
    float cross_screen_y;
    float cross_elapsed;
} FcClickFeedback;

void fc_click_feedback_reset(FcClickFeedback* feedback);

/* Build a read-only preview with the same move-near pathfinder used by
 * fc_step(). The simulation state itself is never modified. */
void fc_click_feedback_select_move(FcClickFeedback* feedback,
                                   const FcState* state,
                                   int tile_x, int tile_y,
                                   float screen_x, float screen_y);

void fc_click_feedback_select_interaction(FcClickFeedback* feedback,
                                          float screen_x, float screen_y);

/* Hand the preview over to the route produced by the authoritative tick. */
void fc_click_feedback_accept_move_tick(FcClickFeedback* feedback,
                                        const FcState* state);

/* Clear a completed or cancelled authoritative destination. */
void fc_click_feedback_sync(FcClickFeedback* feedback,
                            const FcState* state);

void fc_click_feedback_update(FcClickFeedback* feedback,
                              float elapsed_seconds);

int fc_click_feedback_cross_frame(const FcClickFeedback* feedback);

/* Return the immediate preview while the click is buffered, then the live
 * core route after the next simulation tick accepts it. */
int fc_click_feedback_route(const FcClickFeedback* feedback,
                            const FcState* state,
                            const int** out_x, const int** out_y,
                            int* out_start, int* out_len);

#endif /* FC_CLICK_FEEDBACK_H */
