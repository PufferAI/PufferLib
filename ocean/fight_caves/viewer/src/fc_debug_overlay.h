#ifndef FC_DEBUG_OVERLAY_H
#define FC_DEBUG_OVERLAY_H

#include "fc_reward.h"
#include "fc_types.h"
#include "raylib.h"

#define DBG_COLLISION (1 << 0)
#define DBG_LOS (1 << 1)
#define DBG_PATH (1 << 2)
#define DBG_RANGE (1 << 3)
#define DBG_ALL (DBG_COLLISION | DBG_LOS | DBG_PATH | DBG_RANGE)

void dbg_log_clear(void);
void dbg_log_tick(const FcState *state);
void debug_overlay_3d(const FcState *state, int dbg_flags);
void debug_overlay_screen(const FcState *state, Camera3D cam, int dbg_flags);
void dbg_draw_prayer_window_indicator(Vector3 world_anchor, Camera3D cam);
int dbg_draw_panel_tabs(const FcState *state,
                        const FcRewardBreakdown *reward_breakdown,
                        const FcRewardRuntime *reward_runtime,
                        int reward_config_loaded,
                        const char *reward_config_path,
                        int px, int x, int by, int pw, int dbg_tab,
                        int draw_tabs, int content_height);

#endif
