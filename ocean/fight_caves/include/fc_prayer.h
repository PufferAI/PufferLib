#ifndef FC_PRAYER_H
#define FC_PRAYER_H

#include "fc_types.h"

typedef struct {
    int prior_prayer;
    int requested_final_prayer;
    int actual_final_prayer;
    int off_requested;
    int off_performed;
    int on_requested;
    int on_succeeded;
    int explicit_off_then_on;
    int final_state_changed;
} FcPrayerTransition;

/* Drain prayer points based on active prayer and bonus.
 * prayer_active_at_tick_start should reflect the prayer state before the
 * current tick's input actions were applied, so 1-tick flicks do not drain. */
int fc_prayer_drain_tick(FcPlayer* p, int prayer_at_tick_start,
                         const FcPrayerTransition* transition);

/* Apply a prayer action (from FC_PRAYER_* constants in fc_contracts.h) */
FcPrayerTransition fc_prayer_apply_action(FcPlayer* p, int prayer_action);

/* Apply capped Prayer loss in tenths. Every Prayer-loss source uses this
 * boundary so depletion always deactivates the overhead and clears fraction. */
int fc_prayer_apply_loss_tenths(FcPlayer* p, int requested_loss_tenths);

/* Prayer potion restore amount in tenths (level-dependent) */
int fc_prayer_potion_restore(int prayer_level);

#endif /* FC_PRAYER_H */
