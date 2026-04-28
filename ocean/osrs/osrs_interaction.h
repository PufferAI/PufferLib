/**
 * @file osrs_interaction.h
 * @brief shared entity interaction system + spec toggle helpers.
 *
 * in real OSRS, clicking to attack an entity starts a persistent interaction
 * that auto-walks + auto-attacks until explicitly interrupted.
 *
 * ref: OSRS reverse engineering docs "Entity Interactions"
 * interrupts: ground click, inventory item actions (food/potion/gear)
 * NOT interrupts: prayer toggle, spec toggle, interface clicks, delays
 */

#ifndef OSRS_INTERACTION_H
#define OSRS_INTERACTION_H


typedef struct {
    int target_slot;    /* target entity slot index, -1 = no interaction */
} OsrsInteraction;


static inline void osrs_interaction_set(OsrsInteraction* ix, int target_slot) {
    ix->target_slot = target_slot;
}

static inline void osrs_interaction_clear(OsrsInteraction* ix) {
    ix->target_slot = -1;
}

static inline int osrs_interaction_active(const OsrsInteraction* ix) {
    return ix->target_slot >= 0;
}

static inline void osrs_interaction_init(OsrsInteraction* ix) {
    ix->target_slot = -1;
}


#define OSRS_IACT_NONE     0  /* no action / idle — does NOT interrupt */
#define OSRS_IACT_MOVE     1  /* explicit ground click — INTERRUPTS */
#define OSRS_IACT_EAT      2  /* eat food from inventory — INTERRUPTS */
#define OSRS_IACT_DRINK    3  /* drink potion from inventory — INTERRUPTS */
#define OSRS_IACT_EQUIP    4  /* equip/switch gear from inventory — INTERRUPTS */
#define OSRS_IACT_PRAYER   5  /* prayer toggle — does NOT interrupt */
#define OSRS_IACT_SPEC     6  /* spec toggle — does NOT interrupt */
#define OSRS_IACT_ATTACK   7  /* click to attack entity — SETS new interaction (not an interrupt) */


/* check if an action type interrupts the current interaction.
   if it does, clears the interaction and returns 1. otherwise returns 0.
   ATTACK is handled separately (it sets a new interaction, not an interrupt). */
static inline int osrs_interaction_check_interrupt(OsrsInteraction* ix, int action_type) {
    switch (action_type) {
        case OSRS_IACT_MOVE:
        case OSRS_IACT_EAT:
        case OSRS_IACT_DRINK:
        case OSRS_IACT_EQUIP:
            osrs_interaction_clear(ix);
            return 1;
        case OSRS_IACT_NONE:
        case OSRS_IACT_PRAYER:
        case OSRS_IACT_SPEC:
        case OSRS_IACT_ATTACK:
        default:
            return 0;
    }
}


/* spec toggle: arm/disarm special attack.
   in real OSRS: clicking the spec orb toggles spec_armed.
   when attack fires with spec_armed=1, weapon spec is used and spec_armed auto-disarms.
   spec toggle does NOT interrupt entity interactions. */
static inline void osrs_spec_toggle(int* spec_armed) {
    *spec_armed = !(*spec_armed);
}

static inline void osrs_spec_disarm(int* spec_armed) {
    *spec_armed = 0;
}

#endif /* OSRS_INTERACTION_H */
