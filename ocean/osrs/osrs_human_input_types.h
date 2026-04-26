/**
 * @file osrs_pvp_human_input_types.h
 * @brief HumanInput struct and CursorMode enum — separated from human_input.h
 *        to break circular include dependency (gui.h needs HumanInput, but
 *        human_input.h needs gui.h for prayer/spell grid constants).
 */

#ifndef OSRS_HUMAN_INPUT_TYPES_H
#define OSRS_HUMAN_INPUT_TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    CURSOR_NORMAL = 0,
    CURSOR_SPELL_TARGET,   /* clicked a combat spell, waiting for target click */
} CursorMode;

typedef enum {
    HUMAN_COMMAND_NONE = 0,
    HUMAN_COMMAND_WALK,
    HUMAN_COMMAND_ATTACK_NPC,
    HUMAN_COMMAND_OVERHEAD_PRAYER,
    HUMAN_COMMAND_OFFENSIVE_PRAYER,
    HUMAN_COMMAND_EAT,
    HUMAN_COMMAND_DRINK,
    HUMAN_COMMAND_SPELL_TARGET,
    HUMAN_COMMAND_SPEC_TOGGLE,
    HUMAN_COMMAND_EQUIP_INVENTORY_ITEM,
    HUMAN_COMMAND_FIGHT_STYLE,
} HumanCommandKind;

typedef struct {
    HumanCommandKind kind;
    int world_x, world_y;
    int npc_slot;
    int overhead_prayer;
    int offensive_prayer;
    int food;
    int potion;
    int spell;
    int inventory_slot;
    int item_db_idx;
    int gear_slot;
    int fight_style;
} HumanCommand;

typedef struct {
    HumanCommand* items;
    int count;
    int capacity;
} HumanCommandQueue;

typedef struct HumanInput {
    int enabled;                /* H key toggle: 1 = human controls active */

    HumanCommandQueue commands;

    /* semantic action staging (set by clicks, consumed at tick boundary) */
    int pending_move_x, pending_move_y;   /* world tile coords, -1 = none */
    int pending_attack;                    /* 1 = attack target entity */
    int pending_prayer;                    /* OverheadAction value, -1 = no change */
    int pending_offensive_prayer;          /* 0=none, 1=piety, 2=rigour, 3=augury, -1=no change */
    int pending_food;                      /* 1 = eat food */
    int pending_karambwan;                 /* 1 = eat karambwan */
    int pending_potion;                    /* PotionAction-style intent, 0 = none */
    int pending_veng;                      /* 1 = cast vengeance */
    int pending_spec;                      /* 1 = use special attack */
    int pending_spell;                     /* 0=none, ATTACK_ICE or ATTACK_BLOOD */
    int pending_target_idx;                /* NPC entity index to attack, -1 = none */
    int pending_gear;                      /* gear switch action value, 0 = none */

    CursorMode cursor_mode;
    int selected_spell;                    /* ATTACK_ICE or ATTACK_BLOOD for targeting */
    int selected_spell_gui_idx;            /* GuiSpellIdx of the exact spell cell clicked, for UI highlight. -1 = none */

    /* visual feedback: click cross at screen-space position (like real OSRS client) */
    int click_screen_x, click_screen_y;    /* screen pixel where click occurred */
    int click_cross_timer;                 /* counts up from 0, animation progresses over time */
    int click_cross_active;                /* 1 = cross is visible */
    int click_is_attack;                   /* 1 = red cross (attack), 0 = yellow cross (move) */
} HumanInput;

static inline void human_command_queue_reserve(HumanCommandQueue* q, int min_capacity) {
    if (q->capacity >= min_capacity) return;
    int new_capacity = q->capacity > 0 ? q->capacity : 8;
    while (new_capacity < min_capacity)
        new_capacity *= 2;
    HumanCommand* next = (HumanCommand*)realloc(q->items, (size_t)new_capacity * sizeof(HumanCommand));
    if (!next) {
        fprintf(stderr, "human command queue: out of memory\n");
        abort();
    }
    q->items = next;
    q->capacity = new_capacity;
}

static inline void human_input_queue_command(HumanInput* hi, HumanCommand cmd) {
    human_command_queue_reserve(&hi->commands, hi->commands.count + 1);
    hi->commands.items[hi->commands.count++] = cmd;
}

static inline void human_input_clear_commands(HumanInput* hi) {
    hi->commands.count = 0;
}

static inline void human_input_destroy(HumanInput* hi) {
    free(hi->commands.items);
    hi->commands.items = NULL;
    hi->commands.count = 0;
    hi->commands.capacity = 0;
}

static inline void human_input_init(HumanInput* hi) {
    memset(hi, 0, sizeof(*hi));
    hi->pending_move_x = -1;
    hi->pending_move_y = -1;
    hi->pending_prayer = -1;
    hi->pending_offensive_prayer = -1;
    hi->pending_target_idx = -1;
    hi->click_cross_active = 0;
    human_command_queue_reserve(&hi->commands, 8);
}

static inline void human_input_clear_pending(HumanInput* hi) {
    hi->pending_attack = 0;
    hi->pending_prayer = -1;
    hi->pending_offensive_prayer = -1;
    hi->pending_food = 0;
    hi->pending_karambwan = 0;
    hi->pending_potion = 0;
    hi->pending_veng = 0;
    hi->pending_spec = 0;
    hi->pending_spell = 0;
    hi->pending_target_idx = -1;
    hi->pending_gear = 0;
    human_input_clear_commands(hi);
}

static inline void human_input_clear_move(HumanInput* hi) {
    hi->pending_move_x = -1;
    hi->pending_move_y = -1;
}

static inline void human_input_queue_walk(HumanInput* hi, int world_x, int world_y) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_WALK,
        .world_x = world_x,
        .world_y = world_y,
    });
}

static inline void human_input_queue_attack_npc(HumanInput* hi, int npc_slot) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_ATTACK_NPC,
        .npc_slot = npc_slot,
    });
}

static inline void human_input_queue_overhead_prayer(HumanInput* hi, int overhead_prayer) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_OVERHEAD_PRAYER,
        .overhead_prayer = overhead_prayer,
    });
}

static inline void human_input_queue_offensive_prayer(HumanInput* hi, int offensive_prayer) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_OFFENSIVE_PRAYER,
        .offensive_prayer = offensive_prayer,
    });
}

static inline void human_input_queue_eat(HumanInput* hi, int food) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_EAT,
        .food = food,
    });
}

static inline void human_input_queue_drink(HumanInput* hi, int potion, int inventory_slot) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_DRINK,
        .potion = potion,
        .inventory_slot = inventory_slot,
    });
}

static inline void human_input_queue_spell_target(HumanInput* hi, int spell, int npc_slot) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_SPELL_TARGET,
        .spell = spell,
        .npc_slot = npc_slot,
    });
}

static inline void human_input_queue_spec_toggle(HumanInput* hi) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_SPEC_TOGGLE,
    });
}

static inline void human_input_queue_equip_inventory_item(
    HumanInput* hi, int inventory_slot, int item_db_idx, int gear_slot
) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_EQUIP_INVENTORY_ITEM,
        .inventory_slot = inventory_slot,
        .item_db_idx = item_db_idx,
        .gear_slot = gear_slot,
    });
}

static inline void human_input_queue_fight_style(HumanInput* hi, int fight_style) {
    human_input_queue_command(hi, (HumanCommand){
        .kind = HUMAN_COMMAND_FIGHT_STYLE,
        .fight_style = fight_style,
    });
}

#endif /* OSRS_HUMAN_INPUT_TYPES_H */
