#include "fc_api.h"
#include "fc_npc.h"
#include "fc_pathfinding.h"
#include "fc_prayer.h"
#include "fc_wave_internal.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

/*
 * fc_combat.c — OSRS combat math and pending hit resolution.
 *
 * Formulas adapted from osrs_combat_shared.h (PufferLib).
 *
 * PvM prayer semantics:
 *   Correct protection prayer BLOCKS 100% of the matching NPC attack style.
 *   This is standard OSRS PvM — NOT the PvP 60% reduction.
 *   Exceptions must be explicit per NPC/attack (e.g. Jad wrong-prayer still takes damage).
 */

/* ======================================================================== */
/* OSRS accuracy formula                                                     */
/* ======================================================================== */

float fc_hit_chance(int att_roll, int def_roll) {
    if (att_roll > def_roll)
        return 1.0f - (float)(def_roll + 2) / (2.0f * (float)(att_roll + 1));
    else
        return (float)att_roll / (2.0f * (float)(def_roll + 1));
}

/* ======================================================================== */
/* NPC attack/max-hit formulas                                               */
/* ======================================================================== */

int fc_npc_attack_roll(int att_level, int att_bonus) {
    /* NPCs use level + invisible_boost(9) × (bonus + 64) */
    return (att_level + 9) * (att_bonus + 64);
}

/* ======================================================================== */
/* Player defence roll                                                       */
/* ======================================================================== */

int fc_player_def_roll(const FcPlayer* p, FcAttackType attack_type) {
    int def_bonus;
    switch (attack_type) {
        case FC_ATTACK_TYPE_STAB:   def_bonus = p->defence_stab; break;
        case FC_ATTACK_TYPE_SLASH:  def_bonus = p->defence_slash; break;
        case FC_ATTACK_TYPE_CRUSH:  def_bonus = p->defence_crush; break;
        case FC_ATTACK_TYPE_RANGED: def_bonus = p->defence_ranged; break;
        case FC_ATTACK_TYPE_MAGIC:  def_bonus = p->defence_magic; break;
        default:                    def_bonus = 0; break;
    }

    int eff_def;
    if (attack_type == FC_ATTACK_TYPE_MAGIC) {
        /* OSRS truncates the Defence and Magic contributions separately. */
        eff_def = 3 * p->defence_level / 10 +
                  7 * p->magic_level / 10 + 8;
    } else {
        eff_def = p->defence_level + 8;
    }
    return eff_def * (def_bonus + 64);
}

/* ======================================================================== */
/* Player ranged attack / max-hit                                            */
/* ======================================================================== */

static int fc_player_effective_ranged_level(const FcPlayer* p) {
    /* Rapid is the active DPS style for both RCB and TBow in this sim. */
    return p->ranged_level + 8;
}

int fc_player_ranged_base_attack_roll(const FcPlayer* p) {
    int eff_ranged = fc_player_effective_ranged_level(p);
    return eff_ranged * (p->ranged_attack_bonus + 64);
}

static int fc_tbow_target_magic_level(const FcNpc* target) {
    const FcNpcStats* stats = fc_npc_get_stats(target->npc_type);
    int magic_level = stats->magic_level;

    if (magic_level < 0) magic_level = 0;
    if (magic_level > 250) magic_level = 250;  /* non-CoX cap */
    return magic_level;
}

int fc_tbow_accuracy_multiplier_pct(int target_magic_level) {
    int64_t magic = target_magic_level;
    if (magic < 0) magic = 0;
    if (magic > 250) magic = 250;

    int64_t inner = 3 * magic / 10;
    int64_t delta = inner - 100;
    int64_t pct = 140 + (3 * magic - 10) / 100 -
                  delta * delta / 100;
    if (pct < 0) pct = 0;
    if (pct > 140) pct = 140;
    return (int)pct;
}

int fc_tbow_damage_multiplier_pct(int target_magic_level) {
    int64_t magic = target_magic_level;
    if (magic < 0) magic = 0;
    if (magic > 250) magic = 250;

    int64_t inner = 3 * magic / 10;
    int64_t delta = inner - 140;
    int64_t pct = 250 + (3 * magic - 14) / 100 -
                  delta * delta / 100;
    if (pct < 0) pct = 0;
    if (pct > 250) pct = 250;
    return (int)pct;
}

static void fc_crystal_modifiers_bp(int crystal_piece_mask,
                                    int* accuracy_bp, int* damage_bp) {
    int mask = crystal_piece_mask & FC_CRYSTAL_PIECE_ALL;
    *accuracy_bp = 0;
    *damage_bp = 0;

    if (mask & FC_CRYSTAL_PIECE_HELM) {
        *accuracy_bp += FC_CRYSTAL_HELM_ACCURACY_BP;
        *damage_bp += FC_CRYSTAL_HELM_DAMAGE_BP;
    }
    if (mask & FC_CRYSTAL_PIECE_BODY) {
        *accuracy_bp += FC_CRYSTAL_BODY_ACCURACY_BP;
        *damage_bp += FC_CRYSTAL_BODY_DAMAGE_BP;
    }
    if (mask & FC_CRYSTAL_PIECE_LEGS) {
        *accuracy_bp += FC_CRYSTAL_LEGS_ACCURACY_BP;
        *damage_bp += FC_CRYSTAL_LEGS_DAMAGE_BP;
    }
}

static int fc_apply_basis_points(int value, int bonus_bp) {
    return (int)((int64_t)value * (10000 + bonus_bp) / 10000);
}

int fc_player_ranged_attack_roll(const FcPlayer* p, const FcNpc* target) {
    int attack_roll = fc_player_ranged_base_attack_roll(p);

    if (p->weapon_kind == FC_WEAPON_TWISTED_BOW && target) {
        attack_roll = (int)((int64_t)attack_roll *
            fc_tbow_accuracy_multiplier_pct(fc_tbow_target_magic_level(target)) /
            100);
    } else if (p->weapon_kind == FC_WEAPON_BOW_OF_FAERDHINEN) {
        int accuracy_bp;
        int damage_bp;
        fc_crystal_modifiers_bp(p->crystal_piece_mask,
                                &accuracy_bp, &damage_bp);
        (void)damage_bp;
        attack_roll = fc_apply_basis_points(attack_roll, accuracy_bp);
    }

    return attack_roll;
}

int fc_player_ranged_base_max_hit_hp(const FcPlayer* p) {
    int eff_str = fc_player_effective_ranged_level(p);
    return (int)(((int64_t)eff_str * (p->ranged_strength_bonus + 64) + 320) /
                 640);
}

int fc_player_ranged_final_max_hit_hp(const FcPlayer* p, const FcNpc* target) {
    int base_hp = fc_player_ranged_base_max_hit_hp(p);

    if (p->weapon_kind == FC_WEAPON_TWISTED_BOW && target) {
        base_hp = (int)((int64_t)base_hp *
            fc_tbow_damage_multiplier_pct(fc_tbow_target_magic_level(target)) /
            100);
    } else if (p->weapon_kind == FC_WEAPON_BOW_OF_FAERDHINEN) {
        int accuracy_bp;
        int damage_bp;
        fc_crystal_modifiers_bp(p->crystal_piece_mask,
                                &accuracy_bp, &damage_bp);
        (void)accuracy_bp;
        base_hp = fc_apply_basis_points(base_hp, damage_bp);
    }

    return base_hp;
}

static int fc_damage_max_valid(const FcState* state, int final_max_hit_hp) {
    return state != NULL && final_max_hit_hp >= 0 &&
           final_max_hit_hp <= INT_MAX / 10;
}

int fc_roll_player_damage_tenths(FcState* state, int final_max_hit_hp) {
    if (!fc_damage_max_valid(state, final_max_hit_hp) ||
        final_max_hit_hp == 0) {
        return 0;
    }
    int rolled_hp = fc_rng_int(state, final_max_hit_hp + 1);
    if (rolled_hp == 0) rolled_hp = 1;
    return rolled_hp * 10;
}

int fc_roll_npc_damage_tenths(FcState* state, int final_max_hit_hp) {
    if (!fc_damage_max_valid(state, final_max_hit_hp) ||
        final_max_hit_hp == 0) {
        return 0;
    }
    return fc_rng_int(state, final_max_hit_hp + 1) * 10;
}

/* ======================================================================== */
/* Prayer check                                                              */
/* ======================================================================== */

int fc_prayer_blocks_style(int prayer, int attack_style) {
    /*
     * PvM: correct protection prayer blocks 100% of matching style.
     * Our enum mapping:
     *   PRAYER_PROTECT_MELEE(1) blocks ATTACK_MELEE(1)
     *   PRAYER_PROTECT_RANGE(2) blocks ATTACK_RANGED(2)
     *   PRAYER_PROTECT_MAGIC(3) blocks ATTACK_MAGIC(3)
     */
    if (prayer == PRAYER_NONE || attack_style == ATTACK_NONE) return 0;
    return (prayer == attack_style) ? 1 : 0;
}

/* ======================================================================== */
/* Chebyshev distance to multi-tile NPC                                      */
/* ======================================================================== */

int fc_distance_to_npc(int px, int py, const FcNpc* npc) {
    return fc_distance_between_areas(px, py, 1,
                                     npc->x, npc->y, npc->size);
}

/* ======================================================================== */
/* Hit delay formulas                                                        */
/* ======================================================================== */

/*
 * OSRS projectile hit delay:
 *   travel_time = time_offset + (distance * multiplier)  [in client ticks, 20ms each]
 *   game_ticks = travel_time / 30 + 1                    [CLIENT_TICKS.toTicks() = n/30]
 *
 * Per-NPC projectile definitions from tzhaar_fight_cave.gfx.toml:
 *   tok_xil_shoot:   delay=32, height=256, curve=16, no offset/mult → default mult=5
 *   ket_zek_travel:  delay=28, height=128, curve=16, offset=8, mult=8
 *   tztok_jad_travel: delay=86, height=50, curve=16, mult=8, no offset
 *   Jad ranged: no projectile, fixed client delay=120
 *
 * Melee: delay 1 (resolves same tick in our system — queued then resolved in same tick loop)
 */

/* Player ranged projectile timing */
int fc_ranged_hit_delay(int distance) {
    /* Keep the existing lightweight projectile timing for player ranged attacks. */
    int travel = 5 * distance;  /* default multiplier for player ranged */
    return travel / 30 + 1;
}

/*
 * Per-NPC-type hit delay — uses exact projectile timing from Void 634 gfx.toml.
 * Called from NPC attack code for precise parity with RSPS.
 */
int fc_npc_hit_delay(int npc_type, int attack_style, int distance) {
    if (attack_style == ATTACK_MELEE) return 1;

    switch (npc_type) {
        case NPC_TOK_XIL:
            /* tok_xil_shoot: default mult=5, no offset */
            return (5 * distance) / 30 + 1;

        case NPC_KET_ZEK:
            /* ket_zek_travel: offset=8, mult=8 */
            return (8 + 8 * distance) / 30 + 1;

        case NPC_TZTOK_JAD:
            if (attack_style == ATTACK_MAGIC) {
                /* Keep at least one full policy decision tick between Jad's
                 * tell and impact, including when Magic is selected in melee. */
                int delay = (8 * distance) / 30 + 1;
                return delay < 2 ? 2 : delay;
            } else {
                /* Jad ranged: no projectile, fixed client delay=120 */
                return 120 / 30 + 1;  /* = 5 game ticks */
            }

        default:
            /* Fallback for any other ranged/magic NPC */
            if (attack_style == ATTACK_RANGED) return (5 * distance) / 30 + 1;
            return (8 + 8 * distance) / 30 + 1;
    }
}

/* ======================================================================== */
/* NPC defence roll (for player attack accuracy against NPC)                 */
/* ======================================================================== */

int fc_npc_def_roll(int def_level, int def_bonus) {
    /* NPC defence: (def_level + 9) × (def_bonus + 64) */
    return (def_level + 9) * (def_bonus + 64);
}

/* ======================================================================== */
/* Queue a pending hit                                                       */
/* ======================================================================== */

int fc_queue_pending_hit(FcPendingHit hits[], int* num_hits, int max_hits,
                         int damage, int ticks, int style, int source_idx,
                         int prayer_drain) {
    if (*num_hits >= max_hits) return 0;
    FcPendingHit* h = &hits[*num_hits];
    h->active = 1;
    h->damage = damage;
    h->ticks_remaining = ticks;
    h->attack_style = style;
    h->source_npc_idx = source_idx;
    h->prayer_drain = prayer_drain;
    h->prayer_snapshot = PRAYER_NONE;
    h->prayer_lock_tick = -1;
    (*num_hits)++;
    return 1;
}

/* ======================================================================== */
/* Resolve pending hits (called each tick)                                   */
/* ======================================================================== */

static void record_render_hit(FcState* state, int target_entity_type,
                              int target_npc_slot, int source_npc_slot,
                              int attack_style, int damage, int blocked) {
    FcRenderEvents* events = &state->render_events;
    if (events->hit_count >= FC_MAX_RENDER_HITS) return;

    FcRenderHit* hit = &events->hits[events->hit_count++];
    hit->target_entity_type = target_entity_type;
    hit->target_npc_slot = target_npc_slot;
    hit->source_npc_slot = source_npc_slot;
    hit->attack_style = attack_style;
    hit->damage = damage;
    hit->blocked = blocked;
}

void fc_resolve_player_pending_hits(FcState* state) {
    FcPlayer* p = &state->player;
    int write = 0;

    for (int i = 0; i < p->num_pending_hits; i++) {
        FcPendingHit* h = &p->pending_hits[i];
        if (!h->active) continue;

        h->ticks_remaining--;
        if (h->ticks_remaining <= 0) {
            /* Hit resolves now — use the prayer locked into this hit. */
            int locked_prayer = h->prayer_snapshot >= 0
                ? h->prayer_snapshot : PRAYER_NONE;
            int blocked = fc_prayer_blocks_style(locked_prayer, h->attack_style);
            int final_damage = blocked ? 0 : h->damage;

            /* Apply damage */
            p->current_hp -= final_damage;
            if (p->current_hp < 0) p->current_hp = 0;

            p->damage_taken_this_tick += final_damage;
            p->hit_style_this_tick = h->attack_style;
            p->hit_source_npc_type = state->npcs[h->source_npc_idx].npc_type;
            p->hit_locked_prayer_this_tick = locked_prayer;
            p->hit_blocked_this_tick = blocked;
            state->damage_taken_this_tick += final_damage;
            p->total_damage_taken += final_damage;
            p->hit_landed_this_tick = 1;
            record_render_hit(state, ENTITY_PLAYER, -1,
                              h->source_npc_idx, h->attack_style,
                              final_damage, blocked);

            /* Auto-retaliate: if player has no target, target the attacker.
             * approach_target stays 0 — player attacks in place, doesn't chase. */
            if (p->attack_target_idx < 0 && h->source_npc_idx >= 0) {
                FcNpc* attacker = &state->npcs[h->source_npc_idx];
                if (attacker->active && !attacker->is_dead) {
                    p->attack_target_idx = h->source_npc_idx;
                    p->approach_target = 0;  /* don't chase, attack from here */
                    p->approach_target_x = -1;
                    p->approach_target_y = -1;
                    p->approach_target_size = 0;
                }
            }

            /* Tz-Kih drains damage dealt + 1 Prayer point, with the base point
             * still applying to misses and prayer-blocked attacks. HP and Prayer
             * both use tenths internally, so final_damage can be added directly. */
            if (h->prayer_drain > 0) {
                int drain = h->prayer_drain;
                if (h->source_npc_idx >= 0 && h->source_npc_idx < FC_MAX_NPCS &&
                    state->npcs[h->source_npc_idx].npc_type == NPC_TZ_KIH) {
                    drain += final_damage;
                }
                int actual_drain = fc_prayer_apply_loss_tenths(p, drain);
                state->prayer_lost_this_tick += actual_drain;
                state->tz_kih_prayer_drain_this_tick += actual_drain;
                if (h->source_npc_idx >= 0 && h->source_npc_idx < FC_MAX_NPCS) {
                    state->npcs[h->source_npc_idx].prayer_drain_dealt_this_tick +=
                        actual_drain;
                }
            }

            /* Track prayer correctness. Correctly blocked Jad hits also use
             * the shared correct-prayer reward applied to every NPC. */
            if (state->npcs[h->source_npc_idx].npc_type == NPC_TZTOK_JAD) {
                if (blocked) {
                    state->correct_jad_prayer = 1;
                    state->correct_danger_prayer = 1;
                } else {
                    state->wrong_jad_prayer = 1;
                }
            } else if (h->attack_style == ATTACK_RANGED ||
                       h->attack_style == ATTACK_MAGIC ||
                       h->attack_style == ATTACK_MELEE) {
                if (blocked) state->correct_danger_prayer = 1;
                else state->wrong_danger_prayer = 1;
            }

            /* Episode-level hit analytics */
            if (locked_prayer != PRAYER_NONE) {
                if (blocked) {
                    state->ep_correct_blocks++;
                    state->ep_damage_blocked += h->damage;
                } else {
                    state->ep_wrong_prayer_hits++;
                }
            } else {
                state->ep_no_prayer_hits++;
            }

            h->active = 0;  /* consumed */
        } else {
            /* Still in flight — keep */
            if (write != i) p->pending_hits[write] = *h;
            write++;
        }
    }
    p->num_pending_hits = write;
}

static void complete_fight_caves(FcState* state) {
    state->jad_killed = 1;
    state->ep_jad_killed = 1;

    /* Jad death completes the cave and immediately despawns surviving
     * healers, matching the encounter lifecycle. */
    for (int i = 0; i < FC_MAX_NPCS; i++) {
        FcNpc* other = &state->npcs[i];
        if (other->active && !other->is_dead &&
            other->npc_type == NPC_YT_HURKOT) {
            other->is_dead = 1;
            other->died_this_tick = 1;
            other->death_timer = 0;
            state->npcs_remaining--;
        }
    }
    state->wave_just_cleared = 1;
    state->terminal = TERMINAL_CAVE_COMPLETE;
    fc_wave_record_current_duration(state);
}

static void resolve_npc_death(FcState* state, FcNpc* npc) {
    npc->is_dead = 1;
    npc->died_this_tick = 1;
    npc->death_timer = 3;

    /* Each entity is one kill. A Tz-Kek parent is counted here before its two
     * children are counted through this same path later. */
    state->npcs_killed_this_tick++;
    if (npc->npc_type == NPC_YT_HURKOT &&
        npc->is_respawned_jad_healer) {
        state->respawned_jad_healers_killed_this_tick++;
    }
    state->total_npcs_killed++;

    if (npc->npc_type == NPC_TZTOK_JAD) {
        complete_fight_caves(state);
    }

    /* The Tz-Kek parent was pre-counted as two at spawn. Only its split
     * children decrement the wave's remaining-work count when they die. */
    if (npc->npc_type == NPC_TZ_KEK) {
        fc_npc_tz_kek_split(state, npc->x, npc->y);
    } else {
        state->npcs_remaining--;
    }
}

void fc_resolve_npc_pending_hits(FcState* state, int npc_idx) {
    FcNpc* npc = &state->npcs[npc_idx];
    int write = 0;

    for (int i = 0; i < npc->num_pending_hits; i++) {
        FcPendingHit* h = &npc->pending_hits[i];
        if (!h->active) continue;

        h->ticks_remaining--;
        if (h->ticks_remaining <= 0) {
            /* Player's hit lands on NPC */
            npc->current_hp -= h->damage;
            if (npc->current_hp < 0) npc->current_hp = 0;

            npc->damage_taken_this_tick += h->damage;
            state->damage_dealt_this_tick += h->damage;
            if (npc->npc_type > NPC_NONE && npc->npc_type < NPC_TYPE_COUNT) {
                state->ep_resolved_hits_to_npc_type[npc->npc_type]++;
                state->ep_damage_to_npc_type[npc->npc_type] += h->damage;
                if (h->damage > 0) {
                    state->ep_damaging_hits_to_npc_type[npc->npc_type]++;
                }
            }
            if (h->damage > 0) {
                state->hits_landed_this_tick++;
            }
            record_render_hit(state, ENTITY_NPC, npc_idx, -1,
                              h->attack_style, h->damage, 0);

            /* Track Jad-specific damage */
            if (npc->npc_type == NPC_TZTOK_JAD) {
                state->jad_damage_this_tick += h->damage;
            }

            /* Yt-HurKot: any landed player attack distracts healer, including 0s. */
            if (npc->npc_type == NPC_YT_HURKOT) {
                npc->healer_distracted = 1;
                npc->heal_target_idx = -1;
            }

            /* NPC death — keep active for a few ticks so viewer can
             * show the killing hitsplat and death animation. */
            if (npc->current_hp <= 0 && !npc->is_dead) {
                resolve_npc_death(state, npc);
            }

            h->active = 0;
        } else {
            if (write != i) npc->pending_hits[write] = *h;
            write++;
        }
    }
    npc->num_pending_hits = write;
}
