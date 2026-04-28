/**
 * @fileoverview osrs_damage.h — shared OSRS damage application pipeline.
 *
 * pure function that computes the full damage chain: prayer reduction, vengeance
 * reflect, recoil reflect, smite drain. used by PvP (osrs_pvp_combat.h) and
 * available for any encounter that needs the full pipeline.
 *
 * DAMAGE PIPELINE:
 *   osrs_apply_damage_pipeline(...)            prayer -> veng -> recoil -> smite
 *
 * HELPERS:
 *   osrs_has_recoil_ring(equipped)             check ring of recoil / suffering (i)
 *
 * the pipeline is a pure function — computes damage chain without modifying state.
 * callers apply the returned DamageResult to their own game state.
 *
 * pending hits: each encounter manages its own pending hit queue since the queue
 * struct varies by encounter (PvP needs drain/heal/morr fields, PvE needs
 * spell_type/check_prayer). the damage pipeline runs when a hit lands, regardless
 * of how it was queued.
 *
 * ref: osrs wiki "protection prayers", "vengeance", "ring of recoil", "smite"
 * ref: osrs_pvp_combat.h:570-691 (PvP apply_damage)
 * ref: encounter_zulrah.h:651-675 (zulrah recoil)
 */

#ifndef OSRS_DAMAGE_H
#define OSRS_DAMAGE_H

#include "osrs_combat.h"
#include "osrs_items.h"


typedef struct {
    int final_damage;       /* damage after prayer reduction */
    int veng_damage;        /* reflected by vengeance (0 if inactive) */
    int recoil_damage;      /* reflected by recoil ring (0 if no ring) */
    int smite_drain;        /* prayer drained from target (0 if no smite) */
    int prayer_blocked;     /* 1 if correct prayer was active */
    int elysian_reduced;    /* 1 if elysian proc reduced post-prayer damage */
} DamageResult;

static inline DamageResult osrs_apply_post_mitigation_pipeline(
    int mitigated_damage,
    int prayer_blocked,
    int target_veng_active,
    int target_has_recoil,
    int attacker_smite_active
) {
    DamageResult r = {0, 0, 0, 0, prayer_blocked, 0};
    r.final_damage = mitigated_damage;

    if (target_veng_active && r.final_damage > 0) {
        r.veng_damage = (int)(r.final_damage * 0.75f);
    }

    if (target_has_recoil && r.final_damage > 0) {
        r.recoil_damage = r.final_damage / 10 + 1;
    }

    if (attacker_smite_active && r.final_damage > 0) {
        r.smite_drain = r.final_damage / 4;
    }

    return r;
}

/* apply the full OSRS damage pipeline to a hit.
   pure function — does NOT modify any state. caller applies the result.

   raw_damage: damage before any reduction
   attack_style: ATTACK_STYLE_MELEE/RANGED/MAGIC
   target_prayer: target's active overhead prayer (OverheadPrayer enum)
   is_pvp: 1 = 40% prayer reduction, 0 = 100% block
   target_veng_active: 1 if target has vengeance active
   target_has_recoil: 1 if target has ring of recoil / ring of suffering (i)
   attacker_smite_active: 1 if attacker has smite prayer active

   pipeline order:
     1. prayer reduction (osrs_prayer_reduce_damage from osrs_combat.h)
     2. vengeance: floor(final_damage * 0.75) reflected to attacker
     3. recoil: floor(final_damage * 0.1) + 1 reflected to attacker
     4. smite: floor(final_damage / 4) drained from target prayer

   ref: osrs_pvp_combat.h:570-691, encounter_zulrah.h:651-675 */
static inline DamageResult osrs_apply_damage_pipeline(
    int raw_damage, int attack_style,
    int target_prayer, int is_pvp,
    int target_veng_active,
    int target_has_recoil,
    int attacker_smite_active
) {
    int prayer_correct = encounter_prayer_correct_for_style(target_prayer, attack_style);
    int post_prayer_damage = osrs_prayer_reduce_damage(
        raw_damage, target_prayer, attack_style, is_pvp
    );
    return osrs_apply_post_mitigation_pipeline(
        post_prayer_damage,
        prayer_correct,
        target_veng_active,
        target_has_recoil,
        attacker_smite_active
    );
}


/* check if player has a recoil-capable ring equipped.
   ring of recoil (finite charges) or ring of suffering (i) (infinite).
   replaces has_recoil_effect() in osrs_pvp_combat.h:22 and
   zul_has_recoil_effect() in encounter_zulrah.h. */
static inline int osrs_has_recoil_ring(const uint8_t* equipped) {
    uint8_t ring = equipped[GEAR_SLOT_RING];
    return ring == ITEM_RING_OF_RECOIL || ring == ITEM_RING_OF_SUFFERING_RI;
}

#endif /* OSRS_DAMAGE_H */
