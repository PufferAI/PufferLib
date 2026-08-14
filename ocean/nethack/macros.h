// The keystroke dialogue with NetHack: prompt predicates and auto-dismissal,
// plus the per-verb key sequences. Included by nethack.h after the struct.
#pragma once

// helpers

// demo-only: sees the topline after every engine step (macros overwrite it
// many times per RL step); NULL in training
static void (*nethack_msg_tap)(Nethack* env);

static void nethack_engine_step(Nethack* env) {
    env->ctx = nle_step(env->ctx, &env->obs);
    if (nethack_msg_tap) nethack_msg_tap(env);
}

static void nethack_send_key(Nethack* env, int key) {
    env->obs.action = key;
    nethack_engine_step(env);
}

// message ends with '?': single-key prompts NLE doesn't expose via misc[]
static int nethack_msg_is_prompt(const Nethack* env) {
    const unsigned char* m = env->message;
    if (!m[0]) return 0;
    int e = 0;
    while (e < NLE_MESSAGE_SIZE && m[e]) e++;
    while (e > 0 && m[e-1] == ' ') e--;
    return e > 0 && m[e-1] == '?';
}

static int nethack_msg_contains(const Nethack* env, const char* needle) {
    char buf[NLE_MESSAGE_SIZE + 1];
    memcpy(buf, env->message, NLE_MESSAGE_SIZE);
    buf[NLE_MESSAGE_SIZE] = '\0';
    return strstr(buf, needle) != NULL;
}

// parse a getobj bracket list ("[b-d f or ?*]") into cand[]; returns count
static int nethack_parse_candidates(const Nethack* env, char* cand, int cap) {
    const unsigned char* m = env->message;
    int i = 0;
    while (i < NLE_MESSAGE_SIZE && m[i] && m[i] != '[') i++;
    int n = 0;
    for (i++; i < NLE_MESSAGE_SIZE && m[i] && n < cap; i++) {
        unsigned char c = m[i];
        if (n == 0 && (c == '-' || c == ' ' || c == '$')) continue; // leading "- " (allownone) / "$" (gold)
        if (c == '-' && i + 1 < NLE_MESSAGE_SIZE) { // compactified run
            for (char x = cand[n-1] + 1; x <= (char)m[i+1] && n < cap; x++)
                cand[n++] = x;
            i++;
            continue;
        }
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) cand[n++] = (char)c;
        else break; // ' ' before "or ?*", ']', '#', ...: end of the letter list
    }
    return n;
}

// prompts

// dismiss passive prompts (welcome, --More--, getline) until the game is back
// at the main command prompt
static void nethack_drain_prompts(Nethack* env) {
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        // skill-advance notice can flash by mid-drain; flag the pending auto-claim
        if (nethack_msg_contains(env, "more confident in your")) env->enh_ready = 1;
        int yn = env->misc[NETHACK_MISC_YN];
        if (!yn && !env->misc[NETHACK_MISC_GETLIN] && !env->misc[NETHACK_MISC_XWAIT]) break;
        // alternate return/ESC: getpos-class pickers eat '\r' silently and
        // only ESC exits them (the t-frozen wedge class)
        env->obs.action = yn ? 27 : (i & 1 ? 27 : '\r');
        nethack_engine_step(env);
    }
}

// Answer sub-prompts the agent can't: yn prompts commit 'y' EXCEPT the
// no-return climb (ends the game as ESCAPED) and peaceful-attack confirms
// (hostilizes Minetown); those and everything else get ESC. Returns 1 if a
// sub-prompt fired (the illegal_penalty condition).
static int nethack_handle_prompts(Nethack* env) {
    // direction prompts stay live — the agent's next key answers them
    if (env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "n what direction"))
        return 0;
    // the pray confirm is a deliberate action's own prompt: commit, no penalty
    int praying = env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "to pray");
    // ring PUTON asks "Which ring-finger, Right or Left?" ('y' is invalid and
    // aborts): the action's own prompt — answer 'r', no penalty
    int ringq = env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, "ight or Left");
    int illegal = !praying && !ringq && (env->misc[NETHACK_MISC_YN] || env->misc[NETHACK_MISC_GETLIN]
               || nethack_msg_is_prompt(env));
    if (!illegal && !praying && !ringq) {
        if (env->misc[NETHACK_MISC_XWAIT]) nethack_drain_prompts(env);
        return 0;
    }
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        if (nethack_msg_contains(env, "more confident in your")) env->enh_ready = 1;
        int yn = env->misc[NETHACK_MISC_YN];
        if (!yn && !env->misc[NETHACK_MISC_GETLIN] && !env->misc[NETHACK_MISC_XWAIT]
            && !nethack_msg_is_prompt(env)) break;
        int ring = yn && nethack_msg_contains(env, "ight or Left");
        // commit 'y' ONLY to rendered y/n choices ("[yn"): getobj also polls
        // through yn_function but wants a LETTER — auto-'y' loops forever
        int commit = yn && nethack_msg_contains(env, "[yn")
                        && !nethack_msg_contains(env, "no return")
                        && !nethack_msg_contains(env, "eally attack");
        // shopkeeper's "<Shk> offers N gold pieces for your X.  Sell it?" is
        // ynaq-rendered, so commit already accepts it — count the conversion
        if (commit && nethack_msg_contains(env, "gold piece")
                   && nethack_msg_contains(env, "Sell"))
            env->stats.sells++;
        env->obs.action = ring ? 'r' : (commit ? 'y' : 27);
        nethack_engine_step(env);
    }
    if (illegal) env->stats.illegal_actions++;
    return illegal;
}

static void nethack_answer_direction(Nethack* env, int key) {
    if (!env->obs.done && env->misc[NETHACK_MISC_YN]
        && nethack_msg_contains(env, "n what direction"))
        nethack_send_key(env, key);
}

// selection menus (pickup pile, identify) yield inside xwaitforspace: answer
// select-all + RET; on a plain --More-- the '.' just bells and RET dismisses
static void nethack_answer_menu(Nethack* env) {
    for (int r = 0; r < 2 && !env->obs.done && env->misc[NETHACK_MISC_XWAIT]; r++) {
        env->obs.action = '.';
        nethack_engine_step(env);
        if (env->obs.done || !env->misc[NETHACK_MISC_XWAIT]) break;
        env->obs.action = '\r';
        nethack_engine_step(env);
    }
}

// verbs

// item-verb flow: press the command, walk the engine's prompts, choose the
// slot's letter at the getobj gate (ESC + bad_pick if the engine refuses it);
// returns 1 on a successful use
static int nethack_item_use(Nethack* env, int cmd, const char* gate,
                            const char* floor, int slot, long* stat, int* bad_pick) {
    nethack_send_key(env, cmd);
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN] && floor && nethack_msg_contains(env, floor)) {
            // quaff: decline fountain/sink offers so the potion prompt follows;
            // eat: accept floor food except the cockatrice family
            if (cmd == 'q' || nethack_msg_contains(env, "atrice")) env->obs.action = 'n';
            else {
                env->stats.floor_eats++;
                nethack_send_key(env, 'y');
                if (stat) (*stat)++;
                return 1;
            }
        }
        else if (env->misc[NETHACK_MISC_YN] && nethack_msg_contains(env, gate)) {
            char cand[52];
            int n = nethack_parse_candidates(env, cand, (int)sizeof(cand));
            char want = (char)env->inv_letters[slot];
            int ok = 0;
            for (int j = 0; j < n; j++) {
                if (cand[j] != want) continue;
                ok = 1;
                break;
            }
            nethack_send_key(env, ok ? want : 27);
            if (!ok) {
                if (bad_pick) *bad_pick = 1;
                return 0;
            }
            if (stat) (*stat)++;
            return 1;
        }
        else return 0;
        nethack_engine_step(env);
    }
    return 0;
}

// same-slot armor swap: take off the worn piece first so upgrading is atomic
static void nethack_wear_takeoff_conflict(Nethack* env, int slot) {
    int gn = env->inv_glyphs[slot] - NH_GLYPH_OBJ_OFF;
    int cat_new = (gn >= 0 && gn < NH_NUM_OBJECTS) ? nh_obj_armcat[gn] : -1;
    if (cat_new < 0) return;
    for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i]; i++) {
        if (!(env->inv_state[i * NLE_INV_STATE_FIELDS + 5] & 1)) continue;
        int gi = env->inv_glyphs[i] - NH_GLYPH_OBJ_OFF;
        int cat_i = (gi >= 0 && gi < NH_NUM_OBJECTS) ? nh_obj_armcat[gi] : -1;
        if (cat_i == cat_new && i != slot) {
            nethack_item_use(env, 'T', "take off", NULL, i, NULL, NULL);
            return;
        }
    }
}

static void nethack_verb_wield(Nethack* env, int slot, int* bad_pick) {
    // re-selecting the wielded weapon unwields (keeps WIELD reversible)
    if (env->inv_state[slot * NLE_INV_STATE_FIELDS + 5] & 2) {
        nethack_send_key(env, 'w');
        if (!env->obs.done && env->misc[NETHACK_MISC_YN]
            && nethack_msg_contains(env, "wield"))
            nethack_send_key(env, '-');
        env->stats.verb_uses[NETHACK_ACT_WIELD]++;
        return;
    }
    nethack_item_use(env, 'w', "wield", NULL, slot, &env->stats.verb_uses[NETHACK_ACT_WIELD], bad_pick);
}

// engrave Elbereth with a fingertip (E, '-', "Elbereth", RET) — the early
// game's strongest panic button; each stage gated on its expected prompt,
// aborts fall through to nethack_handle_prompts
static void nethack_do_elbereth(Nethack* env) {
    env->obs.action = 'E';
    nethack_engine_step(env);
    if (env->obs.done || !env->misc[NETHACK_MISC_YN]
        || !nethack_msg_contains(env, "write with")) return;
    env->obs.action = '-';
    nethack_engine_step(env);
    // the dust --More-- raises xwait ALONGSIDE the getlin — clear it before
    // typing; decline "add to current engraving?" so the fresh text replaces it
    const char* c = "Elbereth\r";
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done && *c; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN]
                 && nethack_msg_contains(env, "current engraving")) env->obs.action = 'n';
        else if (env->misc[NETHACK_MISC_GETLIN]) env->obs.action = (unsigned char)*c++;
        else break;
        nethack_engine_step(env);
    }
}

// engrave-test the first unidentified wand (E, wand letter, ride the
// dialogue): fire/lightning/digging formally identify (+10 score), the rest
// print their tell. Throwaway text 'x'; same prompt loop as Elbereth.
static void nethack_do_engrave_id(Nethack* env) {
    int slot = -1;
    for (int i = 0; i < NETHACK_INV_SLOTS && env->inv_letters[i]; i++) {
        if (env->inv_oclasses[i] != 11
            || env->inv_state[i * NLE_INV_STATE_FIELDS + 6] != 0) continue;
        int lb = nethack_letter_bit(env->inv_letters[i]);
        if (lb >= 0 && (env->engid_tested & (1ULL << lb))) continue;
        slot = i;
        break;
    }
    if (slot < 0) return;
    int lb = nethack_letter_bit(env->inv_letters[slot]);
    if (lb >= 0) env->engid_tested |= 1ULL << lb;
    env->obs.action = 'E';
    nethack_engine_step(env);
    if (env->obs.done || !env->misc[NETHACK_MISC_YN]
        || !nethack_msg_contains(env, "write with")) return;
    env->obs.action = env->inv_letters[slot];
    nethack_engine_step(env);
    const char* c = "x\r";
    for (int i = 0; i < NETHACK_AUTODISMISS_MAX && !env->obs.done && *c; i++) {
        if (env->misc[NETHACK_MISC_XWAIT]) env->obs.action = ' ';
        else if (env->misc[NETHACK_MISC_YN]
                 && nethack_msg_contains(env, "current engraving")) env->obs.action = 'n';
        else if (env->misc[NETHACK_MISC_GETLIN]) env->obs.action = (unsigned char)*c++;
        else break;
        nethack_engine_step(env);
    }
}

// claim a banked skill advance the moment its notice appears (zero-turn,
// unconditionally good): #enhance, then 'a' = the first advanceable skill
static void nethack_auto_enhance(Nethack* env) {
    if (env->obs.done) return;
    if (nethack_msg_contains(env, "more confident in your")) env->enh_ready = 1;
    if (!env->enh_ready) return;
    env->enh_ready = 0;
    env->stats.enhances++;
    nethack_send_key(env, '#');
    // the extcmd prompt is NOT getlin-flagged: it renders "# " on the topline
    // and autocompletes; typing past the completion point is accepted
    if (!env->obs.done && nethack_msg_contains(env, "# ")) {
        for (const char* c = "enhance\r"; !env->obs.done && *c; c++)
            nethack_send_key(env, (unsigned char)*c);
        if (!env->obs.done && env->misc[NETHACK_MISC_XWAIT])
            nethack_send_key(env, 'a');
    }
    nethack_drain_prompts(env);
    if (!env->obs.done) nle_obs_refresh(env->ctx, &env->obs);
}
