// Static data for the NetHack env: glyph and layout constants, the action
// space and verb table, engine options, and the telemetry structs.
#pragma once

// object tables

// object-type -> armor slot (ARM_SUIT=0..ARM_SHIRT=6, -1 = not armor), indexed
// by otyp = glyph - NH_GLYPH_OBJ_OFF; generated from the engine's objects[]
// (gen_obj_armcat, NetHack 3.6.6). Device copy inlined in ocean/nethack/nethack.cu.
#define NH_NUM_OBJECTS 453
#define NH_GLYPH_OBJ_OFF 1906
static const signed char nh_obj_armcat[NH_NUM_OBJECTS] = {
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,2,2,2,2,2,2,2,2,
  2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,5,5,5,
  5,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,3,3,3,3,
  4,4,4,4,4,4,4,4,4,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};

// glyphs

#define NETHACK_PAD_GLYPH 5976

// corpse glyphs: [GLYPH_BODY_OFF, +NUMMONS), display.h
#define NETHACK_GLYPH_BODY_OFF 1144
#define NETHACK_NUMMONS 381

// object glyphs [GLYPH_OBJ_OFF, GLYPH_CMAP_OFF); underfoot objects win over terrain
#define NETHACK_GLYPH_OBJ_LO 1906
#define NETHACK_GLYPH_OBJ_HI 2359

// cmap wall glyphs S_vwall..S_trwall; S_stone excluded (= "unexplored")
#define NETHACK_WALL_GLYPH_LO 2360
#define NETHACK_WALL_GLYPH_HI 2370

// stair/ladder/altar cmap glyphs: GLYPH_CMAP_OFF(2359) + S_upstair(23)..
#define NETHACK_GLYPH_UPSTAIR 2382
#define NETHACK_GLYPH_DNSTAIR 2383
#define NETHACK_GLYPH_UPLADDER 2384
#define NETHACK_GLYPH_DNLADDER 2385
#define NETHACK_GLYPH_ALTAR 2386

// obs layout: glyphs | blstats | extra | inventory | item state | discovered | message

#define NH_ROWS 21
#define NH_COLS 79
#define NH_GRID (NH_ROWS * NH_COLS)

// encoder views (GPU-side): egocentric crop + 5x5 patches over the grid
#define NETHACK_CROP 9
#define NETHACK_CROP_GRID (NETHACK_CROP * NETHACK_CROP)

#define NETHACK_NUM_OCLASSES 18 // MAXOCLASSES; inv_oclasses pads with 18
#define NETHACK_OFF_GLYPHS 0
#define NETHACK_OFF_BLSTATS (NH_GRID * 2)

// extra ints: [0] engraving, [1] prev_action, [2..] per-class inv counts,
// in-shop bit + affordability percent, spell block (known count + 8 quads of
// id/level/fail%/retention turns; know 0 = forgotten, the re-read cue),
// encumbrance percent (unclipped past 100) + raw carry capacity
#define NETHACK_SPELL_SLOTS 8
#define NETHACK_OFF_EXTRA (NETHACK_OFF_BLSTATS + NLE_BLSTATS_SIZE * 4)
#define NETHACK_EXTRA_INTS (2 + NETHACK_NUM_OCLASSES + 2 + 1 + 4 * NETHACK_SPELL_SLOTS + 2 + 13 + 5 + 2)
#define NETHACK_EXTRA_ROLEOH (2 + NETHACK_NUM_OCLASSES + 2 + 1 + 4 * NETHACK_SPELL_SLOTS + 2)
#define NETHACK_EXTRA_SHOP (2 + NETHACK_NUM_OCLASSES)
#define NETHACK_EXTRA_SPELL (NETHACK_EXTRA_SHOP + 2)
#define NETHACK_EXTRA_WEIGHT (NETHACK_EXTRA_SPELL + 1 + 4 * NETHACK_SPELL_SLOTS)

// inventory: 55 slot glyphs (slot heads index these), then 8 gated int8
// state fields per slot [buc, spe, quan, ero1, ero2, flags, typeknown, rsvd],
// then discovered-type glyphs (true otyp once dknown && oc_name_known, else pad)
#define NETHACK_INV_SLOTS NLE_INVENTORY_SIZE
#define NETHACK_OFF_INV (NETHACK_OFF_EXTRA + NETHACK_EXTRA_INTS * 4)
#define NETHACK_OFF_INVST (NETHACK_OFF_INV + NETHACK_INV_SLOTS * 2)
#define NETHACK_OFF_INVTRUE (NETHACK_OFF_INVST + NETHACK_INV_SLOTS * NLE_INV_STATE_FIELDS)

// raw topline chars, null-padded; must match NH_MSG_LEN in ocean/nethack/nethack.cu
#define NETHACK_OFF_MSG (NETHACK_OFF_INVTRUE + NETHACK_INV_SLOTS * 2)
#define NETHACK_MSG_LEN 128
#define NETHACK_OBS_SIZE (NETHACK_OFF_MSG + NETHACK_MSG_LEN)

// engine state

#define NETHACK_PATH_MAX 128 // engine records up to 128 hero tiles/step
#define NETHACK_INTERNAL_KILLER_MLEV 10 // killer monster level, death only

#define NETHACK_MAX_EPISODE_STEPS 100000
#define NETHACK_AUTODISMISS_MAX 64 // cap on prompt-dismiss keystrokes per step
#define NETHACK_MAX_DEPTH 64 // scout bitmaps: max distinct (dnum, dlevel) floors per episode

// hunger states (hack.h): SATIATED 0, NOT_HUNGRY 1, HUNGRY 2, WEAK 3, FAINTING 4
#define NETHACK_HUNGER_WEAK 3

// nle_obs.misc[] prompt-state flags
enum { NETHACK_MISC_YN = 0, NETHACK_MISC_GETLIN = 1, NETHACK_MISC_XWAIT = 2 };

// actions

// action space: verb head (26) + 12 item-slot heads (55) + 6 per-verb
// direction heads (8 each: MOVE RUN KICK THROW ZAP APPLY) + spell-slot head (8)
#define NETHACK_NUM_ACTIONS 26
#define NETHACK_NUM_DIRS 8
#define NETHACK_DIR_HEADS 6

static const int NETHACK_DIR_KEYS[NETHACK_NUM_DIRS] =
    {'k','j','h','l','y','u','b','n'}; // N S W E NW NE SW SE
static const int NETHACK_DIR_DX[NETHACK_NUM_DIRS] = { 0, 0,-1, 1,-1, 1,-1, 1};
static const int NETHACK_DIR_DY[NETHACK_NUM_DIRS] = {-1, 1, 0, 0,-1,-1, 1, 1};

enum {
    NETHACK_ACT_MOVE = 0,
    NETHACK_ACT_RUN = 1,
    NETHACK_ACT_DOWN = 2,
    NETHACK_ACT_UP = 3,
    NETHACK_ACT_KICK = 4,
    NETHACK_ACT_SEARCH = 5,
    NETHACK_ACT_ELBERETH = 6,
    NETHACK_ACT_WEAR = 7,
    NETHACK_ACT_EAT = 8,
    NETHACK_ACT_QUAFF = 9,
    NETHACK_ACT_PRAY = 10,
    NETHACK_ACT_THROW = 11,
    NETHACK_ACT_ZAP = 12,
    NETHACK_ACT_SEARCH20 = 13, // count-prefixed search: ~20 turns of rest
    NETHACK_ACT_PICKUP = 14, // no slot: grab the pile underfoot (beyond autopickup)
    NETHACK_ACT_TAKEOFF = 15,
    NETHACK_ACT_PUTON = 16,
    NETHACK_ACT_REMOVE = 17,
    NETHACK_ACT_WIELD = 18,
    NETHACK_ACT_APPLY = 19,
    NETHACK_ACT_READ = 20,
    NETHACK_ACT_DROP = 21,
    NETHACK_ACT_ALTAR_ID = 22, // bulk BUC-identify on an altar
    NETHACK_ACT_TIP = 23, // empty a floor container (chest/box) underfoot
    NETHACK_ACT_ENGRAVE_ID = 24, // engrave-test the first unidentified wand
    NETHACK_ACT_CAST = 25, // cast a known spell (fail% is in the obs)
};

// dir-head index (0..NETHACK_DIR_HEADS-1) for verbs that take a direction
static inline int nethack_dir_head(int verb) {
    if (verb == NETHACK_ACT_CAST) return 4; // ZAP's dir head + its hostile-ray mask
    switch (verb) {
        case NETHACK_ACT_MOVE: return 0;
        case NETHACK_ACT_RUN: return 1;
        case NETHACK_ACT_KICK: return 2;
        case NETHACK_ACT_THROW: return 3;
        case NETHACK_ACT_ZAP: return 4;
        case NETHACK_ACT_APPLY: return 5;
    }
    return -1;
}

// inventory letter -> bit index (a-z 0-25, A-Z 26-51), -1 for non-letters
static inline int nethack_letter_bit(int c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return 26 + c - 'A';
    return -1;
}

// engine options

#define NETHACK_OPTIONS_TAIL \
    "autopickup,color,disclose:+i +a +v +g +c +o," \
    "mention_walls,nobones,nocmdassist,nolegacy,nosparkle," \
    "pickup_burden:unencumbered," \
    "runmode:teleport,showexp,showscore,time,"

// !status_updates skips the status renderer + recalc_mapseen (~25% of engine)
#define NETHACK_DEFAULT_OPTIONS "name:Agent-mon-hum-neu-mal," NETHACK_OPTIONS_TAIL "!status_updates"

// verb table

// slot-head legality per verb (masking + decode; execution is nethack_execute)
enum { WORN_ANY = 0, WORN_ONLY = 1, UNWORN_ONLY = 2 };

typedef struct {
    signed char head; // -1 direct verb, 0..11 = item slot head
    unsigned int item_classes;
    unsigned char wornreq;
} Verb;

// DROP excludes COIN: drop-gold was a parser-vetoed no-op exploit
static const Verb NETHACK_VERBS[NETHACK_NUM_ACTIONS] = {
    {-1} /* MOVE */,
    {-1} /* RUN */,
    {-1} /* DOWN */,
    {-1} /* UP */,
    {-1} /* KICK */,
    {-1} /* SEARCH */,
    {-1} /* ELBERETH */,
    {0, 1u<<3, UNWORN_ONLY} /* WEAR */,
    {1, 1u<<7, WORN_ANY} /* EAT */,
    {2, 1u<<8, WORN_ANY} /* QUAFF */,
    {-1} /* PRAY */,
    {3, 1u<<2, WORN_ANY} /* THROW */,
    {4, 1u<<11, WORN_ANY} /* ZAP */,
    {-1} /* SEARCH20 */,
    {-1} /* PICKUP */,
    {5, 1u<<3, WORN_ONLY} /* TAKEOFF */,
    {6, (1u<<4)|(1u<<5), UNWORN_ONLY} /* PUTON */,
    {7, (1u<<4)|(1u<<5), WORN_ONLY} /* REMOVE */,
    {8, 1u<<2, WORN_ANY} /* WIELD */,
    {9, 1u<<6, WORN_ANY} /* APPLY */,
    {10, (1u<<9)|(1u<<10), WORN_ANY} /* READ */,
    {11, 0x3EFFFu, UNWORN_ONLY} /* DROP */,
    {-1} /* ALTAR_ID */,
    {-1} /* TIP */,
    {-1} /* ENGRAVE_ID */,
    {-1} /* CAST */
};

// wandb key per verb success counter (NULL = not logged)
static const char* NETHACK_VERB_STAT[NETHACK_NUM_ACTIONS] = {
    NULL /* MOVE */,
    "runs" /* RUN */,
    NULL /* DOWN */,
    NULL /* UP */,
    NULL /* KICK */,
    "searches" /* SEARCH */,
    "engraves" /* ELBERETH */,
    "wears" /* WEAR */,
    "eats" /* EAT */,
    "quaffs" /* QUAFF */,
    "prayers" /* PRAY */,
    "throws" /* THROW */,
    "zaps" /* ZAP */,
    "search20" /* SEARCH20 */,
    "pickups" /* PICKUP */,
    "takeoffs" /* TAKEOFF */,
    "putons" /* PUTON */,
    "removes" /* REMOVE */,
    "wields" /* WIELD */,
    "applies" /* APPLY */,
    NULL /* READ: exported as reads_scroll + reads_book */,
    "drops" /* DROP */,
    "altar_ids" /* ALTAR_ID */,
    "tips" /* TIP */,
    "engrave_ids" /* ENGRAVE_ID */,
    "casts" /* CAST */
};

// telemetry

typedef struct Log {
    float perf;
    float verb_uses[NETHACK_NUM_ACTIONS]; // success counters, keys = NETHACK_VERB_STAT
    float score;
    float episode_return;
    float episode_length;
    float valid_moves; // steps that advanced NetHack's turn counter
    float illegal_actions; // steps that hit a sub-prompt we ESC'd
    float new_tiles;
    float max_depth; // deepest level reached (depth under-reports at death)
    float floors; // unique (dnum, dlevel) floors visited
    float depth_5, depth_10, depth_15; // fraction of episodes with max_depth >= N
    float mines_depth; // unique Gnomish Mines floors visited (0 = never entered)
    float sokoban_depth; // unique Sokoban floors visited (4 = reached the top)
    float scout_held; // steps where scout_ready withheld a tile claim
    float enhances; // #enhance presses (skill advancement claims)
    float floor_eats; // eats that accepted a floor "eat it?" offer
    float reads_scroll;
    float reads_book;
    float discoveries; // object types discovered this episode (oc_name_known delta)
    float sells; // shop sale offers accepted (deliberate drop in a shop)
    float buys; // shop pickups paid for
    float role_ix; // multi-role: sampled engine role index (mixing signature)
    float burdened_frac; // steps with encumbrance > Unencumbered
    float min_ac; // best (lowest) AC reached this episode
    float game_time; // NetHack turns survived
    float max_xp_level;
    // episode end reason (game_end_types in hack.h); other = traps/wrath/poison/...
    float death_combat;
    float death_starved;
    float death_other;
    float death_weak; // any death at Weak+ hunger (fainting included)
    float death_mon_level; // killer's monster level (vs max_xp_level = the mismatch)
    float death_ac; // AC on the last obs before death
    float truncated; // hit NETHACK_MAX_EPISODE_STEPS
    float n;
} Log;

// per-episode stats; cleared with one memset per reset
typedef struct Stats {
    long verb_uses[NETHACK_NUM_ACTIONS];
    long valid_moves;
    long illegal_actions;
    long new_tiles;
    long scout_held;
    long enhances;
    long burdened_steps;
    int min_ac;
    int last_ac; // AC on the last living obs (death-step blstats are torn down)
    long floor_eats;
    long reads_scroll;
    long reads_book;
    long sells;
    long buys;
    int last_hunger;
    int max_depth;
    int floors;
    unsigned long long floors_bits[16]; // dnum 0..15, bit dlevel-1
    int max_xp;
    float ret;
    int length;
    // per-tile scout claim ledger: the xp level at which the tile was last
    // claimed (0 = unclaimed), so tiles re-claim as the hero levels.
    // Branch levels sharing a depth share one slot.
    unsigned char visited[NETHACK_MAX_DEPTH][NH_GRID];
    unsigned short visited_key[NETHACK_MAX_DEPTH]; // dnum << 8 | dlevel per slot
    int n_visited_floors;
} Stats;
