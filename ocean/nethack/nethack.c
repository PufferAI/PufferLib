#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif
#include <termios.h>
#include <sys/select.h>
#include <signal.h>
#include "nethack.h"
#include "../../src/puffercpu.c"
#include "glyph_map.h"

// NH_TTY=1: the map panel shows the game's real tty screen instead of the obs
static int demo_tty;
static unsigned char demo_tty_chars[NLE_TERM_LI * NLE_TERM_CO];
static signed char demo_tty_colors[NLE_TERM_LI * NLE_TERM_CO];
static unsigned char demo_tty_cursor[2];
static void demo_note_message(Nethack* env);
static void demo_view_setup(void) {
    nethack_msg_tap = demo_note_message; // ring sees every engine step's topline
    const char* t = getenv("NH_TTY");
    demo_tty = t && t[0] && t[0] != '0';
    if (!demo_tty) return;
    nethack_tty_chars_sink = demo_tty_chars;
    nethack_tty_colors_sink = demo_tty_colors;
    nethack_tty_cursor_sink = demo_tty_cursor;
}

// single-agent env, reset immediately (training's puf_reset is lazy)
static void env_open(Nethack* env) {
    demo_view_setup();
    nethack_want_chars = 1; // the demo renders and inspects the char grid
    memset(env, 0, sizeof(*env));
    // dungeon variety: rng feeds init()'s seed; srand() runs before env_open
    // in both demo modes, so NH_SEED replays exactly and no-seed varies by time
    env->rng = (unsigned)rand();
    env->num_agents = 1;
    Agent* a = &env->agents[0];
    a->observations = (unsigned char*)calloc(NETHACK_OBS_SIZE, 1);
    a->actions = (float*)calloc(20, sizeof(float)); // {verb, 12 per-verb slots, 6 per-verb dirs, spell slot}
    a->action_mask = (unsigned char*)calloc(NETHACK_NUM_ACTIONS
                      + 12 * NETHACK_INV_SLOTS + NETHACK_DIR_HEADS * NETHACK_NUM_DIRS
                      + NETHACK_SPELL_SLOTS, 1);
    a->rewards = (float*)calloc(1, sizeof(float));
    a->terminals = (float*)calloc(1, sizeof(float));
    // NH_ROLE=arc (demo only): pin that role; race/gender/align stay random.
    {
        const char* s = getenv("NH_ROLE");
        if (s && s[0]) {
            char buf[32];
            size_t n = 0;
            for (const char* p = s; *p && n + 1 < sizeof(buf); p++) {
                char c = *p;
                if (c >= 'A' && c <= 'Z') {
                    c = (char)(c - 'A' + 'a');
                }
                if (c == ' ' || c == '_' || c == '-') {
                    continue;
                }
                buf[n++] = c;
            }
            buf[n] = 0;
            static const struct { const char* key; const char* role; } map[] = {
                {"0", "archeologist"}, {"arc", "archeologist"},
                {"arch", "archeologist"}, {"archeologist", "archeologist"},
                {"archaeologist", "archeologist"}, {"archologist", "archeologist"},
                {"1", "barbarian"}, {"bar", "barbarian"}, {"barbarian", "barbarian"},
                {"2", "caveman"}, {"cav", "caveman"}, {"caveman", "caveman"},
                {"3", "healer"}, {"hea", "healer"}, {"healer", "healer"},
                {"4", "knight"}, {"kni", "knight"}, {"knight", "knight"},
                {"5", "monk"}, {"mon", "monk"}, {"monk", "monk"},
                {"6", "priest"}, {"pri", "priest"}, {"priest", "priest"},
                {"7", "rogue"}, {"rog", "rogue"}, {"rogue", "rogue"},
                {"8", "ranger"}, {"ran", "ranger"}, {"ranger", "ranger"},
                {"9", "samurai"}, {"sam", "samurai"}, {"samurai", "samurai"},
                {"10", "tourist"}, {"tou", "tourist"}, {"tourist", "tourist"},
                {"11", "valkyrie"}, {"val", "valkyrie"}, {"valkyrie", "valkyrie"},
                {"12", "wizard"}, {"wiz", "wizard"}, {"wizard", "wizard"},
            };
            const char* role = NULL;
            for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
                if (strcmp(buf, map[i].key) == 0) {
                    role = map[i].role;
                    break;
                }
            }
            if (!role) {
                fprintf(stderr, "nethack: unknown NH_ROLE=%s (try arc, bar, cav, ...)\n", s);
                exit(1);
            }
            static char opts[512];
            snprintf(opts, sizeof(opts),
                "name:Agent,role:%s,race:random,gender:random,align:random,"
                NETHACK_OPTIONS_TAIL "!status_updates", role);
            nethack_options_override = opts;
            fprintf(stderr, "nethack demo: NH_ROLE=%s\n", role);
        }
    }
    init(env);
    // NH_MULTI=1: random role/race/gender/align per reset (challenge protocol)
    const char* mr = getenv("NH_MULTI");
    if (mr && mr[0] && mr[0] != '0') env->multi_role = 1.0f;
    // the depth recipe trains with SEARCH20 and RUN masked; mirror it
    const char* wv = getenv("NH_WEIGHTS");
    if (wv && strcmp(wv, "depth") == 0)
        env->mask_search20 = env->mask_run = 1.0f;
    nethack_sync_buffers(env); // flat mask pointer, written by compute_mask
    nethack_do_reset(env);
}

static void env_close(Nethack* env) {
    puf_close(env);
    Agent* a = &env->agents[0];
    free(a->observations);
    free(a->actions);
    free(a->rewards);
    free(a->terminals);
    free(a->action_mask);
}

// CPU port of the CUDA encoder (ocean/nethack/nethack.cu) + puffernet MinGRU/decoder;
// weight order matches param registration: encoder, decoder, mingru
#define DEMO_VOCAB 5977
#define DEMO_EMBED 32
#define DEMO_BL_FEAT (25 + 7 + 13 + NETHACK_NUM_ACTIONS + NETHACK_NUM_OCLASSES + 2 + 8 + 2 + 2 \
                      + 1 + 2 + 20 + 8)
#define DEMO_SPKEY 16
#define DEMO_SPIN (DEMO_EMBED + 4)
#define DEMO_INV_HID 16 // 16-dim slot rep: pool bottleneck + decoder key (unified)
#define DEMO_INV_FLAT (NETHACK_INV_SLOTS * DEMO_INV_HID)
#define DEMO_INV_POOL 128
#define DEMO_SFEAT 24 // buc4 + known+spe + quan + ero2 + flags7 + tk + armcat7
#define DEMO_OD (NETHACK_NUM_ACTIONS + 12 * NETHACK_INV_SLOTS + NETHACK_DIR_HEADS * NETHACK_NUM_DIRS \
                 + NETHACK_SPELL_SLOTS)
#define DEMO_NUM_HEADS 20
#define DEMO_PTR_HEADS 12
#define DEMO_QDIM ((DEMO_PTR_HEADS + 1) * DEMO_INV_HID)
#define DEMO_DEC_LIN (NETHACK_NUM_ACTIONS + NETHACK_DIR_HEADS * NETHACK_NUM_DIRS + 1)
#define DEMO_DEC_PAD ((DEMO_DEC_LIN + 7) / 8 * 8)
#define DEMO_LOCC_DIM 8
#define DEMO_LOCC_CLASSES 9
#define DEMO_LOC_IN (NETHACK_CROP_GRID * DEMO_LOCC_DIM) // 9x9 semantic-class crop
#define DEMO_LOC_HID 256
#define DEMO_LOC_H1 256
#define DEMO_TERRF 592 // 12 landmarks x 4 + 8 sectors x 4 bands x 17 classes
#define DEMO_TERR_H1 256
#define DEMO_GLB_HID 128 // terrain branch output width
// trigram message branch, mirroring NH_MSG_* in ocean/nethack/nethack.cu
#define DEMO_MSG_LEN NETHACK_MSG_LEN
#define DEMO_MSG_VOCAB 1024
#define DEMO_MSG_LOG2V 10
#define DEMO_MSG_HID 256
#define DEMO_BLH 64
#define DEMO_MSG_CONCAT_OFF (DEMO_LOC_HID + DEMO_GLB_HID + DEMO_BLH + DEMO_BL_FEAT)
#define DEMO_SPELL_CONCAT_OFF (DEMO_MSG_CONCAT_OFF + DEMO_MSG_HID)
#define DEMO_SPM 64
#define DEMO_SPELL_SLICE (2*DEMO_SPM + 4) // sum | max entity pools + doorstep scalars
// identity-table channel (NH_ID_EMBED in nethack.cu); presence is inferred
// per-checkpoint so one binary loads both eras
#define DEMO_IDE_ROLE 16
#define DEMO_IDE_RACE 8
#define DEMO_IDE_GEND 8
#define DEMO_IDE_ALGN 8
#define DEMO_IDE_DIM (DEMO_IDE_ROLE + DEMO_IDE_RACE + DEMO_IDE_GEND + DEMO_IDE_ALGN)
#define DEMO_IDE_NUMEL (13*DEMO_IDE_ROLE + 5*DEMO_IDE_RACE + 2*DEMO_IDE_GEND + 3*DEMO_IDE_ALGN)
#define DEMO_IDE_CONCAT_OFF (DEMO_SPELL_CONCAT_OFF + DEMO_SPELL_SLICE)
// V6-min streams (nethack.cu): tok48 -> rep16 -> MLP 64 -> sum|max + gates
#define DEMO_LABK 16
#define DEMO_LAB_IN 48
#define DEMO_MV 64
#define DEMO_MINV_OFF (DEMO_IDE_CONCAT_OFF + DEMO_IDE_DIM)
#define DEMO_PASS_OFF (DEMO_MINV_OFF + 2*DEMO_MV)
#define DEMO_PACC 160 // accessory panel: [amulet | ring A | ring B | eyewear] x [r16|sfeat24]
#define DEMO_MLM_OFF (DEMO_PASS_OFF + 40 + 40 + DEMO_INV_HID + DEMO_PACC)
#define DEMO_MLI_OFF (DEMO_MLM_OFF + 2*DEMO_MV + DEMO_INV_HID)
#define DEMO_CONCAT (DEMO_MLI_OFF + 2*DEMO_MV + 2*DEMO_INV_HID)

// per-blstat normalization, mirroring NH_BL_SCALE / NH_BL_ISLOG in ocean/nethack/nethack.cu
static const float DEMO_BL_SCALE[27] = {
    1.f/79, 1.f/21,
    1.f/25, 1.f/125, 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,
    0.1f, 1.f/200, 1.f/200, 1.f/50, 0.1f,
    1.f/100, 1.f/100, 1.f/10, 1.f/10, 1.f/30,
    0.1f, 0.1f, 0.f, 1.f/4, 0.f, 1.f/50, 0.f, 1.f, // dnum one-hot (scale dead)
};
static const int DEMO_BL_ISLOG[27] =
    {0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0};

typedef struct {
    float *embed; // (5977, 32) E_res
    float *ekind_w, *esub_w; // (14, 32), (944, 32) factor tables
    float *e_eff; // materialized E_res + E_kind + E_sub
    float *loc_w, *loc_b; // (256, 648) over class-crop embeds
    float *terr1_w, *terr1_b, *terr2_w, *terr2_b; // 592 -> 256 -> 128 terrain MLP
    float *locc_w; // (9, 8) semantic-class embed table
    float *inv1_w, *inv1_b; // (16, 32), (16): per-slot features (pointer keys)
    float *inv1s_w; // (16, 24): gated item-state path into the slot MLP
    float *invt_w; // (16, 32): discovered-type channel (zero-init grown)
    float *mv1_w, *mv1_b, *mv2_w, *mv2_b; // inventory deep pool 16->64->64
    float *bl_w, *bl_b; // (64, DEMO_BL_FEAT), (64)
    float *proj_w, *proj_b; // (H, DEMO_CONCAT), (H)
    float *msg_w; // (4096, 32) trigram embedding table
    float *spk_w; // (16, 36) spell slot-rep projection
    float *spm1_w, *spm1_b, *spm2_w, *spm2_b; // spell per-slot MLP 16->64->64
    float *ide_role_w, *ide_race_w, *ide_gend_w, *ide_algn_w; // identity tables (ide era)
    float *dec_lin; // (DEMO_DEC_PAD, H) bias-free; rows [26 verb | 48 dir | value], 75 used
    float *dec_q; // (DEMO_QDIM, H): thirteen stacked 16-dim queries (12 item + spell)
    float *dec_k; // (16, 16): key projection over slot features
    float *dec_tau; // (12,): per-head log cosine temperature
    MinGRU* mingru;
    Multidiscrete* md;
    int hidden_size, num_layers, num_actions;
    float *loc2_w, *loc2_b; float loc_h1[DEMO_LOC_H1];
    float terr_tf[DEMO_TERRF]; float terr_h[DEMO_TERR_H1];
    float *mr_w, *mr_b, *mm1_w, *mm1_b, *mm2_w, *mm2_b; // monster stream
    float *ir_w, *ir_b, *im1_w, *im1_b, *im2_w, *im2_b; // item stream
    float x[DEMO_LOC_IN]; // crop class embeds, flattened
    float wld[NETHACK_INV_SLOTS]; // per-slot wielded bit
    float sfeat[NETHACK_INV_SLOTS * DEMO_SFEAT]; // per-slot state features
    int otyp[NETHACK_INV_SLOTS]; // slot glyph -> otyp (-1 if not an object glyph)
    int   occ[NETHACK_INV_SLOTS]; // per-slot occupancy
    float slots[DEMO_INV_FLAT]; // per-slot post-relu features (decoder keys)
    float spkeys[NETHACK_SPELL_SLOTS * DEMO_SPKEY]; // relu'd spell slot reps
    float concat[DEMO_CONCAT]; // [local hid | global hid | inv pool | bl hidden | bl feats | msg]
    float logits[DEMO_OD + 1]; // assembled decoder output; last entry is value
    float* hidden; // (hidden_size)
} NethackNet;

// (hidden, layers) from the checkpoint float count:
//   total = ENC_FIXED + H*(DEMO_CONCAT + 1) + H*(32 + 192) + DEC_FIXED + L * 3*H*H
// All tensors land on 8-float boundaries; only tau (12) needs padding (+4).
#define DEMO_ENC_FIXED (DEMO_VOCAB*DEMO_EMBED \
                        + NH_GM_NKIND*DEMO_EMBED + NH_GM_NSUB*DEMO_EMBED \
                        + DEMO_LOC_H1*DEMO_LOC_IN + DEMO_LOC_H1 \
                        + DEMO_LOC_HID*DEMO_LOC_H1 + DEMO_LOC_HID \
                        + DEMO_TERR_H1*DEMO_TERRF + DEMO_TERR_H1 \
                        + DEMO_GLB_HID*DEMO_TERR_H1 + DEMO_GLB_HID \
                        + DEMO_LOCC_CLASSES*DEMO_LOCC_DIM \
                        + DEMO_INV_HID*DEMO_EMBED + DEMO_INV_HID \
                        + DEMO_INV_HID*DEMO_SFEAT \
                        + DEMO_INV_HID*DEMO_EMBED \
                        + DEMO_BLH*DEMO_BL_FEAT + DEMO_BLH \
                        + DEMO_MSG_VOCAB*DEMO_MSG_HID \
                        + DEMO_SPKEY*DEMO_SPIN \
                        + DEMO_SPM*DEMO_SPKEY + DEMO_SPM \
                        + DEMO_SPM*DEMO_SPM + DEMO_SPM \
                        + DEMO_IDE_NUMEL \
                        + DEMO_MV*DEMO_INV_HID + DEMO_MV \
                        + DEMO_MV*DEMO_MV + DEMO_MV \
                        + 2*(DEMO_INV_HID*DEMO_LAB_IN + DEMO_INV_HID \
                             + DEMO_MV*DEMO_INV_HID + DEMO_MV \
                             + DEMO_MV*DEMO_MV + DEMO_MV))
#define DEMO_DEC_FIXED (DEMO_INV_HID*DEMO_INV_HID + 16) // k_w + tau padded 12->16
// ambiguities are possible; prefer the fewest layers (real configs have <= 8)
static int demo_infer_arch(int total, int* hidden, int* layers, int* actions) {
    int best_l = 1 << 30;
    for (int H = 8; H <= 4096; H += 8) {
        long rem = (long)total - DEMO_ENC_FIXED - DEMO_DEC_FIXED
                 - (long)H * (DEMO_CONCAT + 1 + DEMO_DEC_PAD + DEMO_QDIM);
        long per_layer = 3L * H * H;
        if (rem <= 0) continue;
        if (rem % per_layer) continue;
        long L = rem / per_layer;
        if (L >= 1 && L < best_l) {
            best_l = (int)L;
            *hidden = H;
            *layers = (int)L;
            *actions = NETHACK_NUM_ACTIONS;
        }
    }
    return best_l == 1 << 30 ? -1 : 0;
}

// bf16 weights file (half the download for the web demo): the loader read it as
// float32 pairs; if no architecture fits that count, expand each 16-bit value.
static Weights* demo_expand_bf16(Weights* w) {
    int n = (w->size - 7) * 2;
    int h, l, a;
    if (demo_infer_arch(n, &h, &l, &a) != 0) return w;
    Weights* x = (Weights*)calloc(1, sizeof(Weights) + ((size_t)n + 7) * sizeof(float));
    x->data = (float*)(x + 1);
    const uint16_t* src = (const uint16_t*)w->data;
    for (int i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&x->data[i], &bits, sizeof(float));
    }
    x->size = n + 7;
    x->idx = 0;
    return x; // caller still owns (and later frees) the original buffer
}

static NethackNet* make_nethack_net(Weights* w) {
    NethackNet* net = (NethackNet*)calloc(1, sizeof(NethackNet));
    int h, l, a;
    if (demo_infer_arch(w->size - 7, &h, &l, &a) != 0) w = demo_expand_bf16(w);
    if (demo_infer_arch(w->size - 7, &net->hidden_size, &net->num_layers,
                        &net->num_actions) != 0) {
        fprintf(stderr, "nethack demo: cannot infer arch from %d floats — "
                "checkpoint is not a nethack policy with %d actions?\n",
                w->size - 7, NETHACK_NUM_ACTIONS);
        exit(1);
    }
    fprintf(stderr, "nethack demo: hidden=%d layers=%d actions=%d (%d floats)\n",
            net->hidden_size, net->num_layers, net->num_actions, w->size - 7);
    net->hidden = (float*)calloc(net->hidden_size, sizeof(float));
    net->embed = get_weights_aligned(w, DEMO_VOCAB * DEMO_EMBED);
    net->ekind_w = get_weights_aligned(w, NH_GM_NKIND * DEMO_EMBED);
    net->esub_w = get_weights_aligned(w, NH_GM_NSUB * DEMO_EMBED);
    net->loc_w = get_weights_aligned(w, DEMO_LOC_H1 * DEMO_LOC_IN);
    net->loc_b = get_weights_aligned(w, DEMO_LOC_H1);
    net->loc2_w = get_weights_aligned(w, DEMO_LOC_HID * DEMO_LOC_H1);
    net->loc2_b = get_weights_aligned(w, DEMO_LOC_HID);
    net->terr1_w = get_weights_aligned(w, DEMO_TERR_H1 * DEMO_TERRF);
    net->terr1_b = get_weights_aligned(w, DEMO_TERR_H1);
    net->terr2_w = get_weights_aligned(w, DEMO_GLB_HID * DEMO_TERR_H1);
    net->terr2_b = get_weights_aligned(w, DEMO_GLB_HID);
    net->locc_w = get_weights_aligned(w, DEMO_LOCC_CLASSES * DEMO_LOCC_DIM);
    net->inv1_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_EMBED);
    net->inv1_b = get_weights_aligned(w, DEMO_INV_HID);
    net->inv1s_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_SFEAT);
    net->invt_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_EMBED);
    net->bl_w = get_weights_aligned(w, DEMO_BLH * DEMO_BL_FEAT);
    net->bl_b = get_weights_aligned(w, DEMO_BLH);
    net->proj_w = get_weights_aligned(w, net->hidden_size * DEMO_CONCAT);
    net->proj_b = get_weights_aligned(w, net->hidden_size);
    net->msg_w = get_weights_aligned(w, DEMO_MSG_VOCAB * DEMO_MSG_HID);
    net->spk_w = get_weights_aligned(w, DEMO_SPKEY * DEMO_SPIN);
    net->spm1_w = get_weights_aligned(w, DEMO_SPM * DEMO_SPKEY);
    net->spm1_b = get_weights_aligned(w, DEMO_SPM);
    net->spm2_w = get_weights_aligned(w, DEMO_SPM * DEMO_SPM);
    net->spm2_b = get_weights_aligned(w, DEMO_SPM);
    net->ide_role_w = get_weights_aligned(w, 13 * DEMO_IDE_ROLE);
    net->ide_race_w = get_weights_aligned(w, 5 * DEMO_IDE_RACE);
    net->ide_gend_w = get_weights_aligned(w, 2 * DEMO_IDE_GEND);
    net->ide_algn_w = get_weights_aligned(w, 3 * DEMO_IDE_ALGN);
    net->mv1_w = get_weights_aligned(w, DEMO_MV * DEMO_INV_HID);
    net->mv1_b = get_weights_aligned(w, DEMO_MV);
    net->mv2_w = get_weights_aligned(w, DEMO_MV * DEMO_MV);
    net->mv2_b = get_weights_aligned(w, DEMO_MV);
    net->mr_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_LAB_IN);
    net->mr_b = get_weights_aligned(w, DEMO_INV_HID);
    net->mm1_w = get_weights_aligned(w, DEMO_MV * DEMO_INV_HID);
    net->mm1_b = get_weights_aligned(w, DEMO_MV);
    net->mm2_w = get_weights_aligned(w, DEMO_MV * DEMO_MV);
    net->mm2_b = get_weights_aligned(w, DEMO_MV);
    net->ir_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_LAB_IN);
    net->ir_b = get_weights_aligned(w, DEMO_INV_HID);
    net->im1_w = get_weights_aligned(w, DEMO_MV * DEMO_INV_HID);
    net->im1_b = get_weights_aligned(w, DEMO_MV);
    net->im2_w = get_weights_aligned(w, DEMO_MV * DEMO_MV);
    net->im2_b = get_weights_aligned(w, DEMO_MV);
    net->dec_lin = get_weights_aligned(w, DEMO_DEC_PAD * net->hidden_size);
    net->dec_q = get_weights_aligned(w, DEMO_QDIM * net->hidden_size);
    net->dec_k = get_weights_aligned(w, DEMO_INV_HID * DEMO_INV_HID);
    net->dec_tau = get_weights_aligned(w, DEMO_PTR_HEADS);
    net->mingru = make_mingru(w, 1, net->hidden_size, net->num_layers);
    static int logit_sizes[DEMO_NUM_HEADS] = {
        NETHACK_NUM_ACTIONS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS,
        NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS,
        NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS, NETHACK_INV_SLOTS,
        NETHACK_INV_SLOTS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS,
        NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_NUM_DIRS, NETHACK_SPELL_SLOTS};
    net->md = make_multidiscrete(1, logit_sizes, DEMO_NUM_HEADS);
    assert(w->idx == w->size - 7);
    // materialize the residual-factorized embedding once (host, load time)
    net->e_eff = (float*)malloc((size_t)DEMO_VOCAB * DEMO_EMBED * sizeof(float));
    for (int g = 0; g < DEMO_VOCAB; g++)
        for (int d = 0; d < DEMO_EMBED; d++)
            net->e_eff[g * DEMO_EMBED + d] = net->embed[g * DEMO_EMBED + d]
                + net->ekind_w[nh_glyph_kind[g] * DEMO_EMBED + d]
                + net->esub_w[nh_glyph_sub[g] * DEMO_EMBED + d];
    return net;
}

static inline int demo_msg_lc(int c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c; // lowercase; keep spaces/punct
}
static inline int demo_msg_hash(int c0, int c1, int c2) {
    unsigned key = ((unsigned)c0 << 16) | ((unsigned)c1 << 8) | (unsigned)c2;
    return (int)((key * 2654435761u) >> (32 - DEMO_MSG_LOG2V));
}
// normalized-sum trigram bag over the null-terminated topline; scaled by
// 1/sqrt(count+1), no relu (raw signed summary)
static void demo_msg_pool(NethackNet* net, const unsigned char* obs, float* out) {
    const unsigned char* m = obs + NETHACK_OFF_MSG;
    for (int d = 0; d < DEMO_MSG_HID; d++) out[d] = 0.0f;
    int count = 0;
    for (int t = 0; t <= DEMO_MSG_LEN - 3; t++) {
        int c0 = m[t], c1 = m[t + 1], c2 = m[t + 2];
        if (c0 == 0 || c1 == 0 || c2 == 0) break;
        int id = demo_msg_hash(demo_msg_lc(c0), demo_msg_lc(c1), demo_msg_lc(c2));
        count++;
        for (int d = 0; d < DEMO_MSG_HID; d++)
            out[d] += net->msg_w[(size_t)id * DEMO_MSG_HID + d];
    }
    float scale = 1.0f / sqrtf((float)count + 1.0f);
    for (int d = 0; d < DEMO_MSG_HID; d++) out[d] *= scale;

}

// blstats/extra live at unaligned byte offsets: assemble, don't cast
static int32_t demo_i32(const unsigned char* p) {
    int32_t v;
    memcpy(&v, p, 4);
    return v;
}

static int demo_glyph_at(const int16_t* glyphs, int r, int c) {
    if (r < 0 || r >= NH_ROWS || c < 0 || c >= NH_COLS) return NETHACK_PAD_GLYPH;
    int g = glyphs[r * NH_COLS + c];
    if (g < 0) g = 0;
    if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
    return g;
}

// glyph -> 9-class local id and glyph -> 17-class terrain id (mirrors nh_v5_luts_init)
static unsigned char demo_locc_lut[DEMO_VOCAB];
static unsigned char demo_terrc_lut[DEMO_VOCAB];
static void demo_v5_luts_init(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    for (int g = 0; g < DEMO_VOCAB; g++) {
        int ci = (g >= 2359 && g < 2359 + 96) ? g - 2359 : -1;
        unsigned char lc = 7; // other
        if ((ci >= 1 && ci <= 11) || ci == 17 || ci == 18) lc = 0;      // wall/bars/tree
        else if (ci >= 19 && ci <= 22) lc = 1;                          // floor/corridor
        else if (ci >= 12 && ci <= 16) lc = 2;                          // door
        else if (ci == 32 || ci == 34 || ci == 41
                 || (ci >= 42 && ci <= 64)) lc = 3;                     // hazard terrain
        else if (g >= 0 && g < 1144) lc = 4;                            // monster
        else if ((g >= 1906 && g < 2359) || (g >= 1144 && g < 1525)) lc = 5; // item
        else if (ci == 0) lc = 6;                                       // unexplored
        if (g == NETHACK_PAD_GLYPH) lc = 8;                             // off-map
        demo_locc_lut[g] = lc;
        unsigned char tc = 255; // not counted
        if (ci == 23 || ci == 25) tc = 0;                               // upstairs
        else if (ci == 24 || ci == 26) tc = 1;                          // downstairs
        else if (ci == 27) tc = 2;                                      // altar
        else if (ci == 31) tc = 3;                                      // fountain
        else if (ci == 30) tc = 4;                                      // sink
        else if (ci == 29) tc = 5;                                      // throne
        else if (ci >= 12 && ci <= 14) tc = 6;                          // open door
        else if (ci == 15 || ci == 16) tc = 7;                          // closed door
        else if (ci >= 42 && ci <= 64) tc = 8;                          // trap
        else if (ci == 28) tc = 9;                                      // grave
        else if (ci >= 1 && ci <= 11) tc = 10;                          // wall
        else if (ci == 17 || ci == 18) tc = 11;                         // bars/tree
        else if (ci == 0) tc = 12;                                      // stone
        else if (ci >= 19 && ci <= 22) tc = 13;                         // floor
        else if ((g >= 1906 && g < 2359) || (g >= 1144 && g < 1525)) tc = 14; // item
        else if (g >= 0 && g < 1144) tc = (g % 381 == 267) ? 16 : 15;   // shk / monster
        demo_terrc_lut[g] = tc;
    }
}

static int nethack_net_forward(NethackNet* net, const unsigned char* obs) { // fills decoder->output
    demo_v5_luts_init();
    const int16_t* glyphs = (const int16_t*)(obs + NETHACK_OFF_GLYPHS);
    const unsigned char* bl = obs + NETHACK_OFF_BLSTATS;

    // local view: 9x9 semantic-class crop (LUT), center forced to class 7
    int hx = demo_i32(bl), hy = demo_i32(bl + 4);
    int half = NETHACK_CROP / 2;
    for (int p = 0; p < NETHACK_CROP_GRID; p++) {
        int g = demo_glyph_at(glyphs, hy - half + p / NETHACK_CROP,
                              hx - half + p % NETHACK_CROP);
        int cls = p == NETHACK_CROP_GRID / 2 ? 7 : demo_locc_lut[g];
        memcpy(net->x + p * DEMO_LOCC_DIM, net->locc_w + cls * DEMO_LOCC_DIM,
               DEMO_LOCC_DIM * sizeof(float));
    }
    _linear(net->x, net->loc_w, net->loc_b, net->loc_h1, 1, DEMO_LOC_IN, DEMO_LOC_H1);
    _relu(net->loc_h1, net->loc_h1, DEMO_LOC_H1);
    _linear(net->loc_h1, net->loc2_w, net->loc2_b, net->concat, 1, DEMO_LOC_H1, DEMO_LOC_HID);
    _relu(net->concat, net->concat, DEMO_LOC_HID);

    // terrain branch: landmark table + sector radar (mirrors nh_terr_feat_kernel)
    {
        float* lm = net->terr_tf;           // 12 landmarks x [seen, dx, dy, dist]
        float* sec = net->terr_tf + 48;     // 8 sectors x 4 bands x 17 classes
        for (int i = 0; i < DEMO_TERRF; i++) net->terr_tf[i] = 0.0f;
        int lmd[12];
        for (int i = 0; i < 12; i++) lmd[i] = 1 << 30;
        int hcell = hy * NH_COLS + hx;
        for (int cell = 0; cell < NH_GRID; cell++) {
            int g = glyphs[cell];
            if (g < 0) g = 0;
            if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
            int tc = cell == hcell ? 13 : demo_terrc_lut[g];
            if (tc == 255) continue;
            int dy = cell / NH_COLS - hy, dx = cell % NH_COLS - hx;
            int ady = dy < 0 ? -dy : dy, adx = dx < 0 ? -dx : dx;
            int cheb = adx > ady ? adx : ady;
            if (tc < 12 && cheb < lmd[tc]) {
                lmd[tc] = cheb;
                lm[tc * 4 + 0] = 1.0f;
                lm[tc * 4 + 1] = (float)dx * (1.0f / 78.0f);
                lm[tc * 4 + 2] = (float)dy * (1.0f / 20.0f);
                lm[tc * 4 + 3] = (float)(cheb < 30 ? cheb : 30) * (1.0f / 30.0f);
            }
            float ang = atan2f((float)dy, (float)dx) + 3.14159265358979f;
            int sct = ((int)(ang / 0.78539816339745f)) & 7;
            int band = cheb < 3 ? 0 : cheb < 7 ? 1 : cheb < 15 ? 2 : 3;
            sec[(sct * 4 + band) * 17 + tc] += 1.0f;
        }
        float inv_log = 1.0f / logf(1660.0f);
        for (int i = 0; i < 8 * 4 * 17; i++) sec[i] = log1pf(sec[i]) * inv_log;
        _linear(net->terr_tf, net->terr1_w, net->terr1_b, net->terr_h, 1, DEMO_TERRF, DEMO_TERR_H1);
        _relu(net->terr_h, net->terr_h, DEMO_TERR_H1);
        float* glb = net->concat + DEMO_LOC_HID;
        _linear(net->terr_h, net->terr2_w, net->terr2_b, glb, 1, DEMO_TERR_H1, DEMO_GLB_HID);
        _relu(glb, glb, DEMO_GLB_HID);
    }
    // inventory entities: per-slot embed -> shared 32->32 linear+relu (kept
    // as the pointer decoder's keys), then 32 -> 128 with max over slots for
    // the trunk (matches the CUDA fused pool)
    const int16_t* inv = (const int16_t*)(obs + NETHACK_OFF_INV);
    const int16_t* invt = (const int16_t*)(obs + NETHACK_OFF_INVTRUE);
    const signed char* invst = (const signed char*)(obs + NETHACK_OFF_INVST);
    for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
        int g = inv[slot];
        if (g < 0) g = 0;
        if (g >= DEMO_VOCAB) g = DEMO_VOCAB - 1;
        const signed char* st = invst + slot * NLE_INV_STATE_FIELDS;
        float* sf = net->sfeat + slot * DEMO_SFEAT;
        for (int c = 0; c < 4; c++) sf[c] = st[0] == c ? 1.0f : 0.0f;
        int spe_known = st[1] != -128;
        sf[4] = (float)spe_known;
        int spe = st[1] < -7 ? -7 : (st[1] > 7 ? 7 : st[1]);
        sf[5] = spe_known ? (float)spe * (1.0f / 7.0f) : 0.0f;
        int quan = st[2] < 0 ? 0 : (st[2] > 30 ? 30 : st[2]);
        sf[6] = (float)quan * (1.0f / 30.0f);
        sf[7] = (float)st[3] * (1.0f / 3.0f);
        sf[8] = (float)st[4] * (1.0f / 3.0f);
        for (int c = 0; c < 7; c++) sf[9 + c] = (float)((st[5] >> c) & 1);
        sf[16] = (float)st[6];
        int ot = inv[slot] - NH_GLYPH_OBJ_OFF; // armor slot category one-hot
        int cat = (ot >= 0 && ot < NH_NUM_OBJECTS) ? nh_obj_armcat[ot] : -1;
        for (int c = 0; c < 7; c++) sf[17 + c] = cat == c ? 1.0f : 0.0f;
        net->otyp[slot] = (ot >= 0 && ot < NH_NUM_OBJECTS) ? ot : -1;
        net->wld[slot] = sf[10];
        net->occ[slot] = inv[slot] != NETHACK_PAD_GLYPH;
        float* h32 = net->slots + slot * DEMO_INV_HID;
        _linear(net->e_eff + g * DEMO_EMBED, net->inv1_w, net->inv1_b,
                h32, 1, DEMO_EMBED, DEMO_INV_HID);
        for (int k = 0; k < DEMO_INV_HID; k++)
            for (int j = 0; j < DEMO_SFEAT; j++)
                h32[k] += net->inv1s_w[k * DEMO_SFEAT + j] * sf[j];
        int gt = invt[slot]; // discovered-type channel; pad = unknown
        if (gt >= 0 && gt < DEMO_VOCAB - 1)
            for (int k = 0; k < DEMO_INV_HID; k++)
                for (int d = 0; d < DEMO_EMBED; d++)
                    h32[k] += net->invt_w[k * DEMO_EMBED + d]
                            * net->e_eff[gt * DEMO_EMBED + d];
        _relu(h32, h32, DEMO_INV_HID);
    }
    { // inventory deep pool + pass-throughs (mirrors mv MLP + nh_min_summax + nh_pass)
        float* dst = net->concat + DEMO_MINV_OFF;
        float mx[DEMO_MV];
        int any = 0;
        for (int d = 0; d < DEMO_MV; d++) { dst[d] = 0.0f; mx[d] = -1e30f; }
        for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
            if (!net->occ[slot]) continue;
            any = 1;
            const float* r = net->slots + slot * DEMO_INV_HID;
            float h1[DEMO_MV];
            for (int d = 0; d < DEMO_MV; d++) {
                float acc = net->mv1_b[d];
                for (int c = 0; c < DEMO_INV_HID; c++)
                    acc += net->mv1_w[d * DEMO_INV_HID + c] * r[c];
                h1[d] = acc > 0.f ? acc : 0.f;
            }
            for (int d = 0; d < DEMO_MV; d++) {
                float acc = net->mv2_b[d];
                for (int c = 0; c < DEMO_MV; c++)
                    acc += net->mv2_w[d * DEMO_MV + c] * h1[c];
                float v = acc > 0.f ? acc : 0.f;
                dst[d] += 0.2f * v;
                if (v > mx[d]) mx[d] = v;
            }
        }
        for (int d = 0; d < DEMO_MV; d++) dst[DEMO_MV + d] = any ? mx[d] : 0.f;
        float* ps = net->concat + DEMO_PASS_OFF; // wld [r|sf] | quiver [r|sf] | worn mean
        for (int d = 0; d < 40 + 40 + DEMO_INV_HID + DEMO_PACC; d++) ps[d] = 0.0f;
        int nworn = 0;
        for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
            if (!net->occ[slot]) continue;
            const float* r = net->slots + slot * DEMO_INV_HID;
            const float* sf = net->sfeat + slot * DEMO_SFEAT;
            if (sf[10] > 0.5f)
                for (int e = 0; e < 40; e++)
                    ps[e] += e < DEMO_INV_HID ? r[e] : sf[e - DEMO_INV_HID];
            if (sf[12] > 0.5f)
                for (int e = 0; e < 40; e++)
                    ps[40 + e] += e < DEMO_INV_HID ? r[e] : sf[e - DEMO_INV_HID];
            if (sf[9] > 0.5f) {
                nworn++;
                for (int e = 0; e < DEMO_INV_HID; e++) ps[80 + e] += r[e];
            }
        }
        if (nworn > 0)
            for (int e = 0; e < DEMO_INV_HID; e++) ps[80 + e] /= (float)nworn;
        { // accessory panel (mirrors nh_acc_slot): first worn owner of each slot wins
            int nring = 0, have[4] = {0, 0, 0, 0};
            for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
                if (!net->occ[slot]) continue;
                const float* sf = net->sfeat + slot * DEMO_SFEAT;
                if (sf[9] <= 0.5f) continue;
                int ot = net->otyp[slot], a = -1;
                if (ot >= 178 && ot <= 188) a = 0;
                else if (ot >= 207 && ot <= 209) a = 3;
                else if (ot >= 150 && ot <= 177) { if (nring < 2) a = 1 + nring; nring++; }
                if (a < 0 || have[a]) continue;
                have[a] = 1;
                const float* r = net->slots + slot * DEMO_INV_HID;
                float* dst = ps + 80 + DEMO_INV_HID + a * (DEMO_INV_HID + DEMO_SFEAT);
                for (int e = 0; e < DEMO_INV_HID + DEMO_SFEAT; e++)
                    dst[e] = e < DEMO_INV_HID ? r[e] : sf[e - DEMO_INV_HID];
            }
        }
    }
    { // typed token streams (mirrors nh_lab_tok_kernel + mr/mm MLPs + nh_min_summax + sgate)
        for (int is_mon = 1; is_mon >= 0; is_mon--) {
            int list_off = is_mon ? NETHACK_OFF_TOKM : NETHACK_OFF_TOKI;
            const float *rw = is_mon ? net->mr_w : net->ir_w;
            const float *rb = is_mon ? net->mr_b : net->ir_b;
            const float *w1 = is_mon ? net->mm1_w : net->im1_w;
            const float *b1 = is_mon ? net->mm1_b : net->im1_b;
            const float *w2 = is_mon ? net->mm2_w : net->im2_w;
            const float *b2 = is_mon ? net->mm2_b : net->im2_b;
            float* dst = net->concat + (is_mon ? DEMO_MLM_OFF : DEMO_MLI_OFF);
            float tok[DEMO_LABK][DEMO_LAB_IN], rep[DEMO_LABK][DEMO_INV_HID];
            int valid[DEMO_LABK];
            for (int k = 0; k < DEMO_LABK; k++) {
                const unsigned char* e = obs + list_off + k * NETHACK_V3_MONF;
                int row = e[0] | (e[1] << 8);
                int dx = (signed char)e[2], dy = (signed char)e[3];
                int f4 = e[4], f5 = e[5], f6 = e[6];
                int g = row <= 0 ? -1 : is_mon ? row - 1
                      : (row < 454 ? 1906 + row - 1 : 1144 + row - 454);
                valid[k] = g >= 0;
                float* o = tok[k];
                for (int d = 0; d < DEMO_LAB_IN; d++) o[d] = 0.f;
                if (g >= 0) {
                    memcpy(o, net->e_eff + (size_t)g * DEMO_EMBED, DEMO_EMBED * sizeof(float));
                    int adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy;
                    int cheb = adx > ady ? adx : ady;
                    float fdx = dx * (1.0f/40.0f), fdy = dy * (1.0f/11.0f);
                    o[32] = fdx < -1.f ? -1.f : (fdx > 1.f ? 1.f : fdx);
                    o[33] = fdy < -1.f ? -1.f : (fdy > 1.f ? 1.f : fdy);
                    o[34] = (cheb < 15 ? cheb : 15) * (1.0f/15.0f);
                    o[35] = k * (1.0f/15.0f);
                    if (is_mon) {
                        o[36] = (f4 & 1) ? 1.f : 0.f;
                        o[37] = (f4 & 8) ? 1.f : 0.f;
                        o[38] = (f4 & 4) ? 1.f : 0.f;
                        o[39] = cheb <= 1 ? 1.f : 0.f;
                        float df = f5 * 0.04f; o[40] = df > 1.f ? 1.f : df;
                        float sp = f6 * (1.0f/24.0f); o[41] = sp > 1.f ? 1.f : sp;
                        int hb = NH_MON_HAZ[(row - 1) % NH_MONS_STATIC_N];
                        for (int hz = 0; hz < 4; hz++) o[42 + hz] = (float)((hb >> hz) & 1);
                    } else {
                        o[36] = (f5 & 1) ? 1.f : 0.f;
                        o[37] = (f5 & 2) ? 1.f : 0.f;
                    }
                }
                for (int j = 0; j < DEMO_INV_HID; j++) { // rep16 (decoder-free stream key)
                    float acc = rb[j];
                    for (int c = 0; c < DEMO_LAB_IN; c++) acc += rw[j * DEMO_LAB_IN + c] * o[c];
                    rep[k][j] = acc > 0.f ? acc : 0.f;
                }
            }
            float mx[DEMO_MV];
            int any = 0;
            for (int d = 0; d < DEMO_MV; d++) { dst[d] = 0.0f; mx[d] = -1e30f; }
            for (int k = 0; k < DEMO_LABK; k++) {
                if (!valid[k]) continue;
                any = 1;
                float h1[DEMO_MV];
                for (int j = 0; j < DEMO_MV; j++) {
                    float acc = b1[j];
                    for (int c = 0; c < DEMO_INV_HID; c++)
                        acc += w1[j * DEMO_INV_HID + c] * rep[k][c];
                    h1[j] = acc > 0.f ? acc : 0.f;
                }
                for (int j = 0; j < DEMO_MV; j++) {
                    float acc = b2[j];
                    for (int c = 0; c < DEMO_MV; c++) acc += w2[j * DEMO_MV + c] * h1[c];
                    float v = acc > 0.f ? acc : 0.f;
                    dst[j] += 0.25f * v;
                    if (v > mx[j]) mx[j] = v;
                }
            }
            for (int d = 0; d < DEMO_MV; d++) dst[DEMO_MV + d] = any ? mx[d] : 0.f;
            float* g0 = dst + 2 * DEMO_MV; // token-0 gate rep
            for (int d = 0; d < DEMO_INV_HID; d++) g0[d] = valid[0] ? rep[0][d] : 0.f;
            if (!is_mon) { // underfoot sum over rep16 of underfoot items (tok[36])
                float* uf = g0 + DEMO_INV_HID;
                for (int d = 0; d < DEMO_INV_HID; d++) uf[d] = 0.f;
                for (int k = 0; k < DEMO_LABK; k++) {
                    if (!valid[k] || tok[k][36] <= 0.5f) continue;
                    for (int d = 0; d < DEMO_INV_HID; d++) uf[d] += rep[k][d];
                }
            }
        }
    }

    // blstats+extra features (25 scalars, hunger 7, cond bits 13, prev verb
    // one-hot, inv class counts, hp/ene frac, dnum one-hot, engraving bits)
    float* f = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + DEMO_BLH;
    int j = 0;
    for (int i = 0; i < 27; i++) {
        if (i == 21 || i == 25) continue; // hunger, condition: expanded below
        float v = (float)demo_i32(bl + 4*i);
        f[j++] = DEMO_BL_ISLOG[i] ? log1pf(fmaxf(v, 0.f)) * DEMO_BL_SCALE[i]
                                  : v * DEMO_BL_SCALE[i];
    }
    int h21 = demo_i32(bl + 4*21);
    int hunger = h21 < 0 ? 0 : (h21 > 6 ? 6 : h21);
    for (int h = 0; h < 7; h++) f[j++] = (h == hunger) ? 1.f : 0.f;
    for (int k = 0; k < 13; k++) f[j++] = (float)(((uint32_t)demo_i32(bl + 4*25) >> k) & 1u);
    const unsigned char* ex = obs + NETHACK_OFF_EXTRA;
    for (int h = 0; h < NETHACK_NUM_ACTIONS; h++) f[j++] = (h == demo_i32(ex + 4)) ? 1.f : 0.f;
    for (int k = 0; k < NETHACK_NUM_OCLASSES; k++) f[j++] = (float)demo_i32(ex + 4*(2 + k)) * 0.125f;
    for (int p = 0; p < 2; p++) { // hp_frac, ene_frac
        int cur = demo_i32(bl + 4*(p ? 14 : 10)), mx = demo_i32(bl + 4*(p ? 15 : 11));
        f[j++] = fminf(fmaxf((float)cur / (float)(mx > 1 ? mx : 1), 0.f), 1.f);
    }
    int d23 = demo_i32(bl + 4*23);
    int dnum = d23 < 0 ? 0 : (d23 > 7 ? 7 : d23);
    for (int d = 0; d < 8; d++) f[j++] = (d == dnum) ? 1.f : 0.f;
    int engr = demo_i32(ex);
    f[j++] = engr >= 1 ? 1.f : 0.f; // any engraving underfoot
    f[j++] = engr >= 2 ? 1.f : 0.f; // active Elbereth
    f[j++] = (float)demo_i32(ex + 4*NETHACK_EXTRA_SHOP); // in shop
    f[j++] = (float)demo_i32(ex + 4*(NETHACK_EXTRA_SHOP+1)) * 0.01f; // affordability
    // spell scalar; mirrors NH_F_SPELL: known count/8 only
    f[j++] = (float)demo_i32(ex + 4*NETHACK_EXTRA_SPELL) * 0.125f;
    { // encumbrance pair; mirrors NH_F_WEIGHT in nethack.cu
      float d = (float)demo_i32(ex + 4*(NETHACK_EXTRA_WEIGHT+0)) * 0.01f - 1.0f;
      f[j++] = d / (1.0f + fabsf(d));
      f[j++] = (float)demo_i32(ex + 4*(NETHACK_EXTRA_WEIGHT+1)) * 0.001f; }
    for (int k = 0; k < 20; k++) // role/race/gender one-hots; dead under the embed channel
        f[j++] = 0.f;
    for (int k = 0; k < 8; k++) f[j++] = (float)((demo_i32(ex + 4*NETHACK_EXTRA_INTRINS) >> k) & 1);
    for (int k = 0; k < DEMO_BL_FEAT; k++) f[k] = fminf(fmaxf(f[k], -1.f), 1.f);

    float* blout = net->concat + DEMO_LOC_HID + DEMO_GLB_HID;
    _linear(f, net->bl_w, net->bl_b, blout, 1, DEMO_BL_FEAT, DEMO_BLH);
    _relu(blout, blout, DEMO_BLH);

    demo_msg_pool(net, obs, net->concat + DEMO_MSG_CONCAT_OFF);

    { // spell-key path; mirrors nh_spkey_kernel + nh_sppool_kernel
      for (int s = 0; s < NETHACK_SPELL_SLOTS; s++) {
          const unsigned char* q = obs + NETHACK_OFF_EXTRA
                                 + 4*(NETHACK_EXTRA_SPELL + 1 + 4*s);
          int sid = demo_i32(q);
          float in[DEMO_SPIN];
          if (sid > 0) {
              int g = sid + 1906; if (g > 5975) g = 5975;
              for (int d = 0; d < DEMO_EMBED; d++) in[d] = net->e_eff[g * DEMO_EMBED + d];
          } else
              for (int d = 0; d < DEMO_EMBED; d++) in[d] = 0.f;
          in[DEMO_EMBED + 0] = sid > 0 ? 1.f : 0.f;
          float lv = (float)demo_i32(q + 4) * 0.142857f;
          float fl = (float)demo_i32(q + 8) * 0.01f;
          float kn = (float)demo_i32(q + 12) * 0.00005f;
          in[DEMO_EMBED + 1] = lv > 1.f ? 1.f : lv;
          in[DEMO_EMBED + 2] = fl > 1.f ? 1.f : fl;
          in[DEMO_EMBED + 3] = kn > 1.f ? 1.f : kn;
          for (int r = 0; r < DEMO_SPKEY; r++) {
              float acc = 0.f;
              for (int c = 0; c < DEMO_SPIN; c++) acc += net->spk_w[r * DEMO_SPIN + c] * in[c];
              net->spkeys[s * DEMO_SPKEY + r] = acc > 0.f ? acc : 0.f;
          }
      }
      float* sp = net->concat + DEMO_SPELL_CONCAT_OFF;
      for (int d = 0; d < 2*DEMO_SPM; d++) sp[d] = 0.0f;
      int mf = 999, ml = 0, nn = 0; long mr = 99999;
      for (int s = 0; s < NETHACK_SPELL_SLOTS; s++) {
          const unsigned char* q = obs + NETHACK_OFF_EXTRA
                                 + 4*(NETHACK_EXTRA_SPELL + 1 + 4*s);
          int sid = demo_i32(q);
          if (sid <= 0) continue;
          nn++;
          int lv = demo_i32(q + 4), fl = demo_i32(q + 8), kn = demo_i32(q + 12);
          if (fl < mf) mf = fl;
          if (lv > ml) ml = lv;
          if (kn < mr) mr = kn;
          // entity value: per-slot MLP over the (already relu'd) key
          float m1[DEMO_SPM], v;
          const float* k = net->spkeys + s * DEMO_SPKEY;
          for (int d = 0; d < DEMO_SPM; d++) {
              float acc = net->spm1_b[d];
              for (int c = 0; c < DEMO_SPKEY; c++) acc += net->spm1_w[d * DEMO_SPKEY + c] * k[c];
              m1[d] = acc > 0.f ? acc : 0.f;
          }
          for (int d = 0; d < DEMO_SPM; d++) {
              float acc = net->spm2_b[d];
              for (int c = 0; c < DEMO_SPM; c++) acc += net->spm2_w[d * DEMO_SPM + c] * m1[c];
              v = acc > 0.f ? acc : 0.f;
              sp[d] += 0.25f * v;
              if (v > sp[DEMO_SPM + d]) sp[DEMO_SPM + d] = v;
          }
      }
      if (nn == 0) { mf = 100; ml = 0; mr = 20000; } // empty book -> [1,0,0,1]
      float e0 = (float)mf * 0.01f, e1 = (float)ml * (1.0f/7.0f);
      float e2 = (float)(nn < 8 ? nn : 8) * 0.125f, e3 = (float)mr * 0.00005f;
      sp[2*DEMO_SPM + 0] = e0 < 0.f ? 0.f : (e0 > 1.f ? 1.f : e0);
      sp[2*DEMO_SPM + 1] = e1 < 0.f ? 0.f : (e1 > 1.f ? 1.f : e1);
      sp[2*DEMO_SPM + 2] = e2 < 0.f ? 0.f : (e2 > 1.f ? 1.f : e2);
      sp[2*DEMO_SPM + 3] = e3 < 0.f ? 0.f : (e3 > 1.f ? 1.f : e3);
    }
    { // identity table tail; mirrors nh_idemb_kernel (last set bit wins)
      float* ide = net->concat + DEMO_IDE_CONCAT_OFF;
      int r = 0, rc = 0, g = 0;
      for (int k = 0; k < 13; k++) if (demo_i32(ex + 4*(NETHACK_EXTRA_ROLEOH + k))) r = k;
      for (int k = 0; k < 5; k++) if (demo_i32(ex + 4*(NETHACK_EXTRA_ROLEOH + 13 + k))) rc = k;
      for (int k = 0; k < 2; k++) if (demo_i32(ex + 4*(NETHACK_EXTRA_ROLEOH + 18 + k))) g = k;
      int al = 1 - demo_i32(bl + 4*26);
      al = al < 0 ? 0 : (al > 2 ? 2 : al);
      for (int d = 0; d < DEMO_IDE_ROLE; d++) *ide++ = net->ide_role_w[r * DEMO_IDE_ROLE + d];
      for (int d = 0; d < DEMO_IDE_RACE; d++) *ide++ = net->ide_race_w[rc * DEMO_IDE_RACE + d];
      for (int d = 0; d < DEMO_IDE_GEND; d++) *ide++ = net->ide_gend_w[g * DEMO_IDE_GEND + d];
      for (int d = 0; d < DEMO_IDE_ALGN; d++) *ide++ = net->ide_algn_w[al * DEMO_IDE_ALGN + d];
    }
    _linear(net->concat, net->proj_w, net->proj_b, net->hidden, 1, DEMO_CONCAT, net->hidden_size);
    _relu(net->hidden, net->hidden, net->hidden_size);

    mingru(net->mingru, net->hidden);

    // pointer decoder: [22 verb | 12x55 slots | 8 dir | value]. verb/dir/value
    // from one bias-free linear; slot logit i = tau_h * cos(q_h, k_i) with
    // keys k_i projected from the per-slot features above.
    float* hs = net->mingru->output;
    int H = net->hidden_size;
    float tmp[DEMO_DEC_LIN];
    for (int r = 0; r < DEMO_DEC_LIN; r++) {
        float acc = 0.0f;
        for (int k = 0; k < H; k++) acc += net->dec_lin[r * H + k] * hs[k];
        tmp[r] = acc;
    }
    float q[DEMO_QDIM];
    for (int r = 0; r < DEMO_QDIM; r++) {
        float acc = 0.0f;
        for (int k = 0; k < H; k++) acc += net->dec_q[r * H + k] * hs[k];
        q[r] = acc;
    }
    // NH_DEC_COS=1: legacy full-cosine slot logits (pre-2026-08-25 checkpoints)
    static int dec_cos = -1;
    if (dec_cos < 0) { const char* e = getenv("NH_DEC_COS"); dec_cos = e && e[0] && e[0] != '0'; }
    float kmat[DEMO_INV_FLAT], kn[NETHACK_INV_SLOTS];
    for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
        float nk = 0.0f;
        for (int r = 0; r < DEMO_INV_HID; r++) {
            float acc = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                acc += net->dec_k[r * DEMO_INV_HID + k] * net->slots[i * DEMO_INV_HID + k];
            kmat[i * DEMO_INV_HID + r] = acc;
            nk += acc * acc;
        }
        kn[i] = dec_cos ? sqrtf(nk) + 1e-6f : 1.0f;
    }
    for (int a = 0; a < NETHACK_NUM_ACTIONS; a++) net->logits[a] = tmp[a];
    for (int h = 0; h < DEMO_PTR_HEADS; h++) {
        const float* qh = q + h * DEMO_INV_HID;
        float nq = 0.0f;
        for (int k = 0; k < DEMO_INV_HID; k++) nq += qh[k] * qh[k];
        nq = sqrtf(nq) + 1e-6f;
        for (int i = 0; i < NETHACK_INV_SLOTS; i++) {
            float dot = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                dot += qh[k] * kmat[i * DEMO_INV_HID + k];
            net->logits[NETHACK_NUM_ACTIONS + h * NETHACK_INV_SLOTS + i] =
                expf(net->dec_tau[h]) * dot / (nq * kn[i]);
        }
    }
    for (int d = 0; d < NETHACK_DIR_HEADS * NETHACK_NUM_DIRS; d++) // 48 dir rows
        net->logits[NETHACK_NUM_ACTIONS + DEMO_PTR_HEADS * NETHACK_INV_SLOTS + d] =
            tmp[NETHACK_NUM_ACTIONS + d];
    { // spell head: dot(q_spell, rep_s) / 4 (dot-product pointer)
      const float* qs = q + DEMO_PTR_HEADS * DEMO_INV_HID;
      for (int sp = 0; sp < NETHACK_SPELL_SLOTS; sp++) {
          float dot = 0.f;
          for (int k = 0; k < DEMO_SPKEY; k++) dot += qs[k] * net->spkeys[sp * DEMO_SPKEY + k];
          net->logits[NETHACK_NUM_ACTIONS + DEMO_PTR_HEADS * NETHACK_INV_SLOTS
                      + NETHACK_DIR_HEADS * NETHACK_NUM_DIRS + sp] = dot * 0.25f;
      }
    }
    net->logits[DEMO_OD] = tmp[NETHACK_NUM_ACTIONS + NETHACK_DIR_HEADS * NETHACK_NUM_DIRS]; // value
    return 0;
}

// interactive TTY demo
// Space: one step on press; hold advances at 5 Hz. Shift+Space (or hold S): 20 Hz.
// Terminals that support xterm modifyOtherKeys report Shift+Space as a CSI
// sequence; 'S' is the fallback for everything else. q / Esc quits.

static struct termios g_term_orig;
static int g_term_raw = 0;

static void demo_restore_term(void) {
    if (!g_term_raw) return;
    printf("\x1b[>4;0m"); // disable modifyOtherKeys
    printf("\x1b[?25h"); // show cursor
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_term_orig);
    g_term_raw = 0;
    fflush(stdout);
}

static void demo_on_signal(int sig) {
    (void)sig;
    demo_restore_term();
    _exit(128 + sig);
}

static void demo_raw_term(void) {
    if (!isatty(STDIN_FILENO)) return;
    if (tcgetattr(STDIN_FILENO, &g_term_orig) != 0) return;
    atexit(demo_restore_term);
    signal(SIGINT, demo_on_signal);
    signal(SIGTERM, demo_on_signal);
    struct termios t = g_term_orig;
    t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &t);
    // modifyOtherKeys level 2: Shift+Space -> CSI 27;2;32~
    printf("\x1b[>4;2m\x1b[?25l");
    fflush(stdout);
    g_term_raw = 1;
}

static double demo_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// NH_WEIGHTS overrides the checked-in demo weights
// NH_WEIGHTS: a path, or the shorthands "score" (multi-role) / "depth"
static const char* demo_find_weights(void) {
    const char* envw = getenv("NH_WEIGHTS");
    if (!envw || !envw[0] || strcmp(envw, "score") == 0)
        return "resources/nethack/nethack_score_weights.bin";
    if (strcmp(envw, "depth") == 0)
        return "resources/nethack/nethack_depth_weights.bin";
    return envw;
}

// Drain stdin. Returns a bitset: bit0=space, bit1=shift+space/S, bit2=quit.
// Hold is detected via OS auto-repeat (and CSI for Shift+Space).
#define DEMO_IN_SPACE 1
#define DEMO_IN_FAST 2
#define DEMO_IN_QUIT 4

// xterm/kitty encode modifiers as 1 + bitmask (Shift=1, Alt=2, Ctrl=4, ...)
static int demo_mod_shift(int mod_param) {
    int bits = mod_param > 0 ? mod_param - 1 : 0;
    return (bits & 1) != 0;
}

static int demo_poll_input(void) {
    int flags = 0;
    for (;;) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(STDIN_FILENO, &rfds);
        struct timeval tv = {0, 0};
        if (select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv) <= 0) break;
        unsigned char buf[64];
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) break;
        for (ssize_t i = 0; i < n; i++) {
            unsigned char c = buf[i];
            if (c == 'q' || c == 'Q') {
                flags |= DEMO_IN_QUIT;
            } else if (c == 0x1b) {
                // Esc alone (no following bytes in this read) => quit. If CSI,
                // parse modifyOtherKeys / kitty sequences for Space.
                if (i + 1 >= n || buf[i + 1] != '[') {
                    flags |= DEMO_IN_QUIT;
                    continue;
                }
                i++; // at '['
                int params[8], np = 0, val = 0, in_num = 0;
                memset(params, 0, sizeof(params));
                for (i++; i < n; i++) {
                    unsigned char d = buf[i];
                    if (d >= '0' && d <= '9') {
                        val = val * 10 + (d - '0');
                        in_num = 1;
                    } else if (d == ';') {
                        if (np < 8) params[np++] = in_num ? val : 0;
                        val = 0;
                        in_num = 0;
                    } else if (d >= 0x40 && d <= 0x7e) {
                        if (in_num && np < 8) params[np++] = val;
                        // xterm modifyOtherKeys: CSI 27 ; mod ; keycode ~
                        if (d == '~' && np >= 3 && params[0] == 27 && params[2] == 32)
                            flags |= demo_mod_shift(params[1]) ? DEMO_IN_FAST : DEMO_IN_SPACE;
                        // kitty CSI u: CSI 32 ; mod u
                        if (d == 'u' && np >= 1 && params[0] == 32) {
                            int mod = np >= 2 ? params[1] : 1;
                            flags |= demo_mod_shift(mod) ? DEMO_IN_FAST : DEMO_IN_SPACE;
                        }
                        break;
                    } else break;
                }
            } else if (c == ' ') {
                flags |= DEMO_IN_SPACE;
            } else if (c == 'S' || c == 's') {
                // fallback fast key when the terminal does not report Shift+Space
                flags |= DEMO_IN_FAST;
            }
        }
    }
    return flags;
}

static void demo_step_once(NethackNet* net, Nethack* env, float* acts_f,
                           float* ep_score, float* ep_len, float* ep_depth,
                           float* ep_xp, float* ep_gt) {
    nethack_net_forward(net, env->agents[0].observations);
    for (int i = 0; i < DEMO_OD; i++)
        if (!env->action_mask[i]) net->logits[i] = -1e9f;
    multidiscrete(net->md, net->logits, acts_f, 0, NULL); // illegal logits already set to -1e9 above
    for (int h = 0; h < DEMO_NUM_HEADS; h++) env->agents[0].actions[h] = acts_f[h];
    puf_step(env);
    if (env->agents[0].terminals[0] > 0.5f) {
        float d = env->log.max_depth - *ep_depth;
        float x = env->log.max_xp_level - *ep_xp;
        float g = env->log.game_time - *ep_gt;
        fprintf(stderr, "episode end: role=%d score=%.0f len=%.0f max_depth=%.0f xp=%.0f game_t=%.0f "
                "eats=%.1f floor_eats=%.1f wears=%.1f throws=%.1f\n",
                env->role_idx, env->log.score - *ep_score, env->log.episode_length - *ep_len, d, x, g,
                env->log.verb_uses[NETHACK_ACT_EAT], env->log.floor_eats,
                env->log.verb_uses[NETHACK_ACT_WEAR],
                env->log.verb_uses[NETHACK_ACT_THROW]);
        *ep_score = env->log.score;
        *ep_len = env->log.episode_length;
        *ep_depth = env->log.max_depth;
        *ep_xp = env->log.max_xp_level;
        *ep_gt = env->log.game_time;
        memset(net->mingru->state, 0,
               (size_t)net->num_layers * net->hidden_size * sizeof(float));
    }
}

// message history ring for the rich view
#define DEMO_MSG_RING 18
static char demo_msgs[DEMO_MSG_RING][96];
static int demo_msg_n = 0;

static void demo_note_message(Nethack* env) {
    if (!env->message[0]) return;
    char buf[96];
    int j = 0;
    for (; j < 95 && env->message[j]; j++) buf[j] = (char)env->message[j];
    buf[j] = 0;
    if (demo_msg_n > 0) {
        char* prev = demo_msgs[(demo_msg_n - 1) % DEMO_MSG_RING];
        if (strcmp(prev, buf) == 0) return;
        // getline echo: the game repaints the topline per keystroke; an
        // extension of the previous message replaces it (lossless)
        size_t pl = strlen(prev);
        if (pl > 0 && j > (int)pl && strncmp(prev, buf, pl) == 0) {
            strcpy(prev, buf);
            return;
        }
    }
    strcpy(demo_msgs[demo_msg_n % DEMO_MSG_RING], buf);
    demo_msg_n++;
}

// menucolor-style BUC tint
static const char* demo_inv_clr(Nethack* env, int i) {
    if (env->inv_oclasses[i] == 12) return "\x1b[33m"; // COIN_CLASS (objclass.h)
    int buc = env->inv_state[i * NLE_INV_STATE_FIELDS];
    return buc == 1 ? "\x1b[31m" : buc == 3 ? "\x1b[32m"
         : buc == 2 ? "\x1b[37m" : "\x1b[36m";
}

// curses-look cell: walls -> box drawing, floors -> '·', engine colors -> ANSI
#define DEMO_CMAP_OFF 2359
#define DEMO_PET_LO 381
#define DEMO_PET_HI 762
static unsigned char demo_colors[NH_GRID]; // bound via nethack_color_sink
static const char* demo_wall_sym[12] = { // S_stone..S_trwall
    " ", "│", "─", "┌", "┐", "└", "┘",
    "┼", "┴", "┬", "┤", "├"};
static const int demo_ansi_clr[16] = // CLR_* 0..15 -> SGR fg
    {90, 31, 32, 33, 34, 35, 36, 37, 39, 91, 92, 93, 94, 95, 96, 97};

static void demo_map_cell(Nethack* env, int r, int c, int hero) {
    // underfoot_glyphs hides the hero glyph; draw the @ explicitly
    if (hero) {
        printf("\x1b[7;1;97m@\x1b[0m");
        return;
    }
    int g = env->glyphs[r * NH_COLS + c];
    unsigned char ch = env->chars[r * NH_COLS + c];
    int cmap = g - DEMO_CMAP_OFF;
    if (cmap >= 1 && cmap <= 11) { // walls in PUFF_CYAN (connect4.h)
        printf("\x1b[38;2;0;187;187m%s\x1b[0m", demo_wall_sym[cmap]);
        return;
    }
    if (cmap == 19) { // S_room
        printf("\x1b[37m·\x1b[0m");
        return;
    }
    if (cmap == 20) { // S_darkroom
        printf("\x1b[2m·\x1b[0m");
        return;
    }
    if (!ch || ch == ' ') {
        putchar(' ');
        return;
    }
    int clr = demo_colors[r * NH_COLS + c] & 15;
    printf("\x1b[%s%dm%c\x1b[0m",
           (g >= DEMO_PET_LO && g < DEMO_PET_HI) ? "4;" : "",
           demo_ansi_clr[clr], ch);
}

// tty map cell: authentic chars/colors from the game's own screen (rows 1..21)
static void demo_map_cell_tty(int r, int c) {
    unsigned char ch = demo_tty_chars[(r + 1) * NLE_TERM_CO + (c + 1)];
    int clr = demo_tty_colors[(r + 1) * NLE_TERM_CO + (c + 1)] & 15;
    if (!ch || ch == ' ') {
        putchar(' ');
        return;
    }
    printf("\x1b[%dm%c\x1b[0m", demo_ansi_clr[clr], ch);
}

#define DEMO_INV_W 42 // inventory pane inner width

// perm_invent pane: doname text grouped by class in game display order
static unsigned char demo_inv_strs[NLE_INVENTORY_SIZE * NLE_INVENTORY_STR_LENGTH];
static const int demo_inv_order[] =
    {12, 5, 2, 3, 7, 9, 10, 8, 4, 11, 6, 13, 14, 15, 16, 17, 1};
static const char* demo_class_name[18] = {0, "Illegal objects", "Weapons",
    "Armor", "Rings", "Amulets", "Tools", "Comestibles", "Potions", "Scrolls",
    "Spellbooks", "Wands", "Coins", "Gems/Stones", "Boulders/Statues",
    "Iron balls", "Chains", "Venoms"};
#define DEMO_PANE_MAX 64

static int demo_inv_pane(Nethack* env, char lines[][DEMO_INV_W + 8],
                         const char* clrs[]) {
    int n = 0;
    for (int k = 0; k < (int)(sizeof(demo_inv_order)/sizeof(*demo_inv_order)); k++) {
        int cls = demo_inv_order[k], first = 1;
        for (int i = 0; i < NETHACK_INV_SLOTS && n < DEMO_PANE_MAX - 1; i++) {
            if (!env->inv_letters[i] || env->inv_oclasses[i] != cls) continue;
            if (first) {
                snprintf(lines[n], DEMO_INV_W + 8, "%s", demo_class_name[cls]);
                clrs[n++] = "\x1b[1m";
                first = 0;
            }
            snprintf(lines[n], DEMO_INV_W + 8, "%c) %s", env->inv_letters[i],
                     demo_inv_strs + i * NLE_INVENTORY_STR_LENGTH);
            clrs[n] = demo_inv_clr(env, i);
            n++;
        }
    }
    return n;
}

static void demo_box_edge(const char* l, const char* r, const char* title, int inner) {
    printf("%s", l);
    int n = 0;
    if (title) n = printf("─ %s ", title) - 2; // rule char is 3 bytes, 1 column
    for (; n < inner; n++) printf("─");
    printf("%s", r);
}

// right-column inventory box segment for composite row k (0 = top border)
static void demo_inv_row(int k, int last, int pn,
                         char plines[][DEMO_INV_W + 8], const char** pclrs) {
    if (k == 0) {
        printf(" ");
        demo_box_edge("┌", "┐\n", "Inventory", DEMO_INV_W);
        return;
    }
    if (k >= last) {
        printf(" ");
        demo_box_edge("└", "┘\n", NULL, DEMO_INV_W);
        return;
    }
    printf(" │ ");
    int i = k - 1, len = 0;
    if (i == last - 2 && pn > last - 1)
        len = printf("\x1b[2m... +%d more\x1b[0m", pn - (last - 2)) - 8;
    else if (i < pn)
        len = printf("%s%.*s\x1b[0m", pclrs[i], DEMO_INV_W - 2, plines[i])
            - (int)strlen(pclrs[i]) - 4;
    printf("%*s│\n", DEMO_INV_W - 1 - len, "");
}

// one box row: bytes in buf, visible columns in vis; segments past the box
// width are clipped so the row never wraps
typedef struct { char buf[512]; int pos, vis; } DemoLine;
static void demo_line_add(DemoLine* l, const char* esc, const char* text) {
    int n = (int)strlen(text), room = NH_COLS + 1 - l->vis; // rows are NH_COLS+2 wide with both edges
    if (room <= 0 || n <= 0) return;
    if (n > room) n = room;
    l->pos += snprintf(l->buf + l->pos, sizeof(l->buf) - l->pos, "%s%.*s%s",
                       esc, n, text, esc[0] ? "\x1b[0m" : "");
    l->vis += n;
}
static void demo_line_flush(DemoLine* l) {
    printf("%s%*s│", l->buf, NH_COLS + 1 - l->vis, "");
}

static void demo_render(Nethack* env, int rate_hz, long steps) {
    long* bl = env->blstats;
    printf("\x1b[H\x1b[2J");
    static char plines[DEMO_PANE_MAX][DEMO_INV_W + 8];
    static const char* pclrs[DEMO_PANE_MAX];
    int pn = demo_inv_pane(env, plines, pclrs);
    // left column: message box, map box, status box; inventory runs full height
    int last = 1 + DEMO_MSG_RING + 1 + 1 + NH_ROWS + 1 + 4 - 1;
    int k = 0;
    demo_box_edge("┌", "┐", NULL, NH_COLS);
    demo_inv_row(k++, last, pn, plines, pclrs);
    int shown = demo_msg_n < DEMO_MSG_RING ? demo_msg_n : DEMO_MSG_RING;
    for (int m = 0; m < DEMO_MSG_RING; m++) {
        printf("│ ");
        int len = 0;
        if (m >= DEMO_MSG_RING - shown) {
            int idx = (demo_msg_n - (DEMO_MSG_RING - m)) % DEMO_MSG_RING;
            len = printf("%s%.*s\x1b[0m", m == DEMO_MSG_RING - 1 ? "\x1b[1m" : "\x1b[2m",
                         NH_COLS - 2, demo_msgs[idx]) - 8;
        }
        printf("%*s│", NH_COLS - 1 - len, "");
        demo_inv_row(k++, last, pn, plines, pclrs);
    }
    demo_box_edge("└", "┘", NULL, NH_COLS);
    demo_inv_row(k++, last, pn, plines, pclrs);
    char title[32];
    snprintf(title, sizeof(title), "Dlvl:%ld", bl[NLE_BL_DEPTH]);
    demo_box_edge("┌", "┐", title, NH_COLS);
    demo_inv_row(k++, last, pn, plines, pclrs);
    for (int r = 0; r < NH_ROWS; r++) {
        printf("│");
        for (int c = 0; c < NH_COLS; c++) {
            if (demo_tty) demo_map_cell_tty(r, c);
            else demo_map_cell(env, r, c, c == bl[NLE_BL_X] && r == bl[NLE_BL_Y]);
        }
        printf("│");
        demo_inv_row(k++, last, pn, plines, pclrs);
    }
    demo_box_edge("└", "┘", NULL, NH_COLS);
    demo_inv_row(k++, last, pn, plines, pclrs);
    // status box: hitpointbar over the agent name, then the stat line
    demo_box_edge("┌", "┐", NULL, NH_COLS);
    demo_inv_row(k++, last, pn, plines, pclrs);
    long hp = bl[NLE_BL_HP], hpm = bl[NLE_BL_HPMAX] > 0 ? bl[NLE_BL_HPMAX] : 1;
    static const char* rolenm[13] = {"Archeologist", "Barbarian", "Caveman",
        "Healer", "Knight", "Monk", "Priest", "Rogue", "Ranger", "Samurai",
        "Tourist", "Valkyrie", "Wizard"};
    static const char* racenm[5] = {"human", "elven", "dwarven", "gnomish", "orcish"};
    static const char* alignnm[3] = {"lawful", "neutral", "chaotic"};
    int ir = 0, ic = 0, ig = 0, ia = 0;
    nle_identity(env->ctx, &ir, &ic, &ig, &ia);
    char name[40];
    int nl = snprintf(name, sizeof(name), "Agent the %s",
        (ir >= 0 && ir < 13) ? rolenm[ir] : "Puffer");
    int fill = (int)((hp * nl + hpm - 1) / hpm);
    if (fill > nl) fill = nl;
    int hpc = hp * 3 >= hpm * 2 ? 32 : hp * 3 >= hpm ? 33 : 31;
    // both status rows are clipped to the box width: a long spell list or
    // several conditions used to wrap onto the next terminal line
    DemoLine L = {"│", 3, 1};
    char esc[16], tmp[160];
    snprintf(esc, sizeof(esc), "\x1b[7;%dm", hpc);
    demo_line_add(&L, "", " [");
    snprintf(tmp, sizeof(tmp), "%.*s", fill, name);
    demo_line_add(&L, esc, tmp);
    snprintf(tmp, sizeof(tmp), "%s] ", name + fill);
    demo_line_add(&L, "", tmp);
    snprintf(tmp, sizeof(tmp), "%s %s %s",
             (ia >= 0 && ia < 3) ? alignnm[ia] : "?", ig == 1 ? "female" : "male",
             (ic >= 0 && ic < 5) ? racenm[ic] : "?");
    demo_line_add(&L, "\x1b[2m", tmp);
    snprintf(tmp, sizeof(tmp), " St:%ld Dx:%ld Co:%ld  Score:%ld",
             bl[NLE_BL_STR25], bl[NLE_BL_DEX], bl[NLE_BL_CON], bl[NLE_BL_SCORE]);
    demo_line_add(&L, "", tmp);
    demo_line_flush(&L);
    demo_inv_row(k++, last, pn, plines, pclrs);
    static const char* conds[10] = {"Stone", "Slime", "Strngl", "FoodPois",
        "TermIll", "Blind", "Deaf", "Stun", "Conf", "Hallu"};
    static const char* hungers[5] = {"Satiated", "", "Hungry", "Weak", "Fainting"};
    long hu = bl[NLE_BL_HUNGER];
    L = (DemoLine){"│", 3, 1};
    snprintf(tmp, sizeof(tmp), " Dlvl:%ld $:%ld HP:%ld(%ld) Pw:%ld(%ld) AC:%ld Xp:%ld/%ld T:%ld",
             bl[NLE_BL_DEPTH], bl[NLE_BL_GOLD], hp, bl[NLE_BL_HPMAX],
             bl[NLE_BL_ENE], bl[NLE_BL_ENEMAX], bl[NLE_BL_AC],
             bl[NLE_BL_XP], bl[NLE_BL_EXP], bl[NLE_BL_TIME]);
    demo_line_add(&L, "", tmp);
    if (hu >= 0 && hu < 5 && hungers[hu][0]) {
        demo_line_add(&L, "", " ");
        demo_line_add(&L, "\x1b[33m", hungers[hu]);
    }
    for (int b = 0; b < 10; b++) {
        if (!(bl[NLE_BL_CONDITION] & (1L << b))) continue;
        demo_line_add(&L, "", " ");
        demo_line_add(&L, "\x1b[31;1m", conds[b]);
    }
    // known spells: name Lv fail% (env->spell_* is refreshed each pack_obs);
    // shown only when the whole list fits after the conditions
    if (env->n_spells > 0) {
        static const struct { int id; const char* nm; } spnames[] = {
            {344, "sleep"}, {348, "healing"}, {377, "protection"},
            {340, "force bolt"}, {342, "magic missile"}, {361, "cure blindness"},
        };
        int n = snprintf(tmp, sizeof(tmp), "Sp:");
        for (int i = 0; i < env->n_spells && i < 2; i++) {
            const char* nm = NULL;
            for (unsigned s = 0; s < sizeof(spnames)/sizeof(spnames[0]); s++) {
                if (spnames[s].id != env->spell_ids[i]) continue;
                nm = spnames[s].nm;
                break;
            }
            n += snprintf(tmp + n, sizeof(tmp) - n, "%s%s(L%d %d%%)", i ? "," : "",
                          nm ? nm : "spell", env->spell_levs[i], env->spell_fails[i]);
        }
        if (L.vis + 2 + n <= NH_COLS + 1) {
            demo_line_add(&L, "", "  ");
            demo_line_add(&L, "\x1b[36m", tmp);
        }
    }
    demo_line_flush(&L);
    demo_inv_row(k++, last, pn, plines, pclrs);
    demo_box_edge("└", "┘", NULL, NH_COLS);
    demo_inv_row(k++, last, pn, plines, pclrs);
    if (rate_hz > 0) // auto-run (web): no interactive keys to advertise
        printf("\x1b[2mstep %ld  |  running %d Hz\x1b[0m\n", steps, rate_hz);
    else
        printf("\x1b[2msteps %ld  |  SPACE step/hold 5Hz  |  Shift+SPACE (or S) 20Hz  |  q quit\x1b[0m\n",
               steps);
    fflush(stdout);
}

// Interactive: press space = 1 action; hold = 5/s; shift+space (or S) = 20/s.
static void run_demo_interactive(long max_steps) {
    const char* wpath = demo_find_weights();
    fprintf(stderr, "nethack demo: weights=%s\n", wpath);
    Weights* w = load_weights((char*)wpath);
    if (!w) {
        fprintf(stderr, "nethack demo: %s missing (set NH_WEIGHTS=path/to.bin)\n", wpath);
        exit(1);
    }
    NethackNet* net = make_nethack_net(w);

    Nethack env;
    nethack_color_sink = demo_colors;
    nethack_invstr_sink = demo_inv_strs;
    const char* seed_env = getenv("NH_SEED");
    srand(seed_env ? (unsigned)strtoul(seed_env, NULL, 10) : (unsigned)time(NULL));
    env_open(&env);

    demo_raw_term();

    float ep_score = 0, ep_len = 0, ep_depth = 0, ep_xp = 0, ep_gt = 0;
    float acts_f[DEMO_NUM_HEADS];
    long steps = 0;

    // Hold model (TTY has no key-up; OS auto-repeat confirms a hold):
    //   first SPACE/S  -> exactly one step
    //   further events -> continuous 5 Hz (space) or 20 Hz (shift+space / S)
    // Grace after first press covers the typical OS key-repeat delay (~0.5 s).
    int mode = 0; // 0=idle, 1=slow, 2=fast
    int confirmed_hold = 0; // saw a second key event (auto-repeat)
    int edge_pending = 0;
    double held_until = 0;
    double next_step_at = 0;

    demo_render(&env, 0, steps);

    while (steps < max_steps) {
        int in = demo_poll_input();
        if (in & DEMO_IN_QUIT) break;

        double now = demo_now();
        int want = 0;
        if (in & DEMO_IN_FAST) want = 2;
        else if (in & DEMO_IN_SPACE) want = 1;

        if (want) {
            if (mode != want) {
                // new press (or speed change): one immediate step, wait for
                // auto-repeat before continuous advance
                mode = want;
                confirmed_hold = 0;
                edge_pending = 1;
                next_step_at = now;
                held_until = now + 0.55;
            } else {
                // same mode again => key is being held
                confirmed_hold = 1;
                held_until = now + 0.12;
            }
        } else if (mode && now > held_until) {
            mode = 0;
            confirmed_hold = 0;
            edge_pending = 0;
        }

        int do_step = 0;
        int rate = mode == 2 ? 20 : (mode == 1 ? 5 : 0);
        if (mode && edge_pending) {
            do_step = 1;
            edge_pending = 0;
            next_step_at = now + 1.0 / (double)rate;
        } else if (mode && confirmed_hold && now >= next_step_at) {
            do_step = 1;
            next_step_at = now + 1.0 / (double)rate;
        }

        if (do_step) {
            demo_step_once(net, &env, acts_f, &ep_score, &ep_len,
                           &ep_depth, &ep_xp, &ep_gt);
            steps++;
            demo_render(&env, confirmed_hold ? rate : 0, steps);
        } else {
            usleep(5000);
        }
    }

    demo_restore_term();
    if (env.log.n > 0)
        printf("episodes=%.0f  avg_score=%.1f  avg_max_depth=%.2f  avg_xp=%.2f  steps=%ld\n",
               env.log.n, env.log.score / env.log.n,
               env.log.max_depth / env.log.n, env.log.max_xp_level / env.log.n, steps);
    else
        printf("steps=%ld\n", steps);
    env_close(&env);
    free_mingru(net->mingru);
    free(net->md);
    free(net->hidden);
    free(net->e_eff);
    free(net);
    free(w);
}

// Auto-run (headless or fixed frame delay) — used by scripts / profiling.
static void run_demo_auto(long max_steps, int frame_ms) {
    const char* wpath = demo_find_weights();
    fprintf(stderr, "nethack demo: weights=%s\n", wpath);
    Weights* w = load_weights((char*)wpath);
    if (!w) {
        fprintf(stderr, "nethack demo: %s missing (set NH_WEIGHTS=path/to.bin)\n", wpath);
        exit(1);
    }
    NethackNet* net = make_nethack_net(w);

    Nethack env;
    // frame mode renders the same composite as interactive
    if (frame_ms > 0) {
        nethack_color_sink = demo_colors;
        nethack_invstr_sink = demo_inv_strs;
    }
    const char* seed_env = getenv("NH_SEED");
    srand(seed_env ? (unsigned)strtoul(seed_env, NULL, 10) : (unsigned)time(NULL));
    env_open(&env);

    float ep_score = 0, ep_len = 0, ep_depth = 0, ep_xp = 0, ep_gt = 0;
    float acts_f[DEMO_NUM_HEADS];
    // NH_TRACE=1: print a line on every floor change (route analysis)
    int trace = getenv("NH_TRACE") != NULL;
    // NH_SKIP=N: run the first N steps unrendered, then play at frame_ms
    long skip = getenv("NH_SKIP") ? atol(getenv("NH_SKIP")) : 0;
    long pf = -1;
    for (long t = 0; t < max_steps; t++) {
#ifdef __EMSCRIPTEN__
        // web: render the first playable frame (step 0, or the NH_SKIP point)
        if (frame_ms > 0 && t == skip) {
            demo_render(&env, 1000 / frame_ms, t);
        }
#endif
        demo_step_once(net, &env, acts_f, &ep_score, &ep_len,
                       &ep_depth, &ep_xp, &ep_gt);
        if (trace) {
            if ((int)acts_f[0] == NETHACK_ACT_APPLY) {
                int sl = (int)acts_f[1 + 9];
                printf("TRACE APPLY g=%d\n", env.inv_glyphs[sl]);
            }
            long f = env.blstats[23] << 8 | env.blstats[24];
            if (f != pf) {
                printf("TRACE t=%ld s=%ld d=%ld:%ld hp=%ld xp=%ld\n",
                       env.blstats[NLE_BL_TIME], t, env.blstats[23],
                       env.blstats[24], env.blstats[10], env.blstats[18]);
                pf = f;
            }
            if (env.agents[0].terminals[0] != 0.0f) {
                printf("TRACE END\n");
                pf = -1;
            }
        }
        if (frame_ms > 0 && t >= skip) {
            demo_render(&env, 1000 / frame_ms, t);
#ifdef __EMSCRIPTEN__
            emscripten_sleep(frame_ms); // usleep busy-waits in wasm: yield or the page never paints
#else
            usleep(frame_ms * 1000);
#endif
        }
#ifdef __EMSCRIPTEN__
        else if (frame_ms > 0 && (t & 255) == 0) emscripten_sleep(0); // keep the tab responsive while skipping
#endif
    }
    if (env.log.n > 0)
        printf("episodes=%.0f  avg_score=%.1f  avg_max_depth=%.2f  avg_xp=%.2f\n",
               env.log.n, env.log.score / env.log.n,
               env.log.max_depth / env.log.n, env.log.max_xp_level / env.log.n);
    env_close(&env);
    free_mingru(net->mingru);
    free(net->md);
    free(net->hidden);
    free(net->e_eff);
    free(net);
    free(w);
}

// ./nethack                  interactive TTY (space / shift+space)
// ./nethack N 0              headless N steps
// ./nethack N MS             auto-run N steps at MS ms/frame
// NH_WEIGHTS=... NH_SEED=...
int main(int argc, char** argv) {
    long max_steps = (argc >= 2) ? atol(argv[1]) : 1000000;
    int interactive = isatty(STDIN_FILENO);
    int frame_ms = 50;
    if (argc >= 3) {
        frame_ms = atoi(argv[2]);
        interactive = 0; // explicit frame timing => auto mode
    } else if (!interactive) {
        frame_ms = 0; // piped/non-TTY default: headless auto-run
    }
    if (interactive) run_demo_interactive(max_steps);
    else run_demo_auto(max_steps, frame_ms);
    return 0;
}
