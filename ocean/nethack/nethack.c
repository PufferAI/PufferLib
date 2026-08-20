#include <time.h>
#include <unistd.h>
#include <string.h>
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
                      + 1 + 2 + 20)
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
#define DEMO_LOC_IN (NETHACK_CROP_GRID * DEMO_EMBED) // 9x9 crop, per-cell embeds
#define DEMO_LOC_HID 256
#define DEMO_PW 5
#define DEMO_PH 5
#define DEMO_PX 16
#define DEMO_PY 5
#define DEMO_TOK (DEMO_PX * DEMO_PY) // 5x5 patches over 79x21
#define DEMO_PCELLS (DEMO_PW * DEMO_PH) // off-map cells read the pad glyph
#define DEMO_P1 16
#define DEMO_GLB_IN (DEMO_PCELLS * DEMO_EMBED) // per-patch flatten (glyph slice)
#define DEMO_GLB_HID 128
// trigram message branch, mirroring NH_MSG_* in ocean/nethack/nethack.cu
#define DEMO_MSG_LEN NETHACK_MSG_LEN
#define DEMO_MSG_VOCAB 4096
#define DEMO_MSG_LOG2V 12
#define DEMO_MSG_HID 32
#define DEMO_MSG_CONCAT_OFF (DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL + 64 + DEMO_BL_FEAT)
#define DEMO_SPELL_CONCAT_OFF (DEMO_MSG_CONCAT_OFF + DEMO_MSG_HID)
// identity-table channel (NH_ID_EMBED in nethack.cu); presence is inferred
// per-checkpoint so one binary loads both eras
#define DEMO_IDE_ROLE 16
#define DEMO_IDE_RACE 8
#define DEMO_IDE_GEND 8
#define DEMO_IDE_ALGN 8
#define DEMO_IDE_DIM (DEMO_IDE_ROLE + DEMO_IDE_RACE + DEMO_IDE_GEND + DEMO_IDE_ALGN)
#define DEMO_IDE_NUMEL (13*DEMO_IDE_ROLE + 5*DEMO_IDE_RACE + 2*DEMO_IDE_GEND + 3*DEMO_IDE_ALGN)
#define DEMO_IDE_CONCAT_OFF (DEMO_SPELL_CONCAT_OFF + DEMO_SPKEY)
#define DEMO_CONCAT (DEMO_IDE_CONCAT_OFF + DEMO_IDE_DIM) // buffer max; live dim = net->concat_dim

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
    float *loc_w, *loc_b; // (256, 2592), (256)
    float *g1_w, *g1_xy, *g1_b; // (16, 800), (16, 2), (16): per-patch embed+flatten + hero dx,dy -> 16
    float *g2_w, *g2_b; // (128, 16), (128): 16 -> 128, maxed over tokens
    float *inv1_w, *inv1_b; // (16, 32), (16): per-slot features (pointer keys)
    float *inv1s_w; // (16, 24): gated item-state path into the slot MLP
    float *invt_w; // (16, 32): discovered-type channel (zero-init grown)
    float *inv2_w, *inv2_b; // (128, 16), (128): pooled trunk summary (max over slots)
    float *bl_w, *bl_b; // (64, DEMO_BL_FEAT), (64)
    float *proj_w, *proj_b; // (H, DEMO_CONCAT), (H)
    float *msg_w; // (4096, 32) trigram embedding table
    float *spk_w; // (16, 36) spell slot-rep projection
    float *spk2_w, *spk2_b; // (16, 16), (16) spell pool (inv2 idiom)
    float *ide_role_w, *ide_race_w, *ide_gend_w, *ide_algn_w; // identity tables (ide era)
    float *dec_lin; // (DEMO_DEC_PAD, H) bias-free; rows [26 verb | 48 dir | value], 75 used
    float *dec_q; // (DEMO_QDIM, H): thirteen stacked 16-dim queries (12 item + spell)
    float *dec_k; // (16, 16): key projection over slot features
    float *dec_tau; // (12,): per-head log cosine temperature
    MinGRU* mingru;
    Multidiscrete* md;
    int hidden_size, num_layers, num_actions;
    int ide, concat_dim; // per-checkpoint layout (identity-table era or not)
    float x[DEMO_LOC_IN]; // crop cell embeds, flattened
    float px[DEMO_GLB_IN]; // one patch's cell embeds, flattened
    float t16[DEMO_P1];
    float t128[DEMO_GLB_HID];
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
                        + DEMO_LOC_HID*DEMO_LOC_IN + DEMO_LOC_HID \
                        + DEMO_P1*DEMO_GLB_IN + DEMO_P1*2 + DEMO_P1 \
                        + DEMO_GLB_HID*DEMO_P1 + DEMO_GLB_HID \
                        + DEMO_INV_HID*DEMO_EMBED + DEMO_INV_HID \
                        + DEMO_INV_HID*DEMO_SFEAT \
                        + DEMO_INV_HID*DEMO_EMBED \
                        + DEMO_INV_POOL*DEMO_INV_HID + DEMO_INV_POOL \
                        + 64*DEMO_BL_FEAT + 64 \
                        + DEMO_MSG_VOCAB*DEMO_MSG_HID \
                        + DEMO_SPKEY*DEMO_SPIN + DEMO_SPKEY*DEMO_SPKEY + DEMO_SPKEY)
#define DEMO_DEC_FIXED (DEMO_INV_HID*DEMO_INV_HID + 16) // k_w + tau padded 12->16
// ambiguities are possible; prefer the fewest layers (real configs have <= 8)
static int demo_infer_arch(int total, int* hidden, int* layers, int* actions, int* ide) {
    int best_l = 1 << 30;
    for (int e = 0; e <= 1; e++)
    for (int H = 8; H <= 4096; H += 8) {
        long rem = (long)total - DEMO_ENC_FIXED - DEMO_DEC_FIXED - (e ? DEMO_IDE_NUMEL : 0)
                 - (long)H * (DEMO_IDE_CONCAT_OFF + (e ? DEMO_IDE_DIM : 0)
                              + 1 + DEMO_DEC_PAD + DEMO_QDIM);
        long per_layer = 3L * H * H;
        if (rem <= 0) break;
        if (rem % per_layer) continue;
        long L = rem / per_layer;
        if (L >= 1 && L < best_l) {
            best_l = (int)L;
            *hidden = H;
            *layers = (int)L;
            *actions = NETHACK_NUM_ACTIONS;
            *ide = e;
        }
    }
    return best_l == 1 << 30 ? -1 : 0;
}

static NethackNet* make_nethack_net(Weights* w) {
    NethackNet* net = (NethackNet*)calloc(1, sizeof(NethackNet));
    if (demo_infer_arch(w->size - 7, &net->hidden_size, &net->num_layers,
                        &net->num_actions, &net->ide) != 0) {
        fprintf(stderr, "nethack demo: cannot infer arch from %d floats — "
                "checkpoint is not a nethack policy with %d actions?\n",
                w->size - 7, NETHACK_NUM_ACTIONS);
        exit(1);
    }
    net->concat_dim = DEMO_IDE_CONCAT_OFF + (net->ide ? DEMO_IDE_DIM : 0);
    fprintf(stderr, "nethack demo: hidden=%d layers=%d actions=%d ide=%d (%d floats)\n",
            net->hidden_size, net->num_layers, net->num_actions, net->ide, w->size - 7);
    net->hidden = (float*)calloc(net->hidden_size, sizeof(float));
    net->embed = get_weights_aligned(w, DEMO_VOCAB * DEMO_EMBED);
    net->ekind_w = get_weights_aligned(w, NH_GM_NKIND * DEMO_EMBED);
    net->esub_w = get_weights_aligned(w, NH_GM_NSUB * DEMO_EMBED);
    net->loc_w = get_weights_aligned(w, DEMO_LOC_HID * DEMO_LOC_IN);
    net->loc_b = get_weights_aligned(w, DEMO_LOC_HID);
    net->g1_w = get_weights_aligned(w, DEMO_P1 * DEMO_GLB_IN);
    net->g1_xy = get_weights_aligned(w, DEMO_P1 * 2);
    net->g1_b = get_weights_aligned(w, DEMO_P1);
    net->g2_w = get_weights_aligned(w, DEMO_GLB_HID * DEMO_P1);
    net->g2_b = get_weights_aligned(w, DEMO_GLB_HID);
    net->inv1_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_EMBED);
    net->inv1_b = get_weights_aligned(w, DEMO_INV_HID);
    net->inv1s_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_SFEAT);
    net->invt_w = get_weights_aligned(w, DEMO_INV_HID * DEMO_EMBED);
    net->inv2_w = get_weights_aligned(w, DEMO_INV_POOL * DEMO_INV_HID);
    net->inv2_b = get_weights_aligned(w, DEMO_INV_POOL);
    net->bl_w = get_weights_aligned(w, 64 * DEMO_BL_FEAT);
    net->bl_b = get_weights_aligned(w, 64);
    net->proj_w = get_weights_aligned(w, net->hidden_size * net->concat_dim);
    net->proj_b = get_weights_aligned(w, net->hidden_size);
    net->msg_w = get_weights_aligned(w, DEMO_MSG_VOCAB * DEMO_MSG_HID);
    net->spk_w = get_weights_aligned(w, DEMO_SPKEY * DEMO_SPIN);
    net->spk2_w = get_weights_aligned(w, DEMO_SPKEY * DEMO_SPKEY);
    net->spk2_b = get_weights_aligned(w, DEMO_SPKEY);
    if (net->ide) {
        net->ide_role_w = get_weights_aligned(w, 13 * DEMO_IDE_ROLE);
        net->ide_race_w = get_weights_aligned(w, 5 * DEMO_IDE_RACE);
        net->ide_gend_w = get_weights_aligned(w, 2 * DEMO_IDE_GEND);
        net->ide_algn_w = get_weights_aligned(w, 3 * DEMO_IDE_ALGN);
    }
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

static int nethack_net_forward(NethackNet* net, const unsigned char* obs) { // fills decoder->output
    const int16_t* glyphs = (const int16_t*)(obs + NETHACK_OFF_GLYPHS);
    const unsigned char* bl = obs + NETHACK_OFF_BLSTATS;

    // local view: per-cell embeds of the egocentric crop, flattened
    int hx = demo_i32(bl), hy = demo_i32(bl + 4);
    int half = NETHACK_CROP / 2;
    for (int p = 0; p < NETHACK_CROP_GRID; p++) {
        int g = demo_glyph_at(glyphs, hy - half + p / NETHACK_CROP,
                              hx - half + p % NETHACK_CROP);
        memcpy(net->x + p * DEMO_EMBED, net->e_eff + g * DEMO_EMBED,
               DEMO_EMBED * sizeof(float));
    }
    _linear(net->x, net->loc_w, net->loc_b, net->concat, 1, DEMO_LOC_IN, DEMO_LOC_HID);
    _relu(net->concat, net->concat, DEMO_LOC_HID);

    // global view: per patch embed+flatten + normalized hero (dx,dy) -> 16 ->
    // 128, elementwise max over the 80 tokens (off-map cells of ragged edge
    // patches read the pad glyph)
    float* glb = net->concat + DEMO_LOC_HID;
    for (int o = 0; o < DEMO_GLB_HID; o++) glb[o] = -1e30f;
    for (int tk = 0; tk < DEMO_TOK; tk++) {
        int r0 = (tk / DEMO_PX) * DEMO_PH, c0 = (tk % DEMO_PX) * DEMO_PW;
        for (int pos = 0; pos < DEMO_PCELLS; pos++) {
            int g = demo_glyph_at(glyphs, r0 + pos / DEMO_PW, c0 + pos % DEMO_PW);
            memcpy(net->px + pos * DEMO_EMBED, net->e_eff + g * DEMO_EMBED,
                   DEMO_EMBED * sizeof(float));
        }
        float dx = (c0 + 0.5f * (DEMO_PW - 1) - hx) / (float)NH_COLS;
        float dy = (r0 + 0.5f * (DEMO_PH - 1) - hy) / (float)NH_ROWS;
        _linear(net->px, net->g1_w, net->g1_b, net->t16, 1, DEMO_GLB_IN, DEMO_P1);
        for (int k = 0; k < DEMO_P1; k++) {
            net->t16[k] += dx * net->g1_xy[k * 2] + dy * net->g1_xy[k * 2 + 1];
            if (net->t16[k] < 0.f) net->t16[k] = 0.f;
        }
        _linear(net->t16, net->g2_w, net->g2_b, net->t128, 1, DEMO_P1, DEMO_GLB_HID);
        for (int o = 0; o < DEMO_GLB_HID; o++)
            if (net->t128[o] > glb[o]) glb[o] = net->t128[o];
    }
    _relu(glb, glb, DEMO_GLB_HID);

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
        float sf[DEMO_SFEAT];
        for (int c = 0; c < 4; c++) sf[c] = st[0] == c ? 1.0f : 0.0f;
        int spe_known = st[1] != -128;
        sf[4] = (float)spe_known;
        sf[5] = spe_known ? (float)st[1] * 0.1f : 0.0f;
        sf[6] = log1pf(fmaxf((float)st[2], 0.0f)) * 0.5f;
        sf[7] = (float)st[3] * (1.0f / 3.0f);
        sf[8] = (float)st[4] * (1.0f / 3.0f);
        for (int c = 0; c < 7; c++) sf[9 + c] = (float)((st[5] >> c) & 1);
        sf[16] = (float)st[6];
        int ot = inv[slot] - NH_GLYPH_OBJ_OFF; // armor slot category one-hot
        int cat = (ot >= 0 && ot < NH_NUM_OBJECTS) ? nh_obj_armcat[ot] : -1;
        for (int c = 0; c < 7; c++) sf[17 + c] = cat == c ? 1.0f : 0.0f;
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
    float* invp = net->concat + DEMO_LOC_HID + DEMO_GLB_HID;
    for (int o = 0; o < DEMO_INV_POOL; o++) {
        float best = -1e30f;
        for (int slot = 0; slot < NETHACK_INV_SLOTS; slot++) {
            float v = 0.0f;
            for (int k = 0; k < DEMO_INV_HID; k++)
                v += net->inv2_w[o * DEMO_INV_HID + k] * net->slots[slot * DEMO_INV_HID + k];
            if (v > best) best = v;
        }
        invp[o] = fmaxf(best + net->inv2_b[o], 0.0f);
    }

    // blstats+extra features (25 scalars, hunger 7, cond bits 13, prev verb
    // one-hot, inv class counts, hp/ene frac, dnum one-hot, engraving bits)
    float* f = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL + 64;
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
    for (int k = 0; k < 20; k++) // role/race/gender one-hots; zeroed in the ide era
        f[j++] = net->ide ? 0.f : (float)demo_i32(ex + 4*(NETHACK_EXTRA_ROLEOH + k));
    for (int k = 0; k < DEMO_BL_FEAT; k++) f[k] = fminf(fmaxf(f[k], -1.f), 1.f);

    float* blout = net->concat + DEMO_LOC_HID + DEMO_GLB_HID + DEMO_INV_POOL;
    _linear(f, net->bl_w, net->bl_b, blout, 1, DEMO_BL_FEAT, 64);
    _relu(blout, blout, 64);

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
      for (int d = 0; d < DEMO_SPKEY; d++) {
          float best = -1e30f;
          for (int s = 0; s < NETHACK_SPELL_SLOTS; s++) {
              float acc = 0.f;
              for (int k = 0; k < DEMO_SPKEY; k++)
                  acc += net->spk2_w[d * DEMO_SPKEY + k] * net->spkeys[s * DEMO_SPKEY + k];
              if (acc > best) best = acc;
          }
          float v = best + net->spk2_b[d];
          sp[d] = v > 0.f ? v : 0.f;
      }
    }
    if (net->ide) { // identity table tail; mirrors nh_idemb_kernel (last set bit wins)
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
    _linear(net->concat, net->proj_w, net->proj_b, net->hidden, 1, net->concat_dim, net->hidden_size);
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
        kn[i] = sqrtf(nk) + 1e-6f;
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
    multidiscrete(net->md, net->logits, acts_f, 0, NULL);
    for (int h = 0; h < DEMO_NUM_HEADS; h++) env->agents[0].actions[h] = acts_f[h];
    puf_step(env);
    if (env->agents[0].terminals[0] > 0.5f) {
        float d = env->log.max_depth - *ep_depth;
        float x = env->log.max_xp_level - *ep_xp;
        float g = env->log.game_time - *ep_gt;
        fprintf(stderr, "episode end: score=%.0f len=%.0f max_depth=%.0f xp=%.0f game_t=%.0f "
                "eats=%.1f floor_eats=%.1f wears=%.1f throws=%.1f\n",
                env->log.score - *ep_score, env->log.episode_length - *ep_len, d, x, g,
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
    int len = printf("│ [\x1b[7;%dm%.*s\x1b[0m%s] \x1b[2m%s %s %s\x1b[0m"
           " St:%ld Dx:%ld Co:%ld  Score:%ld",
           hpc, fill, name, name + fill,
           (ia >= 0 && ia < 3) ? alignnm[ia] : "?", ig == 1 ? "female" : "male",
           (ic >= 0 && ic < 5) ? racenm[ic] : "?",
           bl[NLE_BL_STR25], bl[NLE_BL_DEX], bl[NLE_BL_CON], bl[NLE_BL_SCORE]) - 23;
    if (len < NH_COLS - 1) printf("%*s", NH_COLS - 1 - len, "");
    printf("│");
    demo_inv_row(k++, last, pn, plines, pclrs);
    static const char* conds[10] = {"Stone", "Slime", "Strngl", "FoodPois",
        "TermIll", "Blind", "Deaf", "Stun", "Conf", "Hallu"};
    static const char* hungers[5] = {"Satiated", "", "Hungry", "Weak", "Fainting"};
    long hu = bl[NLE_BL_HUNGER];
    len = printf("│ Dlvl:%ld $:%ld HP:%ld(%ld) Pw:%ld(%ld) AC:%ld Xp:%ld/%ld T:%ld",
           bl[NLE_BL_DEPTH], bl[NLE_BL_GOLD], hp, bl[NLE_BL_HPMAX],
           bl[NLE_BL_ENE], bl[NLE_BL_ENEMAX], bl[NLE_BL_AC],
           bl[NLE_BL_XP], bl[NLE_BL_EXP], bl[NLE_BL_TIME]) - 4;
    if (hu >= 0 && hu < 5 && hungers[hu][0]) len += printf(" \x1b[33m%s\x1b[0m", hungers[hu]) - 9;
    // known spells: name Lv fail%% (env->spell_* is refreshed each pack_obs)
    if (env->n_spells > 0) {
        static const struct { int id; const char* nm; } spnames[] = {
            {344, "sleep"}, {348, "healing"}, {377, "protection"},
            {340, "force bolt"}, {342, "magic missile"}, {361, "cure blindness"},
        };
        len += printf("  \x1b[36mSp:") - 5;
        for (int i = 0; i < env->n_spells && i < 2; i++) {
            const char* nm = NULL;
            for (unsigned s = 0; s < sizeof(spnames)/sizeof(spnames[0]); s++) {
                if (spnames[s].id != env->spell_ids[i]) continue;
                nm = spnames[s].nm;
                break;
            }
            len += printf("%s%s(L%d %d%%)", i ? "," : "",
                   nm ? nm : "spell", env->spell_levs[i], env->spell_fails[i]);
        }
        printf("\x1b[0m");
    }
    for (int b = 0; b < 10; b++)
        if (bl[NLE_BL_CONDITION] & (1L << b))
            len += printf(" \x1b[31;1m%s\x1b[0m", conds[b]) - 11;
    if (len < NH_COLS - 1) printf("%*s", NH_COLS - 1 - len, "");
    printf("│");
    demo_inv_row(k++, last, pn, plines, pclrs);
    demo_box_edge("└", "┘", NULL, NH_COLS);
    demo_inv_row(k++, last, pn, plines, pclrs);
    printf("\x1b[2msteps %ld  |  SPACE step/hold 5Hz  |  Shift+SPACE (or S) 20Hz  |  q quit",
           steps);
    if (rate_hz > 0) printf("  |  running %d Hz", rate_hz);
    printf("\x1b[0m\n");
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
    long pf = -1;
    for (long t = 0; t < max_steps; t++) {
        demo_step_once(net, &env, acts_f, &ep_score, &ep_len,
                       &ep_depth, &ep_xp, &ep_gt);
        if (trace) {
            if ((int)acts_f[0] == NETHACK_ACT_APPLY) {
                int sl = (int)acts_f[1 + 9];
                printf("TRACE APPLY g=%d\n", env.inv_glyphs[sl]);
            }
            long f = env.blstats[23] << 8 | env.blstats[24];
            if (f != pf) {
                printf("TRACE t=%ld d=%ld:%ld hp=%ld xp=%ld\n",
                       env.blstats[NLE_BL_TIME], env.blstats[23],
                       env.blstats[24], env.blstats[10], env.blstats[18]);
                pf = f;
            }
            if (env.agents[0].terminals[0] != 0.0f) {
                printf("TRACE END\n");
                pf = -1;
            }
        }
        if (frame_ms > 0) {
            demo_render(&env, 1000 / frame_ms, t);
            usleep(frame_ms * 1000);
        }
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
