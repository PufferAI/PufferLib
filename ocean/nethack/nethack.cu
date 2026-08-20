// NetHack CUDA encoder: one shared glyph embedding feeding two views of the
// full 79x21 map — an egocentric 9x9 crop at per-cell detail (flatten-linear)
// and a global 5x5-patch view (fused embed+flatten->16->128 max over 16x5 = 80 tokens) — plus
// the blstats MLP. Included by src/ocean.cu — requires trainer kernels/models.
// Bit-deterministic backward: scatter/bias sums via fixed-point integer
// atomics; GEMMs through the shared puf_mm cublas path.

__global__ void nh_bias_relu_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[idx % dim])));
}

// constants
// Obs layout (must match ocean/nethack/nethack.h):
//   [0, 2*NH_MGRID)  full 79x21 glyph grid, int16 LE (map memory included)
//   [2*NH_MGRID, +4*NH_BL_RAW)  blstats, int32 LE (x,y first)
//   [+4*NH_BL_RAW, +4*NH_EX_RAW)  extra stats, int32 LE: prayer cooldown,
//                                 previous action, 18 inventory class counts

static constexpr int NH_MAPW = 79, NH_MAPH = 21;
static constexpr int NH_MGRID = NH_MAPW * NH_MAPH;
static constexpr int NH_GLYPH_VOCAB = 5977; // MAX_GLYPH + 1 (NetHack 3.6.6)
static constexpr int NH_PAD_GLYPH = NH_GLYPH_VOCAB - 1; // NO_GLYPH: off-map crop cells
static constexpr int NH_EMBED_DIM = 32;
static constexpr int NH_CROP = 9, NH_CHALF = 4; // NETHACK_CROP, egocentric
static constexpr int NH_CGRID = NH_CROP * NH_CROP;
static constexpr int NH_PW = 5, NH_PH = 5; // patch size (cells)
static constexpr int NH_PX = 16, NH_PY = 5; // patch grid (ceil 79/5, 21/5)
static constexpr int NH_TOK = NH_PX * NH_PY; // 80 global tokens
static constexpr int NH_PCELLS = NH_PW * NH_PH; // cells per patch (off-map -> pad glyph)
static constexpr int NH_LOC_IN = NH_CGRID * NH_EMBED_DIM;
static constexpr int NH_LOC_HID = 256;
// Global branch: per patch, embed+flatten (25 cells x 32 dims) + normalized
// (dx,dy) patch-center offset from the hero -> 16 -> 128, then elementwise
// MAX over the 80 tokens. The 16-dim bottleneck keeps the fused per-glyph
// gather table (embed+flatten+first layer, NH_TROW cols) L2-resident, and
// fusing the max means the (B, 80, 128) token activations never exist in
// memory. The (dx,dy) slice lives in its own 16x2 weight tensor — same math
// as concatenating onto the flatten.
static constexpr int NH_P1 = 16;
static constexpr int NH_GLB_HID = 128;
static constexpr int NH_TROW = NH_PCELLS * NH_P1; // fused-table row: per-pos 16-dim
static constexpr int NH_PAD_PER_SAMPLE = NH_TOK * NH_PCELLS - NH_MGRID;
static constexpr int NH_HOT_G = 10; // hot-glyph dT smem slots (10x400 int64 = 32KB)
static constexpr int NH_BL_RAW = 27; // NLE_BLSTATS_SIZE
static constexpr int NH_BL_HUNGER = 21, NH_BL_CONDITION = 25;
static constexpr int NH_BL_HP = 10, NH_BL_ENE = 14; // hp/hpmax at 10/11, ene/enemax at 14/15
static constexpr int NH_ACTIONS = 26; // NETHACK_NUM_ACTIONS
static constexpr int NH_OCLASSES = 18; // MAXOCLASSES
static constexpr int NH_EXTRA_SHOP = 2 + NH_OCLASSES; // extra[] index of the shop pair
static constexpr int NH_SPELL_SLOTS = 8; // NETHACK_SPELL_SLOTS
static constexpr int NH_EX_ROLEOH = 2 + NH_OCLASSES + 2 + 1 + 4 * NH_SPELL_SLOTS + 2;
static constexpr int NH_EX_RAW = NH_EX_ROLEOH + 13 + 5 + 2; // NETHACK_EXTRA_INTS (+role/race/gender one-hots)
// blstats feature map (cumulative offsets; each block documented at its
// kernel branch). hp/ene fracs are the danger ratios the linear bl_w can't
// synthesize from separate cur/max scalars; dnum is one-hot because dungeon
// branch is nominal, not ordinal.
static constexpr int NH_F_HUNGER = 25; // 7-way one-hot
static constexpr int NH_F_COND = NH_F_HUNGER + 7; // 13 condition bits
static constexpr int NH_F_PREV = NH_F_COND + 13; // prev-action one-hot
static constexpr int NH_F_INV = NH_F_PREV + NH_ACTIONS; // inv class counts
static constexpr int NH_F_FRAC = NH_F_INV + NH_OCLASSES; // hp_frac, ene_frac
static constexpr int NH_F_DNUM = NH_F_FRAC + 2; // 8-way one-hot
static constexpr int NH_F_ENGR = NH_F_DNUM + 8; // engraving bits
static constexpr int NH_F_SHOP = NH_F_ENGR + 2; // in-shop, affordability
// spell scalar feature: known count only — per-slot content rides the
// spell-key path (v3 pointer), not the blstats block
static constexpr int NH_F_SPELL = NH_F_SHOP + 2;
// encumbrance ratio (softsign around the wall) + carry capacity /1000
static constexpr int NH_F_WEIGHT = NH_F_SPELL + 1;
static constexpr int NH_F_ROLE = NH_F_WEIGHT + 2; // 13 role + 5 race + 2 gender
static constexpr int NH_BL_FEAT = NH_F_ROLE + 20;
static constexpr int NH_BL_DNUM = 23;
static constexpr int NH_BL_HID = 64;
// Inventory entity branch: 55 slot glyphs, each embed -> shared 32->32
// linear -> relu. The per-slot vectors are the pointer decoder's keys (slot
// identity lives there); the trunk only gets the pooled summary below. Fused
// per-glyph table T_inv = E @ inv1_w^T (5977xNH_INV_HID) rebuilt per forward.
static constexpr int NH_INV = 55; // NETHACK_INV_SLOTS
// 16-dim per-slot rep: doubles as the pooled-summary bottleneck AND the
// decoder pointer key (unified, patch-encoder style). Halves the pool max
// MACs, the deterministic max-backward atomics, and the max-kernel smem
// footprint (better occupancy) vs the old 32; also shrinks the decoder.
static constexpr int NH_INV_HID = 16;
static constexpr int NH_INV_FLAT = NH_INV * NH_INV_HID;
// The trunk sees a pooled inventory summary, not the 1760-dim flatten: per
// slot 32 -> 128 with the elementwise max folded in (patch-encoder trick, the
// (B,55,128) intermediate never exists). Slot identity for the action heads
// comes from the pointer decoder's keys (inv_out), not the trunk.
static constexpr int NH_INV_POOL = 128;
// Trigram message branch: char-trigram bag over the raw topline. Each trigram
// hashes into NH_MSG_VOCAB buckets, its NH_MSG_HID-dim embed row summed, then
// scaled by 1/sqrt(count+1) (normalized bag / EmbeddingBag sum). The summary
// is concatenated raw (signed, no relu) like the blstats raw features.
static constexpr int NH_MSG_LEN = 128; // raw topline chars in obs tail
static constexpr int NH_MSG_VOCAB = 4096; // trigram hash buckets
static constexpr int NH_MSG_LOG2V = 12; // log2(NH_MSG_VOCAB)
static constexpr int NH_MSG_HID = 32; // trigram embed = message summary dim
static constexpr int NH_MSG_CONCAT_OFF = NH_LOC_HID + NH_GLB_HID + NH_INV_POOL + NH_BL_HID + NH_BL_FEAT;
// spell-key path (v3 pointer): per slot, key = spk_w . [e_eff(book glyph) |
// known, lev/7, fail/100, know/20000]; keys feed the CAST pointer head and a
// sum-pooled 16-dim trunk summary. Empty slots are exact zeros end to end.
static constexpr int NH_SPKEY = NH_INV_HID; // 16, shared key width
static constexpr int NH_SPIN = NH_EMBED_DIM + 4; // 36 key inputs/slot
static constexpr int NH_SPELL_CONCAT_OFF = NH_MSG_CONCAT_OFF + NH_MSG_HID;
// identity embeddings A/B arm: 1 = explicit tables in a direct concat channel
// (the one-hot bl features go dead); 0 = the committed one-hot representation
#ifndef NH_ID_EMBED
#define NH_ID_EMBED 1
#endif
static constexpr int NH_IDE_ROLE = 16, NH_IDE_RACE = 8;
static constexpr int NH_IDE_GEND = 8, NH_IDE_ALGN = 8;
static constexpr int NH_IDE_DIM = NH_IDE_ROLE + NH_IDE_RACE + NH_IDE_GEND + NH_IDE_ALGN;
static constexpr int NH_IDE_CONCAT_OFF = NH_SPELL_CONCAT_OFF + NH_SPKEY;
static constexpr int NH_CONCAT = NH_IDE_CONCAT_OFF + (NH_ID_EMBED ? NH_IDE_DIM : 0);
static constexpr int NH_BL_OFF = 2 * NH_MGRID; // blstats offset, obs elements
static constexpr int NH_INV_OFF = NH_BL_OFF + (NH_BL_RAW + NH_EX_RAW) * 4;
// obs v4: per-slot identification-gated state, 8 int8 fields per slot
// [buc, spe(-128=unknown), quan, ero1, ero2, flags, typeknown, rsvd],
// expanded to NH_SFEAT features feeding the slot MLP beside the embed
static constexpr int NH_INVST_OFF = NH_INV_OFF + NH_INV * 2;
static constexpr int NH_ST_RAW = 8; // NLE_INV_STATE_FIELDS
static constexpr int NH_SFEAT = 24; // buc4 + known+spe + quan + ero2 + flags7 + tk + armcat7
// discovered-type glyphs: true otyp glyph once dknown && oc_name_known, else pad
static constexpr int NH_INVTRUE_OFF = NH_INVST_OFF + NH_INV * NH_ST_RAW;
static constexpr int NH_MSG_OFF = NH_INVTRUE_OFF + NH_INV * 2; // message block start
static constexpr int NH_OBS_SIZE = NH_MSG_OFF + NH_MSG_LEN;
static constexpr int NH_SORT_BLOCKS = 256; // hist grid (smem histograms)
static constexpr int NH_HOT_T = 16; // hot-glyph smem rows (16x32 int64 = 4KB)

// Residual factorized glyph embedding: E_eff = E_res + E_kind[kind(g)] +
// E_sub[sub(g)]. The (kind, sub) mapping is generated from the engine's own
// display.h macros; sub shares one row per monster SPECIES across the seven
// monster-derived kinds (mon/pet/detect/corpse/ridden/swallow/statue), so
// rare forms inherit what common forms learn. Zero-init factors make the
// initial function identical to the unfactorized baseline.
#define NH_GM_QUAL static __device__ const
#include "glyph_map.h"
#undef NH_GM_QUAL

// armor slot per otyp (ARM_SUIT=0..ARM_SHIRT=6, -1 not armor), device copy of
// nh_obj_armcat in ocean/nethack/netlib.h (NetHack 3.6.6)
static __device__ const signed char nh_obj_armcat_dev[NH_NUM_OBJECTS] = {
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
static_assert(NH_GM_VOCAB == 5977, "glyph map vocab mismatch");
static constexpr int NH_NKIND = NH_GM_NKIND;
static constexpr int NH_NSUB = NH_GM_NSUB;

// 2^24 fixed-point gradient accumulators: integer atomics are associative, so
// scatter/bias sums are bit-identical run to run (float atomicAdd ordering is
// not). Quantization (6e-8) is below fp32 accumulation error at these counts.
static constexpr float NH_FXP = 16777216.0f;
__device__ __forceinline__ void nh_fxp_atomic_add(long long* addr, float v) {
    atomicAdd((unsigned long long*)addr, (unsigned long long)(long long)__float2ll_rn(v * NH_FXP));
}
__device__ __forceinline__ float nh_fxp_to_float(long long v) {
    return (float)((double)v * (1.0 / 16777216.0));
}
__global__ void nh_fxp_to_precision_kernel(
    precision_t* __restrict__ dst, const long long* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(nh_fxp_to_float(src[idx]));
}

// Row-sparse fixed-point -> precision cast for the embed-table grad: a
// minibatch touches only a few hundred of the 5977 rows (counts>0 or hot).
// Untouched rows skip the int64 read; touched rows are re-zeroed in place,
// replacing a full-table memset. Invariant: src is all-zero between
// iterations (alloc_create zeroes it once).
__global__ void nh_fxp_to_precision_rows_kernel(
    precision_t* __restrict__ dst, long long* __restrict__ src,
    const int* __restrict__ counts, const int* __restrict__ hot_map,
    int trow, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = idx / trow;
    if (counts[row] == 0 && hot_map[row] < 0) {
        dst[idx] = from_float(0.0f);
        return;
    }
    dst[idx] = from_float(nh_fxp_to_float(src[idx]));
    src[idx] = 0;
}

// Per-blstat normalization: log1p fields get log1p(max(v,0))*scale, the rest
// v*scale. Hunger (21) and condition (25) are expanded, not scaled.
__constant__ float NH_BL_SCALE[NH_BL_RAW] = {
    1.f/79, 1.f/21, // x, y
    1.f/25, 1.f/125, 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,  // str25 str125 dex con int wis cha
    0.1f, // score (log)
    1.f/200, 1.f/200, 1.f/50, // hp, hpmax, depth
    0.1f, // gold (log)
    1.f/100, 1.f/100, 1.f/10, 1.f/10, 1.f/30, // ene, enemax, ac, hd, xp level
    0.1f, 0.1f, // exp points, time (log)
    0.f, // hunger (expanded)
    1.f/4, 0.f, 1.f/50, // cap, dnum (one-hot), dlevel
    0.f, // condition (expanded)
    1.f, // align
};
__constant__ int NH_BL_ISLOG[NH_BL_RAW] = {
    0,0,0,0,0,0,0,0,0, 1, 0,0,0, 1, 0,0,0,0,0, 1,1, 0, 0,0,0, 0, 0,
};

// kernels

__device__ __forceinline__ int nh_bl_read_i32(const precision_t* p) {
    return (int)((unsigned int)(int)to_float(p[0])
               | ((unsigned int)(int)to_float(p[1]) << 8)
               | ((unsigned int)(int)to_float(p[2]) << 16)
               | ((unsigned int)(int)to_float(p[3]) << 24));
}

// Decode int16 LE glyph ids into an fp32 index buffer (full grid).
__global__ void nh_decode_kernel(
    float* __restrict__ idx, const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_MGRID) return;
    int b = t / NH_MGRID, cell = t % NH_MGRID;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + 2 * cell;
    int g = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    idx[t] = (float)max(0, min(g, NH_GLYPH_VOCAB - 1));
}

// Egocentric crop glyph ids: window centered on the hero (blstats x,y),
// off-map cells get the pad glyph.
__global__ void nh_crop_kernel(
    float* __restrict__ crop, const float* __restrict__ idx,
    const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_CGRID) return;
    int b = t / NH_CGRID, p = t % NH_CGRID;
    const precision_t* bl = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
    int r = nh_bl_read_i32(bl + 4) - NH_CHALF + p / NH_CROP; // blstats[1] = y
    int c = nh_bl_read_i32(bl)     - NH_CHALF + p % NH_CROP; // blstats[0] = x
    crop[t] = (r < 0 || r >= NH_MAPH || c < 0 || c >= NH_MAPW)
        ? (float)NH_PAD_GLYPH : idx[b * NH_MGRID + r * NH_MAPW + c];
}

// Local view: per-cell embedding gather, flattened (B, NH_CGRID*NH_EMBED_DIM).
__global__ void nh_local_gather_kernel(
    precision_t* __restrict__ x, const precision_t* __restrict__ E,
    const float* __restrict__ crop, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LOC_IN) return;
    int d = t % NH_EMBED_DIM;
    int64_t cell = t / NH_EMBED_DIM; // b*NH_CGRID + p
    x[t] = E[(int64_t)(int)crop[cell] * NH_EMBED_DIM + d];
}

// glb1.w (P1, PCELLS*D) -> W' (PCELLS*P1, D), so T = E @ W'^T lands as
// T[g, pos*P1+k]: the fused embed+flatten+layer1 lookup table, rebuilt with
// E_eff materialization: residual factorization over the observed glyph id.
// The observed id is post-shuffle (appearance space), so (kind, sub) cannot
// re-leak identities the shuffle hid.
__global__ void nh_eff_embed_kernel(precision_t* __restrict__ out,
    const precision_t* __restrict__ E, const precision_t* __restrict__ K,
    const precision_t* __restrict__ S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_GLYPH_VOCAB * NH_EMBED_DIM) return;
    int g = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    out[i] = from_float(to_float(E[i])
        + to_float(K[nh_glyph_kind[g] * NH_EMBED_DIM + d])
        + to_float(S[nh_glyph_sub[g] * NH_EMBED_DIM + d]));
}

// Factor grads: dE_eff/dE_res is identity (embed_wgrad IS dE_res); the factor
// rows are deterministic per-row serial sums over the generated CSR lists,
// launched after every dE_eff contribution has landed.
__global__ void nh_ekind_grad_kernel(precision_t* __restrict__ dK,
    const precision_t* __restrict__ dE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_NKIND * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    float acc = 0.0f;
    for (int j = nh_kind_csr_off[r]; j < nh_kind_csr_off[r + 1]; j++)
        acc += to_float(dE[(int64_t)nh_kind_csr_glyph[j] * NH_EMBED_DIM + d]);
    dK[i] = from_float(acc);
}
__global__ void nh_esub_grad_kernel(precision_t* __restrict__ dS,
    const precision_t* __restrict__ dE) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_NSUB * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    float acc = 0.0f;
    for (int j = nh_sub_csr_off[r]; j < nh_sub_csr_off[r + 1]; j++)
        acc += to_float(dE[(int64_t)nh_sub_csr_glyph[j] * NH_EMBED_DIM + d]);
    dS[i] = from_float(acc);
}

// one small GEMM whenever the weights change.
__global__ void nh_permute_g1_kernel(
    precision_t* __restrict__ wp, const precision_t* __restrict__ w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_TROW * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    int pos = r / NH_P1, k = r % NH_P1;
    wp[i] = w[k * (NH_PCELLS * NH_EMBED_DIM) + pos * NH_EMBED_DIM + d];
}

// Inverse for the weight grad.
__global__ void nh_unpermute_g1_kernel(
    precision_t* __restrict__ wg, const precision_t* __restrict__ wpg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_TROW * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    int pos = r / NH_P1, k = r % NH_P1;
    wg[k * (NH_PCELLS * NH_EMBED_DIM) + pos * NH_EMBED_DIM + d] = wpg[i];
}

// Fused global view, one block per sample: stage 1 builds the 80 relu'd
// 16-dim tokens in smem from the fused table (off-map positions read the pad
// glyph's row) plus the (dx,dy) hero-offset slice; stage 2 expands each token
// 16->128 in registers and folds the elementwise max on the fly (argmax saved
// for the sparse backward). Fixed token order + strict > keep argmax
// deterministic. dxy is saved for the w_xy weight grad.
__global__ void nh_patch_max_kernel(
    precision_t* __restrict__ glb_out, precision_t* __restrict__ t16_save,
    precision_t* __restrict__ dxy_save, int* __restrict__ argmax,
    const precision_t* __restrict__ T, const precision_t* __restrict__ b1,
    const precision_t* __restrict__ w_xy, const precision_t* __restrict__ w2,
    const precision_t* __restrict__ b2, const float* __restrict__ idx,
    const precision_t* __restrict__ obs, int B) {
    __shared__ float w2s[NH_GLB_HID * NH_P1];
    __shared__ float t16s[NH_TOK * NH_P1];
    __shared__ float hero[2];
    int b = blockIdx.x;
    if (b >= B) return;
    if (threadIdx.x == 0) {
        const precision_t* bl = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
        hero[0] = (float)nh_bl_read_i32(bl); // x
        hero[1] = (float)nh_bl_read_i32(bl + 4); // y
    }
    for (int i = threadIdx.x; i < NH_GLB_HID * NH_P1; i += blockDim.x)
        w2s[i] = to_float(w2[i]);
    __syncthreads();
    const float* gi = idx + (int64_t)b * NH_MGRID;
    for (int i = threadIdx.x; i < NH_TOK * NH_P1; i += blockDim.x) {
        int tk = i / NH_P1, k = i % NH_P1;
        int r0 = (tk / NH_PX) * NH_PH, c0 = (tk % NH_PX) * NH_PW;
        float dx = (c0 + 0.5f * (NH_PW - 1) - hero[0]) * (1.0f / NH_MAPW);
        float dy = (r0 + 0.5f * (NH_PH - 1) - hero[1]) * (1.0f / NH_MAPH);
        float acc = to_float(b1[k])
                  + dx * to_float(w_xy[k * 2]) + dy * to_float(w_xy[k * 2 + 1]);
        #pragma unroll
        for (int pos = 0; pos < NH_PCELLS; pos++) {
            int r = r0 + pos / NH_PW, c = c0 + pos % NH_PW;
            int g = (r < NH_MAPH && c < NH_MAPW) ? (int)gi[r * NH_MAPW + c] : NH_PAD_GLYPH;
            acc += to_float(T[(int64_t)g * NH_TROW + pos * NH_P1 + k]);
        }
        acc = fmaxf(acc, 0.0f);
        t16s[i] = acc;
        t16_save[(int64_t)b * (NH_TOK * NH_P1) + i] = from_float(acc);
        if (k < 2)
            dxy_save[((int64_t)b * NH_TOK + tk) * 2 + k] = from_float(k == 0 ? dx : dy);
    }
    __syncthreads();
    for (int o = threadIdx.x; o < NH_GLB_HID; o += blockDim.x) {
        float best = -1e30f;
        int bm = 0;
        for (int tk = 0; tk < NH_TOK; tk++) {
            float v = 0.0f;
            for (int k = 0; k < NH_P1; k++)
                v += w2s[o * NH_P1 + k] * t16s[tk * NH_P1 + k];
            if (v > best) {
                best = v;
                bm = tk;
            }
        }
        glb_out[(int64_t)b * NH_GLB_HID + o] = from_float(fmaxf(best + to_float(b2[o]), 0.0f));
        argmax[(int64_t)b * NH_GLB_HID + o] = bm;
    }
}

// Backward through max + layer 2, one block per sample. dglb is already
// relu-masked (and b2's grad accumulated) by nh_relu_bias_bwd. dW2 and dt16
// accumulate in fixed-point smem (deterministic), dt16 is relu-masked against
// the saved t16 and written back over it.
__global__ void nh_patch_max_bwd_kernel(
    precision_t* __restrict__ t16_io, long long* __restrict__ dw2_acc,
    const precision_t* __restrict__ dglb, const precision_t* __restrict__ w2,
    const int* __restrict__ argmax, int B) {
    __shared__ float w2s[NH_GLB_HID * NH_P1];
    __shared__ float t16s[NH_TOK * NH_P1];
    __shared__ long long dt16s[NH_TOK * NH_P1];
    __shared__ long long dw2s[NH_GLB_HID * NH_P1];
    int b = blockIdx.x;
    if (b >= B) return;
    for (int i = threadIdx.x; i < NH_GLB_HID * NH_P1; i += blockDim.x) {
        w2s[i] = to_float(w2[i]);
        dw2s[i] = 0;
    }
    for (int i = threadIdx.x; i < NH_TOK * NH_P1; i += blockDim.x) {
        t16s[i] = to_float(t16_io[(int64_t)b * (NH_TOK * NH_P1) + i]);
        dt16s[i] = 0;
    }
    __syncthreads();
    for (int o = threadIdx.x; o < NH_GLB_HID; o += blockDim.x) {
        float g = to_float(dglb[(int64_t)b * NH_GLB_HID + o]);
        if (g == 0.0f) continue;
        int m = argmax[(int64_t)b * NH_GLB_HID + o];
        for (int k = 0; k < NH_P1; k++) {
            float dt = g * w2s[o * NH_P1 + k];
            if (dt != 0.0f)
                atomicAdd((unsigned long long*)&dt16s[m * NH_P1 + k],
                          (unsigned long long)(long long)__float2ll_rn(dt * NH_FXP));
            float dw = g * t16s[m * NH_P1 + k];
            if (dw != 0.0f)
                atomicAdd((unsigned long long*)&dw2s[o * NH_P1 + k],
                          (unsigned long long)(long long)__float2ll_rn(dw * NH_FXP));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NH_GLB_HID * NH_P1; i += blockDim.x)
        if (dw2s[i] != 0)
            atomicAdd((unsigned long long*)&dw2_acc[i], (unsigned long long)dw2s[i]);
    for (int i = threadIdx.x; i < NH_TOK * NH_P1; i += blockDim.x) {
        float v = t16s[i] > 0.0f ? nh_fxp_to_float(dt16s[i]) : 0.0f;
        t16_io[(int64_t)b * (NH_TOK * NH_P1) + i] = from_float(v);
    }
}

// Decode int32 LE blstats + extra stats and expand to NH_BL_FEAT normalized
// features (block map = the NH_F_* offsets). Warp per sample: one serial
// thread per sample is a 200-op latency chain.
__global__ void nh_blstats_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs, int B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / 32, lane = tid % 32;
    if (b >= B) return;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
    const precision_t* ex = src + 4 * NH_BL_RAW;
    precision_t* dst = out + (int64_t)b * NH_BL_FEAT;
    for (int j = lane; j < NH_BL_FEAT; j += 32) {
        float f;
        if (j < NH_F_HUNGER) {
            // 25 scaled scalars: blstats minus hunger(21) and condition(25)
            int i = j + (j >= 21) + (j >= 24);
            int v = nh_bl_read_i32(src + 4*i);
            f = NH_BL_ISLOG[i] ? log1pf(fmaxf((float)v, 0.0f)) * NH_BL_SCALE[i]
                               : (float)v * NH_BL_SCALE[i];
        } else if (j < NH_F_COND) {
            int v = nh_bl_read_i32(src + 4*NH_BL_HUNGER);
            f = (j - NH_F_HUNGER == max(0, min(v, 6))) ? 1.0f : 0.0f;
        } else if (j < NH_F_PREV) {
            unsigned int cond = (unsigned int)nh_bl_read_i32(src + 4*NH_BL_CONDITION);
            f = (float)((cond >> (j - NH_F_COND)) & 1u);
        } else if (j < NH_F_INV) {
            f = (j - NH_F_PREV == nh_bl_read_i32(ex + 4)) ? 1.0f : 0.0f;
        } else if (j < NH_F_FRAC) {
            f = (float)nh_bl_read_i32(ex + 4*(2 + j - NH_F_INV)) * 0.125f;
        } else if (j < NH_F_DNUM) {
            // hp_frac / ene_frac in [0,1]: the "how close to death/empty" ratio
            int base = (j == NH_F_FRAC) ? NH_BL_HP : NH_BL_ENE;
            int cur = nh_bl_read_i32(src + 4*base);
            int mx = nh_bl_read_i32(src + 4*(base + 1));
            f = fminf(fmaxf((float)cur / (float)(mx > 1 ? mx : 1), 0.0f), 1.0f);
        } else if (j < NH_F_ENGR) {
            int v = nh_bl_read_i32(src + 4*NH_BL_DNUM);
            f = (j - NH_F_DNUM == max(0, min(v, 7))) ? 1.0f : 0.0f;
        } else if (j < NH_F_SHOP) {
            // underfoot engraving from ex[0]: any engraving, active Elbereth
            int v = nh_bl_read_i32(ex);
            f = (j == NH_F_ENGR) ? (v >= 1 ? 1.0f : 0.0f) : (v >= 2 ? 1.0f : 0.0f);
        } else if (j < NH_F_SPELL) {
            // standing on shop goods, and gold/price capped at 1
            int v = nh_bl_read_i32(ex + 4*(NH_EXTRA_SHOP + (j - NH_F_SHOP)));
            f = (j == NH_F_SHOP) ? (float)v : (float)v * 0.01f;
        } else if (j < NH_F_WEIGHT) {
            // known-spell count/8; per-slot content rides the spell-key path
            int v = nh_bl_read_i32(ex + 4*(NH_EXTRA_SHOP + 2));
            f = (float)v * 0.125f;
        } else if (j < NH_F_ROLE) {
            // encumbrance: softsign(ratio-1) is 0 at the wall, keeps gradient
            // through Overloaded (~3x); capacity /1000 (engine caps there)
            int v = nh_bl_read_i32(ex + 4*(NH_EXTRA_SHOP + 2 + 1 + 4 * NH_SPELL_SLOTS
                                           + (j - NH_F_WEIGHT)));
            if (j == NH_F_WEIGHT) {
                float d = (float)v * 0.01f - 1.0f;
                f = d / (1.0f + fabsf(d));
            } else
                f = (float)v * 0.001f;
        } else {
            // role/race/gender one-hots, already 0/1
            // dead under NH_ID_EMBED: identity flows via the embed channel
            f = NH_ID_EMBED ? 0.0f
              : (float)nh_bl_read_i32(ex + 4*(NH_EX_ROLEOH + (j - NH_F_ROLE)));
        }
        // strict [-1,1]: bounds deep-play excursions (AC -15 -> -1.5, hp 300 ->
        // 1.5, stacked inv counts) — validated neutral-now, deep-safe (n=4)
        dst[j] = from_float(fminf(fmaxf(f, -1.0f), 1.0f));
    }
}

// concat = [local hid | global hid | bl hid | bl raw feats]
// trigram message branch
__device__ __forceinline__ int nh_msg_lc(int c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c; // lowercase; keep spaces/punct
}
__device__ __forceinline__ int nh_msg_hash(int c0, int c1, int c2) {
    unsigned key = ((unsigned)c0 << 16) | ((unsigned)c1 << 8) | (unsigned)c2;
    return (int)((key * 2654435761u) >> (32 - NH_MSG_LOG2V)); // top log2V bits
}
// per-position trigram bucket id (-1 for the padded tail / past the null). Ids
// stay contiguous because the topline is null-terminated, so consumers break
// at the first -1.
__global__ void nh_msg_ids_kernel(
    float* __restrict__ ids, const precision_t* __restrict__ obs, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * NH_MSG_LEN) return;
    int b = i / NH_MSG_LEN, t = i % NH_MSG_LEN;
    if (t > NH_MSG_LEN - 3) {
        ids[i] = -1.0f;
        return;
    }
    const precision_t* m = obs + (int64_t)b * NH_OBS_SIZE + NH_MSG_OFF;
    int c0 = (int)to_float(m[t]), c1 = (int)to_float(m[t + 1]), c2 = (int)to_float(m[t + 2]);
    if (c0 == 0 || c1 == 0 || c2 == 0) {
        ids[i] = -1.0f;
        return;
    }
    ids[i] = (float)nh_msg_hash(nh_msg_lc(c0), nh_msg_lc(c1), nh_msg_lc(c2));
}
// normalized-sum bag: block per sample, one warp = NH_MSG_HID lanes (lane d
// owns output dim d). Each trigram is one coalesced read of msg_w[id]; sum
// scaled by 1/sqrt(count+1). No relu (raw signed summary).
__global__ void nh_msg_pool_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ msg_w,
    const float* __restrict__ ids, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int d = threadIdx.x;
    const float* mi = ids + (int64_t)b * NH_MSG_LEN;
    int count = 0; float acc = 0.0f;
    for (int t = 0; t < NH_MSG_LEN; t++) {
        int id = (int)mi[t];
        if (id < 0) break;
        count++;
        acc += to_float(msg_w[(int64_t)id * NH_MSG_HID + d]);
    }
    out[(int64_t)b * NH_MSG_HID + d] = from_float(acc * rsqrtf((float)count + 1.0f));
}
// backward: scatter scale*dout into the trigram-table grad (fixed-point,
// deterministic). Lane d writes bucket dim d, so the warp hits distinct
// addresses (no intra-block contention); cross-block collisions on hot
// buckets are bounded. dout read straight from the concat-grad slice.
__global__ void nh_msg_bwd_kernel(
    long long* __restrict__ dmsg_acc, const precision_t* __restrict__ grad_concat,
    const float* __restrict__ ids, int B) {
    int b = blockIdx.x;
    if (b >= B) return;
    int d = threadIdx.x;
    const float* mi = ids + (int64_t)b * NH_MSG_LEN;
    int count = 0;
    for (int t = 0; t < NH_MSG_LEN; t++) {
        if ((int)mi[t] < 0) break;
        count++;
    }
    float g = to_float(grad_concat[(int64_t)b * NH_CONCAT + NH_MSG_CONCAT_OFF + d])
              * rsqrtf((float)count + 1.0f);
    if (g == 0.0f) return;
    for (int t = 0; t < count; t++)
        nh_fxp_atomic_add(&dmsg_acc[(int64_t)(int)mi[t] * NH_MSG_HID + d], g);
}

// spell-key forward: per slot, build the 36-dim input [e_eff(book glyph) |
// known, lev/7, fail/100, know/20000], key = spk_w . input. Saves inputs +
// glyph ids for the backward; empty slot -> zero input -> zero key.
__global__ void nh_spkey_kernel(precision_t* __restrict__ keys,
    precision_t* __restrict__ sp_in, float* __restrict__ sp_idx,
    const precision_t* __restrict__ spk_w, const precision_t* __restrict__ e_eff,
    const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS) return;
    int b = t / NH_SPELL_SLOTS, s = t % NH_SPELL_SLOTS;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF
                           + 4 * (NH_BL_RAW + NH_EXTRA_SHOP + 2 + 1 + 4 * s);
    int id = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    int g = id > 0 ? min(id + 1906, NH_GLYPH_VOCAB - 1) : -1;
    sp_idx[(int64_t)t] = (float)g;
    float in[NH_SPIN];
    for (int d = 0; d < NH_EMBED_DIM; d++)
        in[d] = g >= 0 ? to_float(e_eff[(int64_t)g * NH_EMBED_DIM + d]) : 0.0f;
    int lev = (int)to_float(src[4]) | ((int)to_float(src[5]) << 8);
    int fail = (int)to_float(src[8]) | ((int)to_float(src[9]) << 8);
    int know = (int)to_float(src[12]) | ((int)to_float(src[13]) << 8)
             | ((int)to_float(src[14]) << 16);
    in[NH_EMBED_DIM + 0] = id > 0 ? 1.0f : 0.0f;
    in[NH_EMBED_DIM + 1] = fminf((float)lev * 0.142857f, 1.0f);
    in[NH_EMBED_DIM + 2] = fminf((float)fail * 0.01f, 1.0f);
    in[NH_EMBED_DIM + 3] = fminf((float)know * 0.00005f, 1.0f);
    precision_t* inb = sp_in + (int64_t)t * NH_SPIN;
    for (int c = 0; c < NH_SPIN; c++) inb[c] = from_float(in[c]);
    precision_t* kb = keys + (int64_t)t * NH_SPKEY;
    for (int r = 0; r < NH_SPKEY; r++) {
        float acc = 0.0f;
        for (int c = 0; c < NH_SPIN; c++)
            acc += to_float(spk_w[r * NH_SPIN + c]) * in[c];
        kb[r] = from_float(fmaxf(acc, 0.0f)); // relu'd slot rep (inv1 idiom)
    }
}

// inventory-style pool: project each relu'd slot rep, max over slots, bias,
// relu (mirrors inv2). Saves the argmax slot per (sample, pool dim).
__global__ void nh_sppool_kernel(precision_t* __restrict__ concat,
    precision_t* __restrict__ pool, int* __restrict__ amax,
    const precision_t* __restrict__ keys,
    const precision_t* __restrict__ spk2_w, const precision_t* __restrict__ spk2_b, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPKEY) return;
    int b = t / NH_SPKEY, d = t % NH_SPKEY;
    float best = -1e30f;
    int bs = 0;
    for (int s = 0; s < NH_SPELL_SLOTS; s++) {
        float acc = 0.0f;
        for (int k = 0; k < NH_SPKEY; k++)
            acc += to_float(spk2_w[d * NH_SPKEY + k])
                 * to_float(keys[((int64_t)b * NH_SPELL_SLOTS + s) * NH_SPKEY + k]);
        if (acc > best) {
            best = acc;
            bs = s;
        }
    }
    amax[t] = bs;
    float v = fmaxf(best + to_float(spk2_b[d]), 0.0f);
    pool[t] = from_float(v);
    concat[(int64_t)b * NH_CONCAT + NH_SPELL_CONCAT_OFF + d] = from_float(v);
}

// backward: scatter the concat-grad spell slice into the fxp dE staging
// spell-key backward, stage 1: total per-slot rep grad = pool grad routed
// through relu + argmax + spk2 projection, plus the pointer's key grads;
// finally masked by the rep's own relu.
__global__ void nh_spkey_dk_kernel(precision_t* __restrict__ dkeys,
    const precision_t* __restrict__ grad_concat, const precision_t* __restrict__ pool,
    const int* __restrict__ amax, const precision_t* __restrict__ spk2_w,
    const precision_t* __restrict__ keys, const precision_t* __restrict__ ptr_dkeys, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_SPKEY) return;
    int b = t / (NH_SPELL_SLOTS * NH_SPKEY);
    int s = (t / NH_SPKEY) % NH_SPELL_SLOTS;
    int k = t % NH_SPKEY;
    float v = ptr_dkeys ? to_float(ptr_dkeys[t]) : 0.0f;
    for (int d = 0; d < NH_SPKEY; d++) {
        if (amax[(int64_t)b * NH_SPKEY + d] != s) continue;
        if (to_float(pool[(int64_t)b * NH_SPKEY + d]) <= 0.0f)
            continue; // pool relu gate
        v += to_float(grad_concat[(int64_t)b * NH_CONCAT + NH_SPELL_CONCAT_OFF + d])
           * to_float(spk2_w[d * NH_SPKEY + k]);
    }
    if (to_float(keys[t]) <= 0.0f) v = 0.0f; // rep relu gate
    dkeys[t] = from_float(v);
}

// spk2 grads: (16x16 + 16) threads loop the batch; small enough to be cheap
__global__ void nh_spk2_grad_kernel(precision_t* __restrict__ spk2_wgrad,
    precision_t* __restrict__ spk2_bgrad, const precision_t* __restrict__ grad_concat,
    const precision_t* __restrict__ pool, const int* __restrict__ amax,
    const precision_t* __restrict__ keys, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= NH_SPKEY * (NH_SPKEY + 1)) return;
    int d = t / (NH_SPKEY + 1), k = t % (NH_SPKEY + 1);
    float acc = 0.0f;
    for (int b = 0; b < B; b++) {
        if (to_float(pool[(int64_t)b * NH_SPKEY + d]) <= 0.0f)
            continue;
        float g = to_float(grad_concat[(int64_t)b * NH_CONCAT + NH_SPELL_CONCAT_OFF + d]);
        if (g == 0.0f) continue;
        if (k == NH_SPKEY) { // bias
            acc += g;
            continue;
        }
        int s = amax[(int64_t)b * NH_SPKEY + d];
        acc += g * to_float(keys[((int64_t)b * NH_SPELL_SLOTS + s) * NH_SPKEY + k]);
    }
    if (k == NH_SPKEY) spk2_bgrad[d] = from_float(acc);
    else spk2_wgrad[d * NH_SPKEY + k] = from_float(acc);
}

// stage 2: scatter d_embed = spk_w[:, :32]^T . dkey into the fxp dE staging
__global__ void nh_spkey_dE_kernel(long long* __restrict__ dE_i,
    const precision_t* __restrict__ dkeys, const precision_t* __restrict__ spk_w,
    const float* __restrict__ sp_idx, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_EMBED_DIM) return;
    int bs = t / NH_EMBED_DIM;
    int d = t % NH_EMBED_DIM;
    int g = (int)sp_idx[bs];
    if (g < 0) return;
    float acc = 0.0f;
    for (int r = 0; r < NH_SPKEY; r++)
        acc += to_float(spk_w[r * NH_SPIN + d]) * to_float(dkeys[(int64_t)bs * NH_SPKEY + r]);
    if (acc != 0.0f) nh_fxp_atomic_add(&dE_i[(int64_t)g * NH_EMBED_DIM + d], acc);
}

__global__ void nh_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ loc,
    const precision_t* __restrict__ glb, const precision_t* __restrict__ inv,
    const precision_t* __restrict__ bl_out, const precision_t* __restrict__ bl_feats,
    const precision_t* __restrict__ msg, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_CONCAT) return;
    int b = idx / NH_CONCAT, c = idx % NH_CONCAT;
    precision_t val;
    if (c < NH_LOC_HID)
        val = loc[(int64_t)b * NH_LOC_HID + c];
    else if (c < NH_LOC_HID + NH_GLB_HID)
        val = glb[(int64_t)b * NH_GLB_HID + (c - NH_LOC_HID)];
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INV_POOL)
        val = inv[(int64_t)b * NH_INV_POOL + (c - NH_LOC_HID - NH_GLB_HID)];
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INV_POOL + NH_BL_HID)
        val = bl_out[(int64_t)b * NH_BL_HID + (c - NH_LOC_HID - NH_GLB_HID - NH_INV_POOL)];
    else if (c < NH_MSG_CONCAT_OFF)
        val = bl_feats[(int64_t)b * NH_BL_FEAT + (c - NH_LOC_HID - NH_GLB_HID - NH_INV_POOL - NH_BL_HID)];
    else if (c < NH_SPELL_CONCAT_OFF)
        val = msg[(int64_t)b * NH_MSG_HID + (c - NH_MSG_CONCAT_OFF)];
    else
        return; // spell slice: nh_spell_gather_kernel fills it afterwards
    out[idx] = val;
}

// Copy a per-sample slice [offset, offset+n) of a (B, stride) tensor into (B, n).
__global__ void nh_slice_kernel(
    precision_t* __restrict__ dst, const precision_t* __restrict__ src,
    int B, int stride, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n) return;
    dst[idx] = src[(idx / n) * stride + offset + idx % n];
}

// inventory entity branch

__global__ void nh_inv_decode_kernel(
    float* __restrict__ idx, const precision_t* __restrict__ obs, int B, int off) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV) return;
    int b = t / NH_INV, s = t % NH_INV;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + off + 2 * s;
    int g = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    idx[t] = (float)max(0, min(g, NH_GLYPH_VOCAB - 1));
}

// T_inv[g,k] = dot(E[g,:], inv1_w[k,:]) — 5977x16, sequential inner loop
__global__ void nh_inv_table_kernel(precision_t* __restrict__ T,
    const precision_t* __restrict__ E, const precision_t* __restrict__ w1) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_GLYPH_VOCAB * NH_INV_HID) return;
    int g = i / NH_INV_HID, k = i % NH_INV_HID;
    float acc = 0.0f;
    for (int d = 0; d < NH_EMBED_DIM; d++)
        acc += to_float(E[g * NH_EMBED_DIM + d]) * to_float(w1[k * NH_EMBED_DIM + d]);
    T[i] = from_float(acc);
}

// Expand the 8 gated int8 state fields into NH_SFEAT normalized features.
// Sentinels stay honest: unknown spe contributes (0 known-bit, 0 value).
__global__ void nh_inv_sfeat_kernel(precision_t* __restrict__ out,
    const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV) return;
    int b = t / NH_INV, s = t % NH_INV;
    const precision_t* src = obs + (int64_t)b * NH_OBS_SIZE + NH_INVST_OFF + s * NH_ST_RAW;
    int st[NH_ST_RAW];
    for (int j = 0; j < NH_ST_RAW; j++) {
        int v = (int)to_float(src[j]);
        st[j] = v >= 128 ? v - 256 : v; // bytes -> int8
    }
    precision_t* f = out + (int64_t)t * NH_SFEAT;
    for (int c = 0; c < 4; c++) f[c] = from_float(st[0] == c ? 1.0f : 0.0f);
    int spe_known = st[1] != -128;
    f[4] = from_float((float)spe_known);
    f[5] = from_float(spe_known ? (float)st[1] * 0.1f : 0.0f);
    f[6] = from_float(log1pf(fmaxf((float)st[2], 0.0f)) * 0.5f); // guard vs unclamped export drift
    f[7] = from_float((float)st[3] * (1.0f / 3.0f));
    f[8] = from_float((float)st[4] * (1.0f / 3.0f));
    for (int c = 0; c < 7; c++)
        f[9 + c] = from_float((float)((st[5] >> c) & 1));
    f[16] = from_float((float)st[6]);
    // armor slot category one-hot (suit/shield/helm/gloves/boots/cloak/shirt)
    // from the slot glyph via the engine's baked otyp->ARM_* table
    const precision_t* gsrc = obs + (int64_t)b * NH_OBS_SIZE + NH_INV_OFF + 2 * s;
    int g = (int)to_float(gsrc[0]) + ((int)to_float(gsrc[1]) << 8);
    int ot = g - NH_GLYPH_OBJ_OFF;
    int cat = (ot >= 0 && ot < NH_NUM_OBJECTS) ? nh_obj_armcat_dev[ot] : -1;
    for (int c = 0; c < 7; c++)
        f[17 + c] = from_float(cat == c ? 1.0f : 0.0f);
}

__global__ void nh_inv_gather_kernel(precision_t* __restrict__ out,
    const precision_t* __restrict__ T, const precision_t* __restrict__ b1,
    const precision_t* __restrict__ ws, const precision_t* __restrict__ sfeat,
    const float* __restrict__ idx, const precision_t* __restrict__ Tt,
    const float* __restrict__ idxt, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV_FLAT) return;
    int b = t / NH_INV_FLAT, r = t % NH_INV_FLAT;
    int s = r / NH_INV_HID, k = r % NH_INV_HID;
    int g = (int)idx[b * NH_INV + s];
    float v = to_float(T[g * NH_INV_HID + k]) + to_float(b1[k]);
    // discovered-type channel: pad = unknown identity, hard zero contribution
    int gt = (int)idxt[b * NH_INV + s];
    if (gt != NH_GLYPH_VOCAB - 1)
        v += to_float(Tt[gt * NH_INV_HID + k]);
    const precision_t* f = sfeat + ((int64_t)b * NH_INV + s) * NH_SFEAT;
    for (int j = 0; j < NH_SFEAT; j++)
        v += to_float(ws[k * NH_SFEAT + j]) * to_float(f[j]);
    out[t] = from_float(v > 0.0f ? v : 0.0f);
}

// Pooled inventory summary for the trunk: per-slot 32 -> 128 with the
// elementwise max over the 55 slots folded in (patch-encoder trick — the
// (B,55,128) tokens never exist). Fixed slot order + strict > keep argmax
// deterministic; ties (empty pad slots are identical) resolve to the lowest
// slot, matching torch.max in the test reference.
__global__ void nh_inv_max_kernel(precision_t* __restrict__ pool_out,
    int* __restrict__ argmax, const precision_t* __restrict__ inv_out,
    const precision_t* __restrict__ w2, const precision_t* __restrict__ b2, int B) {
    __shared__ float w2s[NH_INV_POOL * NH_INV_HID];
    __shared__ float ss[NH_INV_FLAT];
    int b = blockIdx.x;
    if (b >= B) return;
    for (int i = threadIdx.x; i < NH_INV_POOL * NH_INV_HID; i += blockDim.x)
        w2s[i] = to_float(w2[i]);
    for (int i = threadIdx.x; i < NH_INV_FLAT; i += blockDim.x)
        ss[i] = to_float(inv_out[(int64_t)b * NH_INV_FLAT + i]);
    __syncthreads();
    for (int o = threadIdx.x; o < NH_INV_POOL; o += blockDim.x) {
        float best = -1e30f;
        int bm = 0;
        for (int s = 0; s < NH_INV; s++) {
            float v = 0.0f;
            for (int k = 0; k < NH_INV_HID; k++)
                v += w2s[o * NH_INV_HID + k] * ss[s * NH_INV_HID + k];
            if (v > best) {
                best = v;
                bm = s;
            }
        }
        pool_out[(int64_t)b * NH_INV_POOL + o] = from_float(fmaxf(best + to_float(b2[o]), 0.0f));
        argmax[(int64_t)b * NH_INV_POOL + o] = bm;
    }
}

// Backward through the pooled max: dpool is already relu-masked (and b2's
// grad accumulated) by nh_relu_bias_bwd. dW2 stages in fixed-point smem; ds
// lands in inv_grad — every entry written, so callers skip the memset — where
// the pointer-decoder key grads and the slot relu mask are applied next.
// w2/inv_out are read through L2 (smem holds the two fxp accumulators: 46KB).
__global__ void nh_inv_max_bwd_kernel(precision_t* __restrict__ inv_grad,
    long long* __restrict__ dw2_acc, const precision_t* __restrict__ dpool,
    const precision_t* __restrict__ w2, const precision_t* __restrict__ inv_out,
    const int* __restrict__ argmax, int B) {
    __shared__ long long dss[NH_INV_FLAT];
    __shared__ long long dw2s[NH_INV_POOL * NH_INV_HID];
    int b = blockIdx.x;
    if (b >= B) return;
    for (int i = threadIdx.x; i < NH_INV_FLAT; i += blockDim.x) dss[i] = 0;
    for (int i = threadIdx.x; i < NH_INV_POOL * NH_INV_HID; i += blockDim.x) dw2s[i] = 0;
    __syncthreads();
    for (int o = threadIdx.x; o < NH_INV_POOL; o += blockDim.x) {
        float g = to_float(dpool[(int64_t)b * NH_INV_POOL + o]);
        if (g == 0.0f) continue;
        int m = argmax[(int64_t)b * NH_INV_POOL + o];
        for (int k = 0; k < NH_INV_HID; k++) {
            float dt = g * to_float(w2[o * NH_INV_HID + k]);
            if (dt != 0.0f)
                nh_fxp_atomic_add(&dss[m * NH_INV_HID + k], dt);
            float dw = g * to_float(inv_out[(int64_t)b * NH_INV_FLAT + m * NH_INV_HID + k]);
            if (dw != 0.0f)
                dw2s[o * NH_INV_HID + k] = (long long)__float2ll_rn(dw * NH_FXP);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NH_INV_POOL * NH_INV_HID; i += blockDim.x)
        if (dw2s[i] != 0)
            atomicAdd((unsigned long long*)&dw2_acc[i], (unsigned long long)dw2s[i]);
    for (int i = threadIdx.x; i < NH_INV_FLAT; i += blockDim.x)
        inv_grad[(int64_t)b * NH_INV_FLAT + i] = from_float(nh_fxp_to_float(dss[i]));
}

// dT_inv scatter, plain global fxp atomics: 55x16 per sample is too small
// for the hot-row machinery to pay
__global__ void nh_dTinv_scatter_kernel(long long* __restrict__ dT,
    const precision_t* __restrict__ dflat, const float* __restrict__ idx, int64_t n,
    int skip_pad) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float v = to_float(dflat[t]);
    if (v == 0.0f) return;
    int g = (int)idx[t / NH_INV_HID];
    // true-glyph channel: pad slots contributed hard zero in the forward
    if (skip_pad && g == NH_GLYPH_VOCAB - 1) return;
    nh_fxp_atomic_add(&dT[(int64_t)g * NH_INV_HID + t % NH_INV_HID], v);
}

__global__ void nh_add_inplace_kernel(precision_t* __restrict__ dst,
    const precision_t* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = from_float(to_float(dst[i]) + to_float(src[i]));
}

// Fused relu backward + bias grad: masks grad in place against out and
// accumulates the per-column sum into fixed-point bias_acc. Launch via
// nh_colsum_grid so (gridDim*blockDim) % dim == 0: each thread's column is
// then fixed across its grid-stride, so the sum lives in one register (fixed
// order -> deterministic) and costs one quantize + global atomic.
__global__ void nh_relu_bias_bwd_kernel(
    precision_t* __restrict__ grad, const precision_t* __restrict__ out,
    long long* __restrict__ bias_acc, int64_t total, int dim) {
    extern __shared__ long long sdata[];
    for (int j = threadIdx.x; j < dim; j += blockDim.x) sdata[j] = 0;
    __syncthreads();
    int64_t i0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int64_t i = i0; i < total; i += stride) {
        // branch-free: a divergent store-vs-load branch here runs ~3x slower
        float g = to_float(out[i]) > 0.0f ? to_float(grad[i]) : 0.0f;
        grad[i] = from_float(g);
        acc += g;
    }
    if (acc != 0.0f) nh_fxp_atomic_add(&sdata[(int)(i0 % dim)], acc);
    __syncthreads();
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        if (sdata[j] != 0) atomicAdd((unsigned long long*)&bias_acc[j], (unsigned long long)sdata[j]);
}

static inline int nh_colsum_grid(int64_t total, int dim) {
    int64_t g = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (g > 1024) g = 1024;
    while ((g * BLOCK_SIZE) % dim) g++;
    return (int)g;
}

// Cast the packed fixed-point bias accumulators to their grad tensors in one launch.
__global__ void nh_bias_flush_kernel(
    const long long* __restrict__ acc,
    precision_t* __restrict__ d0, int n0, precision_t* __restrict__ d1, int n1,
    precision_t* __restrict__ d2, int n2, precision_t* __restrict__ d3, int n3,
    precision_t* __restrict__ d4, int n4, precision_t* __restrict__ d5, int n5,
    precision_t* __restrict__ d6, int n6) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n0 + n1 + n2 + n3 + n4 + n5 + n6 ? nh_fxp_to_float(acc[i]) : 0.0f;
    if (i < n0) d0[i] = from_float(v);
    else if ((i -= n0) < n1) d1[i] = from_float(v);
    else if ((i -= n1) < n2) d2[i] = from_float(v);
    else if ((i -= n2) < n3) d3[i] = from_float(v);
    else if ((i -= n3) < n4) d4[i] = from_float(v);
    else if ((i -= n4) < n5) d5[i] = from_float(v);
    else if ((i -= n5) < n6) d6[i] = from_float(v);
}

// embedding backward
// Both views are linear in the embeddings, so dE is a scatter-add of per-cell
// 32-dim grad vectors into glyph rows. The dominant glyphs (unexplored stone,
// floor, walls cover ~80% of cells) contend on the same rows; per-block smem
// accumulators for the top-NH_HOT_T glyphs absorb that, the cold tail goes
// straight to global atomics. Every element quantizes exactly once.

// Per-minibatch glyph histogram (per-block smem: hot counters would otherwise
// serialize global atomics).
__global__ void nh_hist_kernel(int* __restrict__ counts, const float* __restrict__ idx, int N) {
    __shared__ int hist[NH_GLYPH_VOCAB];
    for (int i = threadIdx.x; i < NH_GLYPH_VOCAB; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    int chunk = (N + gridDim.x - 1) / gridDim.x;
    int start = blockIdx.x * chunk, end = min(start + chunk, N);
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
        atomicAdd(&hist[(int)idx[i]], 1);
    __syncthreads();
    for (int g = threadIdx.x; g < NH_GLYPH_VOCAB; g += blockDim.x)
        if (hist[g]) atomicAdd(&counts[g], hist[g]);
}

// Top-K glyphs by count (single block; counts are consumed).
// hot_map must be pre-set to -1.
__global__ void nh_hot_select_kernel(
    int* __restrict__ hot_map, int* __restrict__ hot_list, int* __restrict__ hot_n,
    int* __restrict__ counts, int K) {
    __shared__ int best_v[1024], best_g[1024];
    int tid = threadIdx.x;
    for (int k = 0; k < K; k++) {
        int bv = 0, bg = -1;
        for (int g = tid; g < NH_GLYPH_VOCAB; g += blockDim.x)
            if (counts[g] > bv) {
                bv = counts[g];
                bg = g;
            }
        best_v[tid] = bv; best_g[tid] = bg;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off && best_v[tid + off] > best_v[tid]) {
                best_v[tid] = best_v[tid + off]; best_g[tid] = best_g[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0 && best_g[0] >= 0) {
            hot_map[best_g[0]] = k;
            hot_list[k] = best_g[0];
            counts[best_g[0]] = 0;
            *hot_n = k + 1;
        }
        __syncthreads();
    }
}

// Local-view dE scatter: add each crop cell's 32-dim grad vector (contiguous
// in dvec) into its glyph's embed-grad row.
__global__ void nh_dE_scatter_kernel(
    long long* __restrict__ dE_i, const precision_t* __restrict__ dvec,
    const float* __restrict__ gidx, const int* __restrict__ hot_map,
    const int* __restrict__ hot_list, const int* __restrict__ hot_n,
    int64_t ncell) {
    extern __shared__ long long acc_s[]; // NH_HOT_T x NH_EMBED_DIM
    for (int i = threadIdx.x; i < NH_HOT_T * NH_EMBED_DIM; i += blockDim.x)
        acc_s[i] = 0;
    __syncthreads();
    int64_t total = ncell * NH_EMBED_DIM;
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (int64_t)gridDim.x * blockDim.x) {
        float v = to_float(dvec[t]);
        if (v == 0.0f) continue;
        unsigned long long q = (unsigned long long)(long long)__float2ll_rn(v * NH_FXP);
        if (q == 0) continue;
        int d = t % NH_EMBED_DIM;
        int g = (int)gidx[t / NH_EMBED_DIM];
        int slot = hot_map[g];
        if (slot >= 0) atomicAdd((unsigned long long*)&acc_s[slot * NH_EMBED_DIM + d], q);
        else atomicAdd((unsigned long long*)&dE_i[(int64_t)g * NH_EMBED_DIM + d], q);
    }
    __syncthreads();
    int n = *hot_n;
    for (int i = threadIdx.x; i < n * NH_EMBED_DIM; i += blockDim.x) {
        long long v = acc_s[i];
        if (v != 0)
            atomicAdd((unsigned long long*)&dE_i[(int64_t)hot_list[i / NH_EMBED_DIM] * NH_EMBED_DIM + i % NH_EMBED_DIM],
                      (unsigned long long)v);
    }
}

// Global-view dT scatter: dT[g, pos*P1+k] += dt16[b, tk*P1+k] for every
// (token, position) occurrence of glyph g. One thread per (b, tk, k) element,
// quantized once; the 32 positions then add the same integer (hot glyphs via
// smem, cold straight to global — same scheme as the old conv1 dT scatter).
__global__ void nh_dT_patch_scatter_kernel(
    long long* __restrict__ dT_i, const precision_t* __restrict__ dt16,
    const float* __restrict__ idx, const int* __restrict__ hot_map,
    const int* __restrict__ hot_list, const int* __restrict__ hot_n, int B) {
    extern __shared__ long long acc_s[]; // NH_HOT_G x NH_TROW
    for (int i = threadIdx.x; i < NH_HOT_G * NH_TROW; i += blockDim.x)
        acc_s[i] = 0;
    __syncthreads();
    int64_t total = (int64_t)B * NH_TOK * NH_P1;
    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (int64_t)gridDim.x * blockDim.x) {
        float g = to_float(dt16[t]);
        if (g == 0.0f) continue;
        unsigned long long q = (unsigned long long)(long long)__float2ll_rn(g * NH_FXP);
        if (q == 0) continue;
        int k = t % NH_P1;
        int tk = (t / NH_P1) % NH_TOK;
        int64_t b = t / (NH_P1 * NH_TOK);
        int r0 = (tk / NH_PX) * NH_PH, c0 = (tk % NH_PX) * NH_PW;
        const float* gi = idx + b * NH_MGRID;
        #pragma unroll
        for (int pos = 0; pos < NH_PCELLS; pos++) {
            int r = r0 + pos / NH_PW, c = c0 + pos % NH_PW;
            int gl = (r < NH_MAPH && c < NH_MAPW) ? (int)gi[r * NH_MAPW + c] : NH_PAD_GLYPH;
            int slot = hot_map[gl];
            if (slot >= 0) atomicAdd((unsigned long long*)&acc_s[slot * NH_TROW + pos * NH_P1 + k], q);
            else atomicAdd((unsigned long long*)&dT_i[(int64_t)gl * NH_TROW + pos * NH_P1 + k], q);
        }
    }
    __syncthreads();
    int n = *hot_n;
    for (int i = threadIdx.x; i < n * NH_TROW; i += blockDim.x) {
        long long v = acc_s[i];
        if (v != 0)
            atomicAdd((unsigned long long*)&dT_i[(int64_t)hot_list[i / NH_TROW] * NH_TROW + i % NH_TROW],
                      (unsigned long long)v);
    }
}

// Plain per-column sum (b1's grad from dt16 rows); same launch contract as
// nh_relu_bias_bwd (dt16 is already relu-masked there, so no second mask).
__global__ void nh_col_sum_kernel(
    long long* __restrict__ bias_acc, const precision_t* __restrict__ src,
    int64_t total, int dim) {
    extern __shared__ long long sdata[];
    for (int j = threadIdx.x; j < dim; j += blockDim.x) sdata[j] = 0;
    __syncthreads();
    int64_t i0 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (int64_t i = i0; i < total; i += stride) acc += to_float(src[i]);
    if (acc != 0.0f) nh_fxp_atomic_add(&sdata[(int)(i0 % dim)], acc);
    __syncthreads();
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        if (sdata[j] != 0) atomicAdd((unsigned long long*)&bias_acc[j], (unsigned long long)sdata[j]);
}

// += variant of the row-sparse cast: adds the local view's embed grads on top
// of the global view's GEMM-produced dE (same guard + re-zero contract).
__global__ void nh_fxp_add_rows_kernel(
    precision_t* __restrict__ dst, long long* __restrict__ src,
    const int* __restrict__ counts, const int* __restrict__ hot_map,
    int trow, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int row = idx / trow;
    if (counts[row] == 0 && hot_map[row] < 0) return;
    dst[idx] = from_float(to_float(dst[idx]) + nh_fxp_to_float(src[idx]));
    src[idx] = 0;
}

// Seed the grid-glyph histogram with the static pad-glyph count (edge-patch
// positions past the map read the pad row; the hist over glyph_idx can't see
// them, but the rows-cast guard and hot selection must).
__global__ void nh_count_pad_kernel(int* __restrict__ counts, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        counts[NH_PAD_GLYPH] += NH_PAD_PER_SAMPLE * B;
}

#if NH_ID_EMBED
// identity embeddings: indices recovered from the one-hot obs block (align
// from blstats), table rows copied raw into the concat tail (bl-feats idiom)
__global__ void nh_idemb_kernel(precision_t* __restrict__ concat,
    float* __restrict__ idx_out, const precision_t* __restrict__ obs,
    const precision_t* __restrict__ role_w, const precision_t* __restrict__ race_w,
    const precision_t* __restrict__ gend_w, const precision_t* __restrict__ algn_w, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const precision_t* bl = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
    const precision_t* ex = bl + 4 * NH_BL_RAW;
    int r = 0, rc = 0, g = 0;
    for (int i = 0; i < 13; i++) if (nh_bl_read_i32(ex + 4 * (NH_EX_ROLEOH + i))) r = i;
    for (int i = 0; i < 5; i++) if (nh_bl_read_i32(ex + 4 * (NH_EX_ROLEOH + 13 + i))) rc = i;
    for (int i = 0; i < 2; i++) if (nh_bl_read_i32(ex + 4 * (NH_EX_ROLEOH + 18 + i))) g = i;
    int al = 1 - nh_bl_read_i32(bl + 4 * 26); // align: lawful 1, neutral 0, chaotic -1
    al = al < 0 ? 0 : al > 2 ? 2 : al;
    idx_out[b * 4 + 0] = (float)r;
    idx_out[b * 4 + 1] = (float)rc;
    idx_out[b * 4 + 2] = (float)g;
    idx_out[b * 4 + 3] = (float)al;
    precision_t* c = concat + (int64_t)b * NH_CONCAT + NH_IDE_CONCAT_OFF;
    for (int d = 0; d < NH_IDE_ROLE; d++) c[d] = role_w[r * NH_IDE_ROLE + d];
    c += NH_IDE_ROLE;
    for (int d = 0; d < NH_IDE_RACE; d++) c[d] = race_w[rc * NH_IDE_RACE + d];
    c += NH_IDE_RACE;
    for (int d = 0; d < NH_IDE_GEND; d++) c[d] = gend_w[g * NH_IDE_GEND + d];
    c += NH_IDE_GEND;
    for (int d = 0; d < NH_IDE_ALGN; d++) c[d] = algn_w[al * NH_IDE_ALGN + d];
}

// table grads: one thread per (row, dim), fixed batch loop -- deterministic
__global__ void nh_idemb_grad_kernel(precision_t* __restrict__ role_g,
    precision_t* __restrict__ race_g, precision_t* __restrict__ gend_g,
    precision_t* __restrict__ algn_g, const precision_t* __restrict__ grad_concat,
    const float* __restrict__ idx, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_role = 13 * NH_IDE_ROLE, n_race = 5 * NH_IDE_RACE;
    const int n_gend = 2 * NH_IDE_GEND, n_algn = 3 * NH_IDE_ALGN;
    if (t >= n_role + n_race + n_gend + n_algn) return;
    int comp, wdt, off;
    precision_t* out;
    int u = t;
    if (u < n_role) { comp = 0; wdt = NH_IDE_ROLE; off = 0; out = role_g; }
    else if ((u -= n_role) < n_race) { comp = 1; wdt = NH_IDE_RACE; off = NH_IDE_ROLE; out = race_g; }
    else if ((u -= n_race) < n_gend) { comp = 2; wdt = NH_IDE_GEND; off = NH_IDE_ROLE + NH_IDE_RACE; out = gend_g; }
    else { u -= n_gend; comp = 3; wdt = NH_IDE_ALGN; off = NH_IDE_ROLE + NH_IDE_RACE + NH_IDE_GEND; out = algn_g; }
    int row = u / wdt, d = u % wdt;
    float acc = 0.0f;
    for (int b = 0; b < B; b++) {
        if ((int)idx[b * 4 + comp] != row) continue;
        acc += to_float(grad_concat[(int64_t)b * NH_CONCAT + NH_IDE_CONCAT_OFF + off + d]);
    }
    out[row * wdt + d] = from_float(acc);
}
#endif

// encoder structs

struct NethackEncoderWeights {
    Prec embed_w, ekind_w, esub_w, loc_w, loc_b;
    Prec glb1_w, glb1_xy, glb1_b, glb2_w, glb2_b;
    Prec inv1_w, inv1_b, inv1s_w, invt_w, inv2_w, inv2_b;
    Prec bl_w, bl_b, proj_w, proj_b;
    Prec msg_w; // trigram embedding table (NH_MSG_VOCAB, NH_MSG_HID)
    Prec spk_w; // spell slot-rep projection (NH_SPKEY, NH_SPIN)
    Prec spk2_w, spk2_b; // spell pool projection + bias (inv2 idiom)
#if NH_ID_EMBED
    Prec ide_role_w, ide_race_w, ide_gend_w, ide_algn_w; // identity tables
#endif
    int obs_size, hidden;
};

struct NethackEncoderActivations {
    Float glyph_idx, crop_glyph; // decoded grid + crop glyph ids
    Prec e_eff; // materialized E_res + E_kind + E_sub
    Prec x_local; // crop embeds (grad aliases it)
    Prec w_perm, glyph_T; // fused embed+flatten+glb1 table
    Prec t16; // relu'd patch tokens (dt16 overwrites)
    Prec dxy; // per-token hero offsets (w_xy wgrad)
    Int tok_argmax; // winning token per (sample, out dim)
    Float inv_idx; // inventory slot glyph ids
    Float spell_idx; // per-slot book glyphs (-1 = empty slot)
    Prec spk_in, spk_keys; // spell-key inputs (B, 8*36) + relu'd reps (B, 8*16)
    Prec spk_dkeys; // total per-slot rep grads (pool + pointer)
    Int spk_amax; // pool argmax slot per (sample, pool dim)
    Prec spk_pool; // pooled forward values (relu gate for backward;
                                // concat is grad-aliased and unreadable then)
    Float invt_idx; // discovered-type glyph ids (pad = unknown)
    Prec inv_sfeat; // per-slot state features (B, 55*NH_SFEAT)
    Prec inv_T, inv_out; // fused inv table + relu'd flat slots
    Prec invt_T; // fused discovered-type table
    Prec inv_pool; // pooled 128-dim trunk summary
    Int inv_amax; // winning slot per (sample, pool dim)
    Prec loc_out, glb_out;
    Prec bl_feats, bl_out;
    Float msg_ids; // per-position trigram bucket ids (-1 pad)
    Prec msg_out; // normalized trigram-bag summary (B, NH_MSG_HID)
    Prec concat, out;
    Prec loc_grad, glb_grad, inv_grad, bl_grad; // contiguous concat slices
    Prec inv_pool_grad; // pooled-summary slice of concat grad
    Prec dT, dw_perm; // dT table + permuted glb1 wgrad
    Prec dTinv, dE_tmp; // inv-table grad + its dE staging
    Prec dTtrue; // discovered-type table grad
    Long dT_i, dTinv_i, dTtrue_i; // fixed-point dT scatter staging
    Long dE_i; // fixed-point local embed-grad staging
    Long dw2_acc; // fixed-point glb2 wgrad staging
    Long dw2i_acc; // fixed-point inv2 wgrad staging
    Long dmsg_acc; // fixed-point trigram-table wgrad staging
    Long bias_acc; // fixed-point bias grads: proj | loc | glb2 | bl | glb1 | inv1 | inv2
    Int sort_local, sort_grid; // counts | hot_map | hot_list | hot_n
    Prec embed_wgrad, ekind_wgrad, esub_wgrad, loc_wgrad, loc_bgrad;
    Prec glb1_wgrad, glb1_xygrad, glb1_bgrad, glb2_wgrad, glb2_bgrad;
    Prec inv1_wgrad, inv1_bgrad, inv1s_wgrad, invt_wgrad, inv2_wgrad, inv2_bgrad;
    Prec bl_wgrad, bl_bgrad, proj_wgrad, proj_bgrad;
    Prec msg_wgrad, spk_wgrad, spk2_wgrad, spk2_bgrad;
#if NH_ID_EMBED
    Float ide_idx; // per-sample [role, race, gend, align] saved for backward
    Prec ide_role_wgrad, ide_race_wgrad, ide_gend_wgrad, ide_algn_wgrad;
#endif
};

static NethackEncoderWeights* nethack_encoder_create(int obs_size, int hidden) {
    if (obs_size != NH_OBS_SIZE) {
        fprintf(stderr, "nethack encoder: obs size %d != expected %d "
            "(env obs layout out of sync with ocean/nethack/nethack.cu?)\n",
            obs_size, NH_OBS_SIZE);
        exit(1);
    }
    NethackEncoderWeights* ew = (NethackEncoderWeights*)calloc(1, sizeof(NethackEncoderWeights));
    ew->obs_size = obs_size; ew->hidden = hidden;
    return ew;
}

// encoder interface

// encoder <-> pointer-decoder wiring
// The decoder's slot head is a pointer over the inventory branch's per-slot
// vectors: it reads the encoder's inv_out (keys), and the encoder backward
// adds the decoder's key gradients into the inv slice (inv_out has two grad
// consumers: the concat slice and the keys). Struct pointers are fixed at
// registration (cudagraph-safe); train vs rollout resolved by batch size.
struct NethackDecoderActivations;
// encoder acts register immediately before their partner decoder acts inside
// each arch_reg_* call; the decoder captures this at its own reg time, so
// every rollout buffer's decoder reads its own buffer's inv_out.
static NethackEncoderActivations* nh_enc_last = NULL;
static Prec* nh_ptr_keygrad = NULL; // train decoder's (B_TT, NH_INV_FLAT)
static Prec* nh_ptr_spkeygrad = NULL; // train decoder's spell-key grads (B_TT, 8*NH_SPKEY)

static Prec nethack_encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = input.shape[0];

    nh_eff_embed_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->e_eff.data, ew->embed_w.data, ew->ekind_w.data, ew->esub_w.data);
    nh_decode_kernel<<<grid_size(B * NH_MGRID), BLOCK_SIZE, 0, stream>>>(
        a->glyph_idx.data, input.data, B);
    nh_crop_kernel<<<grid_size(B * NH_CGRID), BLOCK_SIZE, 0, stream>>>(
        a->crop_glyph.data, a->glyph_idx.data, input.data, B);

    nh_local_gather_kernel<<<grid_size(B * NH_LOC_IN), BLOCK_SIZE, 0, stream>>>(
        a->x_local.data, a->e_eff.data, a->crop_glyph.data, B);
    puf_mm(&a->x_local, &ew->loc_w, &a->loc_out, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LOC_HID), BLOCK_SIZE, 0, stream>>>(
        a->loc_out.data, ew->loc_b.data, B * NH_LOC_HID, NH_LOC_HID);

    nh_permute_g1_kernel<<<grid_size(NH_TROW * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->w_perm.data, ew->glb1_w.data);
    puf_mm(&a->e_eff, &a->w_perm, &a->glyph_T, stream);
    nh_patch_max_kernel<<<B, 128, 0, stream>>>(
        a->glb_out.data, a->t16.data, a->dxy.data, a->tok_argmax.data,
        a->glyph_T.data, ew->glb1_b.data, ew->glb1_xy.data, ew->glb2_w.data,
        ew->glb2_b.data, a->glyph_idx.data, input.data, B);

    nh_inv_decode_kernel<<<grid_size(B * NH_INV), BLOCK_SIZE, 0, stream>>>(
        a->inv_idx.data, input.data, B, NH_INV_OFF);
    nh_inv_decode_kernel<<<grid_size(B * NH_INV), BLOCK_SIZE, 0, stream>>>(
        a->invt_idx.data, input.data, B, NH_INVTRUE_OFF);
    nh_inv_table_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->inv_T.data, a->e_eff.data, ew->inv1_w.data);
    nh_inv_table_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->invt_T.data, a->e_eff.data, ew->invt_w.data);
    nh_inv_sfeat_kernel<<<grid_size(B * NH_INV), BLOCK_SIZE, 0, stream>>>(
        a->inv_sfeat.data, input.data, B);
    nh_inv_gather_kernel<<<grid_size(B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->inv_out.data, a->inv_T.data, ew->inv1_b.data, ew->inv1s_w.data,
        a->inv_sfeat.data, a->inv_idx.data, a->invt_T.data, a->invt_idx.data, B);
    nh_inv_max_kernel<<<B, 128, 0, stream>>>(
        a->inv_pool.data, a->inv_amax.data, a->inv_out.data,
        ew->inv2_w.data, ew->inv2_b.data, B);

    nh_blstats_kernel<<<grid_size(B * 32), BLOCK_SIZE, 0, stream>>>(
        a->bl_feats.data, input.data, B);
    puf_mm(&a->bl_feats, &ew->bl_w, &a->bl_out, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_out.data, ew->bl_b.data, B * NH_BL_HID, NH_BL_HID);

    nh_msg_ids_kernel<<<grid_size(B * NH_MSG_LEN), BLOCK_SIZE, 0, stream>>>(
        a->msg_ids.data, input.data, B);
    nh_msg_pool_kernel<<<B, NH_MSG_HID, 0, stream>>>(
        a->msg_out.data, ew->msg_w.data, a->msg_ids.data, B);

    nh_concat_kernel<<<grid_size(B * NH_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->loc_out.data, a->glb_out.data, a->inv_pool.data,
        a->bl_out.data, a->bl_feats.data, a->msg_out.data, B);
    nh_spkey_kernel<<<grid_size(B * NH_SPELL_SLOTS), BLOCK_SIZE, 0, stream>>>(
        a->spk_keys.data, a->spk_in.data, a->spell_idx.data,
        ew->spk_w.data, a->e_eff.data, input.data, B);
    nh_sppool_kernel<<<grid_size(B * NH_SPKEY), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->spk_pool.data, a->spk_amax.data, a->spk_keys.data,
        ew->spk2_w.data, ew->spk2_b.data, B);
#if NH_ID_EMBED
    nh_idemb_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->ide_idx.data, input.data,
        ew->ide_role_w.data, ew->ide_race_w.data,
        ew->ide_gend_w.data, ew->ide_algn_w.data, B);
#endif
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    nh_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nethack_encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    // fixed-point bias-grad accumulators: [proj H | loc 256 | glb2 128 | bl 64 | glb1 16 | inv1 32 | inv2 128]
    long long* bacc = (long long*)a->bias_acc.data;
    cudaMemsetAsync(bacc, 0, (H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID + NH_INV_POOL) * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * H, H), BLOCK_SIZE, H * sizeof(long long), stream>>>(
        grad.data, a->out.data, bacc, (int64_t)B * H, H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);

    Prec grad_concat = {.data = a->concat.data, .shape = {B, NH_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

#if NH_ID_EMBED
    nh_idemb_grad_kernel<<<grid_size(13 * NH_IDE_ROLE + 5 * NH_IDE_RACE
        + 2 * NH_IDE_GEND + 3 * NH_IDE_ALGN), BLOCK_SIZE, 0, stream>>>(
        a->ide_role_wgrad.data, a->ide_race_wgrad.data, a->ide_gend_wgrad.data,
        a->ide_algn_wgrad.data, grad_concat.data, a->ide_idx.data, B);
#endif

    // Local view: wgrad against saved x_local, then the input grad overwrites
    // x_local in place before scattering into the embed table.
    nh_slice_kernel<<<grid_size(B * NH_LOC_HID), BLOCK_SIZE, 0, stream>>>(
        a->loc_grad.data, grad_concat.data, B, NH_CONCAT, 0, NH_LOC_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LOC_HID, NH_LOC_HID), BLOCK_SIZE, NH_LOC_HID * sizeof(long long), stream>>>(
        a->loc_grad.data, a->loc_out.data, bacc + H, (int64_t)B * NH_LOC_HID, NH_LOC_HID);
    Prec locg = {.data = a->loc_grad.data, .shape = {B, NH_LOC_HID}};
    puf_mm_tn(&locg, &a->x_local, &a->loc_wgrad, stream);
    Prec dx_local = {.data = a->x_local.data, .shape = {B, NH_LOC_IN}};
    puf_mm_nn(&locg, &ew->loc_w, &dx_local, stream);

    // Global view: relu mask + b2 grad, then the fused max backward (dW2 via
    // fixed-point staging, dt16 overwrites t16), then b1's column sum.
    nh_slice_kernel<<<grid_size(B * NH_GLB_HID), BLOCK_SIZE, 0, stream>>>(
        a->glb_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID, NH_GLB_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_GLB_HID, NH_GLB_HID), BLOCK_SIZE, NH_GLB_HID * sizeof(long long), stream>>>(
        a->glb_grad.data, a->glb_out.data, bacc + H + NH_LOC_HID, (int64_t)B * NH_GLB_HID, NH_GLB_HID);
    cudaMemsetAsync(a->dw2_acc.data, 0, NH_GLB_HID * NH_P1 * sizeof(long long), stream);
    nh_patch_max_bwd_kernel<<<B, 128, 0, stream>>>(
        a->t16.data, (long long*)a->dw2_acc.data, a->glb_grad.data,
        ew->glb2_w.data, a->tok_argmax.data, B);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLB_HID * NH_P1), BLOCK_SIZE, 0, stream>>>(
        a->glb2_wgrad.data, (long long*)a->dw2_acc.data, NH_GLB_HID * NH_P1);
    nh_col_sum_kernel<<<nh_colsum_grid((int64_t)B * NH_TOK * NH_P1, NH_P1), BLOCK_SIZE, NH_P1 * sizeof(long long), stream>>>(
        bacc + H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID, a->t16.data, (int64_t)B * NH_TOK * NH_P1, NH_P1);
    // (dx,dy) weight slice: dW_xy = dt16^T @ dxy (tall-K, 16x2 output)
    Prec dt16v = {.data = a->t16.data, .shape = {B * NH_TOK, NH_P1}};
    Prec dxyv = {.data = a->dxy.data, .shape = {B * NH_TOK, 2}};
    puf_mm_tn(&dt16v, &dxyv, &a->glb1_xygrad, stream);

    // Inventory branch: slice the pooled-summary grad, relu-mask it (inv2
    // bias grad rides along), backprop the fused max into inv_grad (dW2 via
    // fixed-point staging), then the per-slot relu mask + inv1 bias, dT_inv
    // scatter by slot glyph, dinv1_w = dT_inv^T @ E; dE added at the end.
    nh_slice_kernel<<<grid_size(B * NH_INV_POOL), BLOCK_SIZE, 0, stream>>>(
        a->inv_pool_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID + NH_GLB_HID, NH_INV_POOL);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_INV_POOL, NH_INV_POOL), BLOCK_SIZE, NH_INV_POOL * sizeof(long long), stream>>>(
        a->inv_pool_grad.data, a->inv_pool.data,
        bacc + H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID,
        (int64_t)B * NH_INV_POOL, NH_INV_POOL);
    cudaMemsetAsync(a->dw2i_acc.data, 0, NH_INV_POOL * NH_INV_HID * sizeof(long long), stream);
    nh_inv_max_bwd_kernel<<<B, 128, 0, stream>>>(
        a->inv_grad.data, (long long*)a->dw2i_acc.data, a->inv_pool_grad.data,
        ew->inv2_w.data, a->inv_out.data, a->inv_amax.data, B);
    nh_fxp_to_precision_kernel<<<grid_size(NH_INV_POOL * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->inv2_wgrad.data, (long long*)a->dw2i_acc.data, NH_INV_POOL * NH_INV_HID);
    // pointer-decoder key grads: second consumer of inv_out, summed before
    // the relu mask (both paths read the post-relu slot vectors)
    if (nh_ptr_keygrad != NULL)
        nh_add_inplace_kernel<<<grid_size(B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
            a->inv_grad.data, nh_ptr_keygrad->data, B * NH_INV_FLAT);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_INV_FLAT, NH_INV_HID), BLOCK_SIZE, NH_INV_HID * sizeof(long long), stream>>>(
        a->inv_grad.data, a->inv_out.data, bacc + H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1,
        (int64_t)B * NH_INV_FLAT, NH_INV_HID);
    // state-path weight grad: dW_s = dslot^T @ sfeat over the B*55 slot rows
    Prec dsflat = {.data = a->inv_grad.data, .shape = {B * NH_INV, NH_INV_HID}};
    Prec sfflat = {.data = a->inv_sfeat.data, .shape = {B * NH_INV, NH_SFEAT}};
    puf_mm_tn(&dsflat, &sfflat, &a->inv1s_wgrad, stream);
    cudaMemsetAsync(a->dTinv_i.data, 0, (size_t)NH_GLYPH_VOCAB * NH_INV_HID * sizeof(long long), stream);
    nh_dTinv_scatter_kernel<<<grid_size((int64_t)B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dTinv_i.data, a->inv_grad.data, a->inv_idx.data, (int64_t)B * NH_INV_FLAT, 0);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->dTinv.data, (long long*)a->dTinv_i.data, NH_GLYPH_VOCAB * NH_INV_HID);
    puf_mm_tn(&a->dTinv, &a->e_eff, &a->inv1_wgrad, stream);
    // discovered-type channel: same fused-table backward keyed by true glyph
    cudaMemsetAsync(a->dTtrue_i.data, 0, (size_t)NH_GLYPH_VOCAB * NH_INV_HID * sizeof(long long), stream);
    nh_dTinv_scatter_kernel<<<grid_size((int64_t)B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dTtrue_i.data, a->inv_grad.data, a->invt_idx.data, (int64_t)B * NH_INV_FLAT, 1);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->dTtrue.data, (long long*)a->dTtrue_i.data, NH_GLYPH_VOCAB * NH_INV_HID);
    puf_mm_tn(&a->dTtrue, &a->e_eff, &a->invt_wgrad, stream);

    // Blstats branch (raw-feature slice of concat has no upstream params)
    nh_slice_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID + NH_GLB_HID + NH_INV_POOL, NH_BL_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_BL_HID, NH_BL_HID), BLOCK_SIZE, NH_BL_HID * sizeof(long long), stream>>>(
        a->bl_grad.data, a->bl_out.data, bacc + H + NH_LOC_HID + NH_GLB_HID, (int64_t)B * NH_BL_HID, NH_BL_HID);
    Prec blg = {.data = a->bl_grad.data, .shape = {B, NH_BL_HID}};
    puf_mm_tn(&blg, &a->bl_feats, &a->bl_wgrad, stream);

    // Message branch: scatter (1/sqrt(count+1))*dout into the trigram embedding
    // grad (fixed-point). Reads its grad straight off the concat-grad slice;
    // ids saved from the forward, so no obs re-read.
    cudaMemsetAsync(a->dmsg_acc.data, 0, (size_t)NH_MSG_VOCAB * NH_MSG_HID * sizeof(long long), stream);
    nh_msg_bwd_kernel<<<B, NH_MSG_HID, 0, stream>>>(
        (long long*)a->dmsg_acc.data, grad_concat.data, a->msg_ids.data, B);
    nh_fxp_to_precision_kernel<<<grid_size(NH_MSG_VOCAB * NH_MSG_HID), BLOCK_SIZE, 0, stream>>>(
        a->msg_wgrad.data, (long long*)a->dmsg_acc.data, NH_MSG_VOCAB * NH_MSG_HID);

    // Global branch to the embed table + glb1: scatter dt16 occurrences into
    // dT, then dE = dT @ W' and dW' = dT^T @ E (the fused-table backward).
    int* counts_g = a->sort_grid.data;
    int* hot_map_g = counts_g + NH_GLYPH_VOCAB;
    int* hot_list_g = hot_map_g + NH_GLYPH_VOCAB;
    int* hot_n_g = hot_list_g + NH_HOT_G;
    cudaMemsetAsync(counts_g, 0, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_map_g, 0xFF, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_n_g, 0, sizeof(int), stream);
    nh_hist_kernel<<<NH_SORT_BLOCKS, 256, 0, stream>>>(counts_g, a->glyph_idx.data, B * NH_MGRID);
    nh_count_pad_kernel<<<1, 1, 0, stream>>>(counts_g, B);
    nh_hot_select_kernel<<<1, 1024, 0, stream>>>(hot_map_g, hot_list_g, hot_n_g, counts_g, NH_HOT_G);
    nh_dT_patch_scatter_kernel<<<128, 1024, NH_HOT_G * NH_TROW * sizeof(long long), stream>>>(
        (long long*)a->dT_i.data, a->t16.data, a->glyph_idx.data,
        hot_map_g, hot_list_g, hot_n_g, B);
    int dT_n = NH_GLYPH_VOCAB * NH_TROW;
    nh_fxp_to_precision_rows_kernel<<<grid_size(dT_n), BLOCK_SIZE, 0, stream>>>(
        a->dT.data, (long long*)a->dT_i.data, counts_g, hot_map_g, NH_TROW, dT_n);
    puf_mm_nn(&a->dT, &a->w_perm, &a->embed_wgrad, stream); // dE = dT @ W'
    puf_mm_tn(&a->dT, &a->e_eff, &a->dw_perm, stream); // dW' = dT^T @ E_eff
    nh_unpermute_g1_kernel<<<grid_size(NH_TROW * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->glb1_wgrad.data, a->dw_perm.data);

    // Local branch adds its embed grads on top of the GEMM-produced dE.
    int* counts_l = a->sort_local.data;
    int* hot_map_l = counts_l + NH_GLYPH_VOCAB;
    int* hot_list_l = hot_map_l + NH_GLYPH_VOCAB;
    int* hot_n_l = hot_list_l + NH_HOT_T;
    cudaMemsetAsync(counts_l, 0, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_map_l, 0xFF, NH_GLYPH_VOCAB * sizeof(int), stream);
    cudaMemsetAsync(hot_n_l, 0, sizeof(int), stream);
    nh_hist_kernel<<<NH_SORT_BLOCKS, 256, 0, stream>>>(counts_l, a->crop_glyph.data, B * NH_CGRID);
    nh_hot_select_kernel<<<1, 1024, 0, stream>>>(hot_map_l, hot_list_l, hot_n_l, counts_l, NH_HOT_T);
    nh_dE_scatter_kernel<<<1024, BLOCK_SIZE, NH_HOT_T * NH_EMBED_DIM * sizeof(long long), stream>>>(
        (long long*)a->dE_i.data, dx_local.data, a->crop_glyph.data,
        hot_map_l, hot_list_l, hot_n_l, (int64_t)B * NH_CGRID);
    int dE_n = NH_GLYPH_VOCAB * NH_EMBED_DIM;
    nh_fxp_add_rows_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, (long long*)a->dE_i.data, counts_l, hot_map_l, NH_EMBED_DIM, dE_n);

    // Inventory branch adds its embed grads last: dE += dT_inv @ inv1_w.
    puf_mm_nn(&a->dTinv, &ew->inv1_w, &a->dE_tmp, stream);
    nh_add_inplace_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->dE_tmp.data, dE_n);
    // ...and the discovered-type channel's: dE += dT_true @ invt_w.
    puf_mm_nn(&a->dTtrue, &ew->invt_w, &a->dE_tmp, stream);
    nh_add_inplace_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->dE_tmp.data, dE_n);

    // spell-embed channel: scatter its concat-grad slice into dE (reuse dE_i)
    cudaMemsetAsync(a->dE_i.data, 0, (size_t)NH_GLYPH_VOCAB * NH_EMBED_DIM * sizeof(long long), stream);
    nh_spkey_dk_kernel<<<grid_size(B * NH_SPELL_SLOTS * NH_SPKEY), BLOCK_SIZE, 0, stream>>>(
        a->spk_dkeys.data, grad_concat.data, a->spk_pool.data, a->spk_amax.data,
        ew->spk2_w.data, a->spk_keys.data,
        nh_ptr_spkeygrad != NULL ? nh_ptr_spkeygrad->data : NULL, B);
    nh_spk2_grad_kernel<<<grid_size(NH_SPKEY * (NH_SPKEY + 1)), BLOCK_SIZE, 0, stream>>>(
        a->spk2_wgrad.data, a->spk2_bgrad.data, grad_concat.data, a->spk_pool.data,
        a->spk_amax.data, a->spk_keys.data, B);
    nh_spkey_dE_kernel<<<grid_size(B * NH_SPELL_SLOTS * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dE_i.data, a->spk_dkeys.data, ew->spk_w.data, a->spell_idx.data, B);
    { Prec dkf = {.data = a->spk_dkeys.data, .shape = {B * NH_SPELL_SLOTS, NH_SPKEY}};
      Prec inf = {.data = a->spk_in.data, .shape = {B * NH_SPELL_SLOTS, NH_SPIN}};
      puf_mm_tn(&dkf, &inf, &a->spk_wgrad, stream); }
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->dE_tmp.data, (long long*)a->dE_i.data, NH_GLYPH_VOCAB * NH_EMBED_DIM);
    nh_add_inplace_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->dE_tmp.data, dE_n);

    // dE_eff complete: embed_wgrad doubles as dE_res (identity path); factor
    // tables get deterministic CSR row sums of the same buffer
    nh_ekind_grad_kernel<<<grid_size(NH_NKIND * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->ekind_wgrad.data, a->embed_wgrad.data);
    nh_esub_grad_kernel<<<grid_size(NH_NSUB * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->esub_wgrad.data, a->embed_wgrad.data);

    nh_bias_flush_kernel<<<grid_size(H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID + NH_INV_POOL), BLOCK_SIZE, 0, stream>>>(
        bacc, a->proj_bgrad.data, H, a->loc_bgrad.data, NH_LOC_HID,
        a->glb2_bgrad.data, NH_GLB_HID, a->bl_bgrad.data, NH_BL_HID,
        a->glb1_bgrad.data, NH_P1, a->inv1_bgrad.data, NH_INV_HID,
        a->inv2_bgrad.data, NH_INV_POOL);
}

static void nethack_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    // zero factors: E_eff == E_res at init, function-identical to the
    // unfactorized baseline; sharing grows in only where gradients ask
    cudaMemsetAsync(ew->ekind_w.data, 0, numel(ew->ekind_w.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->esub_w.data, 0, numel(ew->esub_w.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->loc_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->loc_b.data, 0, numel(ew->loc_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->glb1_w, 1.0f, (*seed)++, stream);
    // zero: kaiming's fan_in=2 would run 20x hotter than the glyph slice
    cudaMemsetAsync(ew->glb1_xy.data, 0, numel(ew->glb1_xy.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->glb1_b.data, 0, numel(ew->glb1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->glb2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->glb2_b.data, 0, numel(ew->glb2_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->inv1_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->inv1_b.data, 0, numel(ew->inv1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->inv1s_w, 1.0f, (*seed)++, stream);
    // zero: discovered-type channel starts as an exact no-op (ekind_w idiom)
    cudaMemsetAsync(ew->invt_w.data, 0, numel(ew->invt_w.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->inv2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->inv2_b.data, 0, numel(ew->inv2_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->bl_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->bl_b.data, 0, numel(ew->bl_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->proj_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->proj_b.data, 0, numel(ew->proj_b.shape) * sizeof(precision_t), stream);
    puf_normal_init(&ew->msg_w, 1.0f, (*seed)++, stream); // trigram embedding
    puf_kaiming_init(&ew->spk_w, 1.0f, (*seed)++, stream); // spell slot-rep projection
    puf_kaiming_init(&ew->spk2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->spk2_b.data, 0, numel(ew->spk2_b.shape) * sizeof(precision_t), stream);
#if NH_ID_EMBED
    // zero: the identity channel starts as an exact no-op (ekind_w idiom)
    cudaMemsetAsync(ew->ide_role_w.data, 0, numel(ew->ide_role_w.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->ide_race_w.data, 0, numel(ew->ide_race_w.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->ide_gend_w.data, 0, numel(ew->ide_gend_w.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->ide_algn_w.data, 0, numel(ew->ide_algn_w.shape) * sizeof(precision_t), stream);
#endif
}

// Param and grad registration orders must match pairwise (muon walks both flat).
static void nethack_encoder_reg_params(void* w, Allocator* alloc) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    ew->embed_w = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    ew->ekind_w = {.shape = {NH_NKIND, NH_EMBED_DIM}}; // 14x32=448, mult of 8
    ew->esub_w = {.shape = {NH_NSUB, NH_EMBED_DIM}}; // 944x32=30208, mult of 8
    ew->loc_w = {.shape = {NH_LOC_HID, NH_LOC_IN}};
    ew->loc_b = {.shape = {NH_LOC_HID}};
    ew->glb1_w = {.shape = {NH_P1, NH_PCELLS * NH_EMBED_DIM}};
    ew->glb1_xy = {.shape = {NH_P1, 2}};
    ew->glb1_b = {.shape = {NH_P1}};
    ew->glb2_w = {.shape = {NH_GLB_HID, NH_P1}};
    ew->glb2_b = {.shape = {NH_GLB_HID}};
    ew->inv1_w = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    ew->inv1_b = {.shape = {NH_INV_HID}};
    ew->inv1s_w = {.shape = {NH_INV_HID, NH_SFEAT}};
    ew->invt_w = {.shape = {NH_INV_HID, NH_EMBED_DIM}}; // 16x32=512, mult of 8
    ew->inv2_w = {.shape = {NH_INV_POOL, NH_INV_HID}};
    ew->inv2_b = {.shape = {NH_INV_POOL}};
    ew->bl_w = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    ew->bl_b = {.shape = {NH_BL_HID}};
    ew->proj_w = {.shape = {ew->hidden, NH_CONCAT}};
    ew->proj_b = {.shape = {ew->hidden}};
    ew->msg_w = {.shape = {NH_MSG_VOCAB, NH_MSG_HID}}; // 4096x32=131072, mult of 8
    ew->spk_w = {.shape = {NH_SPKEY, NH_SPIN}}; // 16x36=576, mult of 8
    ew->spk2_w = {.shape = {NH_SPKEY, NH_SPKEY}}; // 16x16=256, mult of 8
    ew->spk2_b = {.shape = {NH_SPKEY}}; // 16, mult of 8
#if NH_ID_EMBED
    ew->ide_role_w = {.shape = {13, NH_IDE_ROLE}}; // 208, mult of 8
    ew->ide_race_w = {.shape = {5, NH_IDE_RACE}}; // 40
    ew->ide_gend_w = {.shape = {2, NH_IDE_GEND}}; // 16
    ew->ide_algn_w = {.shape = {3, NH_IDE_ALGN}}; // 24
#endif
    alloc_register(alloc,&ew->embed_w);
    alloc_register(alloc,&ew->ekind_w); alloc_register(alloc,&ew->esub_w);
    alloc_register(alloc,&ew->loc_w);   alloc_register(alloc,&ew->loc_b);
    alloc_register(alloc,&ew->glb1_w);  alloc_register(alloc,&ew->glb1_xy);
    alloc_register(alloc,&ew->glb1_b);
    alloc_register(alloc,&ew->glb2_w);  alloc_register(alloc,&ew->glb2_b);
    alloc_register(alloc,&ew->inv1_w);  alloc_register(alloc,&ew->inv1_b);
    alloc_register(alloc,&ew->inv1s_w); alloc_register(alloc,&ew->invt_w);
    alloc_register(alloc,&ew->inv2_w);  alloc_register(alloc,&ew->inv2_b);
    alloc_register(alloc,&ew->bl_w);    alloc_register(alloc,&ew->bl_b);
    alloc_register(alloc,&ew->proj_w);  alloc_register(alloc,&ew->proj_b);
    alloc_register(alloc,&ew->msg_w);
    alloc_register(alloc,&ew->spk_w);
    alloc_register(alloc,&ew->spk2_w);  alloc_register(alloc,&ew->spk2_b);
#if NH_ID_EMBED
    alloc_register(alloc,&ew->ide_role_w); alloc_register(alloc,&ew->ide_race_w);
    alloc_register(alloc,&ew->ide_gend_w); alloc_register(alloc,&ew->ide_algn_w);
#endif
}

static void nethack_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    *a = {};
    a->glyph_idx = {.shape = {B_TT, NH_MGRID}};
    a->crop_glyph = {.shape = {B_TT, NH_CGRID}};
    a->e_eff = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->x_local = {.shape = {B_TT, NH_LOC_IN}};
    a->w_perm = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->glyph_T = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->t16 = {.shape = {B_TT, NH_TOK * NH_P1}};
    a->dxy = {.shape = {B_TT, NH_TOK * 2}};
    a->tok_argmax = {.shape = {B_TT, NH_GLB_HID}};
    a->inv_idx = {.shape = {B_TT, NH_INV}};
    a->spell_idx = {.shape = {B_TT, 8}};
    a->invt_idx = {.shape = {B_TT, NH_INV}};
    a->inv_sfeat = {.shape = {B_TT, NH_INV * NH_SFEAT}};
    a->inv_T = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->invt_T = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->inv_out = {.shape = {B_TT, NH_INV_FLAT}};
    a->inv_pool = {.shape = {B_TT, NH_INV_POOL}};
    a->inv_amax = {.shape = {B_TT, NH_INV_POOL}};
    a->loc_out = {.shape = {B_TT, NH_LOC_HID}};
    a->glb_out = {.shape = {B_TT, NH_GLB_HID}};
    a->bl_feats = {.shape = {B_TT, NH_BL_FEAT}};
    a->bl_out = {.shape = {B_TT, NH_BL_HID}};
    a->msg_ids = {.shape = {B_TT, NH_MSG_LEN}};
    a->msg_out = {.shape = {B_TT, NH_MSG_HID}};
    a->spk_in = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SPIN}};
    a->spk_keys = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SPKEY}};
    a->spk_dkeys = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SPKEY}};
    a->spk_amax = {.shape = {B_TT, NH_SPKEY}};
    a->spk_pool = {.shape = {B_TT, NH_SPKEY}};
#if NH_ID_EMBED
    a->ide_idx = {.shape = {B_TT, 4}};
#endif
    a->concat = {.shape = {B_TT, NH_CONCAT}};
    a->out = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts,&a->glyph_idx); alloc_register(acts,&a->crop_glyph);
    alloc_register(acts,&a->e_eff);
    alloc_register(acts,&a->x_local);
    alloc_register(acts,&a->w_perm);    alloc_register(acts,&a->glyph_T);
    alloc_register(acts,&a->t16);       alloc_register(acts,&a->dxy);
    alloc_register(acts,&a->tok_argmax);
    alloc_register(acts,&a->inv_idx);   alloc_register(acts,&a->invt_idx);
    alloc_register(acts,&a->spell_idx);
    alloc_register(acts,&a->spk_in);    alloc_register(acts,&a->spk_keys);
    alloc_register(acts,&a->spk_dkeys); alloc_register(acts,&a->spk_amax);
    alloc_register(acts,&a->spk_pool);
#if NH_ID_EMBED
    alloc_register(acts,&a->ide_idx);
#endif
    alloc_register(acts,&a->inv_sfeat);
    alloc_register(acts,&a->inv_T);     alloc_register(acts,&a->invt_T);
    alloc_register(acts,&a->inv_out);
    alloc_register(acts,&a->inv_pool);  alloc_register(acts,&a->inv_amax);
    alloc_register(acts,&a->loc_out);   alloc_register(acts,&a->glb_out);
    alloc_register(acts,&a->bl_feats);  alloc_register(acts,&a->bl_out);
    alloc_register(acts,&a->msg_ids);   alloc_register(acts,&a->msg_out);
    alloc_register(acts,&a->concat);    alloc_register(acts,&a->out);
    a->loc_grad = {.shape = {B_TT, NH_LOC_HID}};
    a->glb_grad = {.shape = {B_TT, NH_GLB_HID}};
    a->inv_grad = {.shape = {B_TT, NH_INV_FLAT}};
    a->inv_pool_grad = {.shape = {B_TT, NH_INV_POOL}};
    a->bl_grad = {.shape = {B_TT, NH_BL_HID}};
    a->dT = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dT_i = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->dTinv = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->dTinv_i = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->dTtrue = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->dTtrue_i = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->dE_tmp = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->dw_perm = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->dE_i = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->dw2_acc = {.shape = {NH_GLB_HID * NH_P1}};
    a->dw2i_acc = {.shape = {NH_INV_POOL * NH_INV_HID}};
    a->dmsg_acc = {.shape = {NH_MSG_VOCAB * NH_MSG_HID}};
    a->sort_local = {.shape = {2 * NH_GLYPH_VOCAB + NH_HOT_T + 1}};
    a->sort_grid = {.shape = {2 * NH_GLYPH_VOCAB + NH_HOT_G + 1}};
    a->bias_acc = {.shape = {ew->hidden + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID + NH_INV_POOL}};
    alloc_register(acts,&a->loc_grad);  alloc_register(acts,&a->glb_grad);
    alloc_register(acts,&a->inv_grad);  alloc_register(acts,&a->inv_pool_grad);
    alloc_register(acts,&a->bl_grad);
    alloc_register(acts,&a->dT);        alloc_register(acts,&a->dT_i);
    alloc_register(acts,&a->dTinv);     alloc_register(acts,&a->dTinv_i);
    alloc_register(acts,&a->dTtrue);    alloc_register(acts,&a->dTtrue_i);
    alloc_register(acts,&a->dE_tmp);
    alloc_register(acts,&a->dw_perm);   alloc_register(acts,&a->dE_i);
    alloc_register(acts,&a->dw2_acc);   alloc_register(acts,&a->dw2i_acc);
    alloc_register(acts,&a->dmsg_acc);
    alloc_register(acts,&a->sort_local); alloc_register(acts,&a->sort_grid);
    alloc_register(acts,&a->bias_acc);
    a->embed_wgrad = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->ekind_wgrad = {.shape = {NH_NKIND, NH_EMBED_DIM}};
    a->esub_wgrad = {.shape = {NH_NSUB, NH_EMBED_DIM}};
    a->loc_wgrad = {.shape = {NH_LOC_HID, NH_LOC_IN}};
    a->loc_bgrad = {.shape = {NH_LOC_HID}};
    a->glb1_wgrad = {.shape = {NH_P1, NH_PCELLS * NH_EMBED_DIM}};
    a->glb1_xygrad = {.shape = {NH_P1, 2}};
    a->glb1_bgrad = {.shape = {NH_P1}};
    a->glb2_wgrad = {.shape = {NH_GLB_HID, NH_P1}};
    a->glb2_bgrad = {.shape = {NH_GLB_HID}};
    a->inv1_wgrad = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    a->inv1_bgrad = {.shape = {NH_INV_HID}};
    a->inv1s_wgrad = {.shape = {NH_INV_HID, NH_SFEAT}};
    a->invt_wgrad = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    a->inv2_wgrad = {.shape = {NH_INV_POOL, NH_INV_HID}};
    a->inv2_bgrad = {.shape = {NH_INV_POOL}};
    a->bl_wgrad = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    a->bl_bgrad = {.shape = {NH_BL_HID}};
    a->proj_wgrad = {.shape = {ew->hidden, NH_CONCAT}};
    a->proj_bgrad = {.shape = {ew->hidden}};
    a->msg_wgrad = {.shape = {NH_MSG_VOCAB, NH_MSG_HID}};
    a->spk_wgrad = {.shape = {NH_SPKEY, NH_SPIN}};
    a->spk2_wgrad = {.shape = {NH_SPKEY, NH_SPKEY}};
    a->spk2_bgrad = {.shape = {NH_SPKEY}};
#if NH_ID_EMBED
    a->ide_role_wgrad = {.shape = {13, NH_IDE_ROLE}};
    a->ide_race_wgrad = {.shape = {5, NH_IDE_RACE}};
    a->ide_gend_wgrad = {.shape = {2, NH_IDE_GEND}};
    a->ide_algn_wgrad = {.shape = {3, NH_IDE_ALGN}};
#endif
    alloc_register(grads,&a->embed_wgrad);
    alloc_register(grads,&a->ekind_wgrad); alloc_register(grads,&a->esub_wgrad);
    alloc_register(grads,&a->loc_wgrad);   alloc_register(grads,&a->loc_bgrad);
    alloc_register(grads,&a->glb1_wgrad);  alloc_register(grads,&a->glb1_xygrad);
    alloc_register(grads,&a->glb1_bgrad);
    alloc_register(grads,&a->glb2_wgrad);  alloc_register(grads,&a->glb2_bgrad);
    alloc_register(grads,&a->inv1_wgrad);  alloc_register(grads,&a->inv1_bgrad);
    alloc_register(grads,&a->inv1s_wgrad); alloc_register(grads,&a->invt_wgrad);
    alloc_register(grads,&a->inv2_wgrad);  alloc_register(grads,&a->inv2_bgrad);
    alloc_register(grads,&a->bl_wgrad);    alloc_register(grads,&a->bl_bgrad);
    alloc_register(grads,&a->proj_wgrad);  alloc_register(grads,&a->proj_bgrad);
    alloc_register(grads,&a->msg_wgrad);
    alloc_register(grads,&a->spk_wgrad);
    alloc_register(grads,&a->spk2_wgrad); alloc_register(grads,&a->spk2_bgrad);
#if NH_ID_EMBED
    alloc_register(grads,&a->ide_role_wgrad); alloc_register(grads,&a->ide_race_wgrad);
    alloc_register(grads,&a->ide_gend_wgrad); alloc_register(grads,&a->ide_algn_wgrad);
#endif
    nh_enc_last = a;
}

static void nethack_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    a->glyph_idx = {.shape = {B, NH_MGRID}};
    a->crop_glyph = {.shape = {B, NH_CGRID}};
    a->e_eff = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->x_local = {.shape = {B, NH_LOC_IN}};
    a->w_perm = {.shape = {NH_TROW, NH_EMBED_DIM}};
    a->glyph_T = {.shape = {NH_GLYPH_VOCAB, NH_TROW}};
    a->t16 = {.shape = {B, NH_TOK * NH_P1}};
    a->dxy = {.shape = {B, NH_TOK * 2}};
    a->tok_argmax = {.shape = {B, NH_GLB_HID}};
    a->inv_idx = {.shape = {B, NH_INV}};
    a->invt_idx = {.shape = {B, NH_INV}};
    a->spell_idx = {.shape = {B, 8}};
    a->inv_sfeat = {.shape = {B, NH_INV * NH_SFEAT}};
    a->inv_T = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->invt_T = {.shape = {NH_GLYPH_VOCAB, NH_INV_HID}};
    a->inv_out = {.shape = {B, NH_INV_FLAT}};
    a->inv_pool = {.shape = {B, NH_INV_POOL}};
    a->inv_amax = {.shape = {B, NH_INV_POOL}};
    a->loc_out = {.shape = {B, NH_LOC_HID}};
    a->glb_out = {.shape = {B, NH_GLB_HID}};
    a->bl_feats = {.shape = {B, NH_BL_FEAT}};
    a->bl_out = {.shape = {B, NH_BL_HID}};
    a->msg_ids = {.shape = {B, NH_MSG_LEN}};
    a->msg_out = {.shape = {B, NH_MSG_HID}};
    a->spk_in = {.shape = {B, NH_SPELL_SLOTS * NH_SPIN}};
    a->spk_keys = {.shape = {B, NH_SPELL_SLOTS * NH_SPKEY}};
    a->spk_amax = {.shape = {B, NH_SPKEY}};
    a->spk_pool = {.shape = {B, NH_SPKEY}};
#if NH_ID_EMBED
    a->ide_idx = {.shape = {B, 4}};
#endif
    a->concat = {.shape = {B, NH_CONCAT}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc,&a->glyph_idx); alloc_register(alloc,&a->crop_glyph);
    alloc_register(alloc,&a->e_eff);
    alloc_register(alloc,&a->x_local);
    alloc_register(alloc,&a->w_perm);    alloc_register(alloc,&a->glyph_T);
    alloc_register(alloc,&a->t16);       alloc_register(alloc,&a->dxy);
    alloc_register(alloc,&a->tok_argmax);
    alloc_register(alloc,&a->inv_idx);   alloc_register(alloc,&a->invt_idx);
    alloc_register(alloc,&a->spell_idx);
#if NH_ID_EMBED
    alloc_register(alloc,&a->ide_idx);
#endif
    alloc_register(alloc,&a->inv_sfeat);
    alloc_register(alloc,&a->inv_T);     alloc_register(alloc,&a->invt_T);
    alloc_register(alloc,&a->inv_out);
    alloc_register(alloc,&a->inv_pool);  alloc_register(alloc,&a->inv_amax);
    alloc_register(alloc,&a->loc_out);   alloc_register(alloc,&a->glb_out);
    alloc_register(alloc,&a->bl_feats);  alloc_register(alloc,&a->bl_out);
    alloc_register(alloc,&a->msg_ids);   alloc_register(alloc,&a->msg_out);
    alloc_register(alloc,&a->spk_in);    alloc_register(alloc,&a->spk_keys);
    alloc_register(alloc,&a->spk_amax); alloc_register(alloc,&a->spk_pool);
    alloc_register(alloc,&a->concat);    alloc_register(alloc,&a->out);
    nh_enc_last = a;
}

static void* nethack_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return nethack_encoder_create(e->in_dim, e->out_dim);
}

static void create_nethack_encoder(Encoder* enc) {
    *enc = Encoder{
        .forward = nethack_encoder_forward,
        .backward = nethack_encoder_backward,
        .init_weights = nethack_encoder_init_weights,
        .reg_params = nethack_encoder_reg_params,
        .reg_train = nethack_encoder_reg_train,
        .reg_rollout = nethack_encoder_reg_rollout,
        .create_weights = nethack_encoder_create_weights,
        .in_dim = enc->in_dim, .out_dim = enc->out_dim,
        .activation_size = sizeof(NethackEncoderActivations),
    };
}

// decoder: per-verb pointer slot heads
// Output layout matches DefaultDecoder: [14 verb | 5x55 slots | 8 dir | value].
// verb/dir/value are one small linear. Each item verb (wear/eat/quaff/throw/
// zap) owns a query q_h = W_qh . hidden; slot logit i = exp(ltau_h) * cos(q_h, k_i)
// (log-parameterized temperature: raw tau crossing zero would NaN the
// backward, which divides logits by it)
// where k_i projects the inventory branch's post-relu slot vector. Keys are
// shared across heads, so the item->action mapping is position-invariant and
// every item use trains the same projections.

static constexpr int NH_DIRS = 8;
static constexpr int NH_DIRHEADS = 6; // move|run|kick|throw|zap|apply
static constexpr int NH_HEADS = 12; // wear|eat|quaff|throw|zap|takeoff|puton|remove|wield|apply|read|drop
static constexpr int NH_SLOT_OD = NH_HEADS * NH_INV; // 660 slot logits
static constexpr int NH_DEC_OD = NH_ACTIONS + NH_SLOT_OD + NH_DIRHEADS * NH_DIRS
                                + NH_SPELL_SLOTS; // logits: verbs|slots|dirs|spell
static constexpr int NH_DEC_LIN = NH_ACTIONS + NH_DIRHEADS * NH_DIRS + 1; // verbs|dirs|value
static constexpr int NH_DEC_PAD = (NH_DEC_LIN + 7) / 8 * 8; // lin rows padded to mult of 8 (cublasLt alignment)
// queries: 12 cosine inv heads + 1 dot-product spell head (row NH_HEADS)
static constexpr int NH_QHEADS = NH_HEADS + 1;
static constexpr int NH_QDIM = NH_QHEADS * NH_INV_HID; // stacked queries
static constexpr int NH_SPELL_BASE = NH_ACTIONS + NH_SLOT_OD + NH_DIRHEADS * NH_DIRS;
// tau is padded to 8 entries (first NH_HEADS live): checkpoints are saved
// compactly and the puffernet loader assumes every tensor is a multiple of
// 8 floats (16-byte bf16 alignment). Pad slots are dead but NOT frozen
// (optimizer weight decay can drift them from init) — never read.
static constexpr int NH_TAU_PAD = 16;

struct NethackDecoderWeights {
    // Header mirrors DecoderWeights EXACTLY: the framework casts decoder
    // weights to DecoderWeights to read .continuous / .logstd when deciding
    // discrete-vs-continuous sampling (pufferlib.cu sample + train sites).
    // weight_unused is never registered; logstd stays null; continuous false.
    Prec weight_unused, logstd;
    int hidden_dim, output_dim;
    bool continuous;
    // pointer-head weights (v3: per-verb queries, shared cosine keys)
    Prec lin_w; // (NH_DEC_PAD rows, hidden); first NH_DEC_LIN used
    Prec q_w; // (NH_QDIM, hidden) stacked per-head query projections
    Prec k_w; // (NH_INV_HID, NH_INV_HID) key projection over inv features
    Prec tau; // (NH_TAU_PAD,) learnable LOG temperatures, first NH_HEADS live
};

struct NethackDecoderActivations {
    NethackEncoderActivations* enc; // partner encoder acts (keys source)
    Prec out; // (B, NH_DEC_OD+1)
    Prec tmp, q; // (B, NH_DEC_PAD), (B, NH_QDIM)
    Prec saved_input, grad_input, grad_input2;
    Prec grad_out; // assembled logits+value grad
    Prec dtmp, dq;
    Prec keygrad; // (B, NH_INV_FLAT) -> encoder inv slice
    Prec kmat; // (B, NH_INV_FLAT) projected keys
    Prec kn, qn; // key norms (B, NH_INV), query norms (B, NH_QHEADS)
    Prec slot_logits; // (B, NH_SLOT_OD) tau_h * cos
    Prec dkmat; // backward scratch
    Prec spdk; // spell-key grads from the pointer (B, 8*NH_SPKEY)
    Long tau_acc; // fixed-point dtau staging (NH_TAU_PAD,)
    Prec lin_wgrad, q_wgrad, k_wgrad, tau_grad;
};

__global__ void nh_dec_assemble_kernel(precision_t* __restrict__ out,
    const precision_t* __restrict__ tmp, const precision_t* __restrict__ slot_logits,
    const precision_t* __restrict__ q, const precision_t* __restrict__ spkeys, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int od1 = NH_DEC_OD + 1;
    if (idx >= B * od1) return;
    int b = idx / od1, c = idx % od1;
    float v;
    if (c < NH_ACTIONS) v = to_float(tmp[(int64_t)b * NH_DEC_PAD + c]);
    else if (c < NH_ACTIONS + NH_SLOT_OD)
        v = to_float(slot_logits[(int64_t)b * NH_SLOT_OD + (c - NH_ACTIONS)]);
    else if (c < NH_SPELL_BASE) // per-verb dir rows from lin
        v = to_float(tmp[(int64_t)b * NH_DEC_PAD + NH_ACTIONS + (c - NH_ACTIONS - NH_SLOT_OD)]);
    else if (c < NH_SPELL_BASE + NH_SPELL_SLOTS) {
        // spell head: dot(q_spell, key_s) / sqrt(keydim)
        int s = c - NH_SPELL_BASE;
        const precision_t* qs = q + ((int64_t)b * NH_QHEADS + NH_HEADS) * NH_INV_HID;
        const precision_t* ks = spkeys + ((int64_t)b * NH_SPELL_SLOTS + s) * NH_SPKEY;
        float dot = 0.0f;
        for (int k = 0; k < NH_SPKEY; k++) dot += to_float(qs[k]) * to_float(ks[k]);
        v = dot * 0.25f;
    } else // value
        v = to_float(tmp[(int64_t)b * NH_DEC_PAD + NH_DEC_LIN - 1]);
    out[idx] = from_float(v);
}

// L2 norm (+eps) of each 32-dim row; runs over queries (B*NH_HEADS rows) and
// projected keys (B*NH_INV rows)
__global__ void nh_ptr_rownorm_kernel(precision_t* __restrict__ n,
    const precision_t* __restrict__ rows, int total) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total) return;
    float acc = 0.0f;
    for (int k = 0; k < NH_INV_HID; k++) {
        float v = to_float(rows[(int64_t)r * NH_INV_HID + k]);
        acc += v * v;
    }
    n[r] = from_float(sqrtf(acc) + 1e-6f);
}

// slot logit = exp(ltau_h) * cos(q_h, k_i), one thread per (sample, head, slot)
__global__ void nh_ptr3_cos_kernel(precision_t* __restrict__ slot_logits,
    const precision_t* __restrict__ q, const precision_t* __restrict__ qn,
    const precision_t* __restrict__ kmat, const precision_t* __restrict__ kn,
    const precision_t* __restrict__ tau, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_SLOT_OD) return;
    int b = idx / NH_SLOT_OD, hi = idx % NH_SLOT_OD;
    int h = hi / NH_INV, i = hi % NH_INV;
    const precision_t* qb = q + ((int64_t)b * NH_QHEADS + h) * NH_INV_HID;
    const precision_t* ki = kmat + ((int64_t)b * NH_INV + i) * NH_INV_HID;
    float dot = 0.0f;
    for (int k = 0; k < NH_INV_HID; k++)
        dot += to_float(qb[k]) * to_float(ki[k]);
    slot_logits[idx] = from_float(expf(to_float(tau[h])) * dot /
        (to_float(qn[(int64_t)b * NH_QHEADS + h]) * to_float(kn[(int64_t)b * NH_INV + i])));
}

__global__ void nh_dec_dtmp_kernel(precision_t* __restrict__ dtmp,
    const precision_t* __restrict__ g, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_DEC_PAD) return;
    int b = idx / NH_DEC_PAD, c = idx % NH_DEC_PAD;
    if (c >= NH_DEC_LIN) { // pad rows
        dtmp[idx] = from_float(0.0f);
        return;
    }
    // lin rows: verbs, dirs, then value (spell logits are pointer-derived)
    int src = c < NH_ACTIONS ? c
            : c < NH_DEC_LIN - 1 ? NH_ACTIONS + NH_SLOT_OD + (c - NH_ACTIONS)
            : NH_DEC_OD;
    dtmp[idx] = g[(int64_t)b * (NH_DEC_OD + 1) + src];
}

// dltau_h (fxp scalar, chain rule: dL/dltau = tau * sum g cos) + dv = tau *
// sum_i g_hi * u_i (u = normalized key); dq_h = (dv - v (v.dv)) / ||q_h||,
// per (sample, head) with sequential loops.
__global__ void nh_ptr3_dq_kernel(precision_t* __restrict__ dq,
    long long* __restrict__ tau_acc, const precision_t* __restrict__ g,
    const precision_t* __restrict__ out, const precision_t* __restrict__ q,
    const precision_t* __restrict__ qn, const precision_t* __restrict__ kmat,
    const precision_t* __restrict__ kn, const precision_t* __restrict__ tau, int B) {
    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * NH_HEADS) return;
    int b = bh / NH_HEADS, h = bh % NH_HEADS;
    int64_t qrow = (int64_t)b * NH_QHEADS + h;
    float tauv = expf(to_float(tau[h]));
    float qnv = to_float(qn[qrow]);
    float vhat[NH_INV_HID], dv[NH_INV_HID];
    for (int k = 0; k < NH_INV_HID; k++) {
        vhat[k] = to_float(q[qrow * NH_INV_HID + k]) / qnv;
        dv[k] = 0.0f;
    }
    float dtau = 0.0f;
    const int64_t gbase = (int64_t)b * (NH_DEC_OD + 1) + NH_ACTIONS + h * NH_INV;
    for (int i = 0; i < NH_INV; i++) {
        float gi = to_float(g[gbase + i]);
        if (gi == 0.0f) continue;
        float cosv = to_float(out[gbase + i]) / tauv;
        dtau += gi * cosv;
        float knv = to_float(kn[(int64_t)b * NH_INV + i]);
        const precision_t* ki = kmat + ((int64_t)b * NH_INV + i) * NH_INV_HID;
        for (int k = 0; k < NH_INV_HID; k++)
            dv[k] += tauv * gi * to_float(ki[k]) / knv;
    }
    float vdv = 0.0f;
    for (int k = 0; k < NH_INV_HID; k++) vdv += vhat[k] * dv[k];
    for (int k = 0; k < NH_INV_HID; k++)
        dq[qrow * NH_INV_HID + k] = from_float((dv[k] - vhat[k] * vdv) / qnv);
    if (dtau != 0.0f) nh_fxp_atomic_add(&tau_acc[h], dtau * tauv);
}

// spell head backward (dot-product pointer): dq_spell = 0.25 sum_s g_s k_s,
// spdk_s = 0.25 g_s q_spell (+ the pool grad, added encoder-side). Always
// writes its dq row and all spdk entries (zeros when CAST saw no gradient).
__global__ void nh_spq_bwd_kernel(precision_t* __restrict__ dq,
    precision_t* __restrict__ spdk, const precision_t* __restrict__ g,
    const precision_t* __restrict__ q, const precision_t* __restrict__ spkeys, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPKEY) return;
    int b = t / NH_SPKEY, k = t % NH_SPKEY;
    int64_t qrow = ((int64_t)b * NH_QHEADS + NH_HEADS) * NH_INV_HID;
    float qk = to_float(q[qrow + k]);
    float dqk = 0.0f;
    for (int s = 0; s < NH_SPELL_SLOTS; s++) {
        float gs = to_float(g[(int64_t)b * (NH_DEC_OD + 1) + NH_SPELL_BASE + s]);
        dqk += 0.25f * gs * to_float(spkeys[((int64_t)b * NH_SPELL_SLOTS + s) * NH_SPKEY + k]);
        spdk[((int64_t)b * NH_SPELL_SLOTS + s) * NH_SPKEY + k] = from_float(0.25f * gs * qk);
    }
    dq[qrow + k] = from_float(dqk);
}

// dk_i = sum_h tau_h * g_hi * (v_h - u_i * cos_hi) / ||k_i||
__global__ void nh_ptr3_dkmat_kernel(precision_t* __restrict__ dkmat,
    const precision_t* __restrict__ g, const precision_t* __restrict__ out,
    const precision_t* __restrict__ q, const precision_t* __restrict__ qn,
    const precision_t* __restrict__ kmat, const precision_t* __restrict__ kn,
    const precision_t* __restrict__ tau, int B) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * NH_INV_FLAT) return;
    int64_t bi = idx / NH_INV_HID;
    int64_t b = bi / NH_INV;
    int i = (int)(bi % NH_INV);
    int k = (int)(idx % NH_INV_HID);
    float knv = to_float(kn[bi]);
    float uk = to_float(kmat[idx]) / knv;
    float acc = 0.0f;
    for (int h = 0; h < NH_HEADS; h++) {
        int64_t gi_idx = b * (NH_DEC_OD + 1) + NH_ACTIONS + h * NH_INV + i;
        float gi = to_float(g[gi_idx]);
        if (gi == 0.0f) continue;
        float tauv = expf(to_float(tau[h]));
        float cosv = to_float(out[gi_idx]) / tauv;
        float vk = to_float(q[((int64_t)b * NH_QHEADS + h) * NH_INV_HID + k])
                 / to_float(qn[(int64_t)b * NH_QHEADS + h]);
        acc += tauv * gi * (vk - uk * cosv);
    }
    dkmat[idx] = from_float(acc / knv);
}

static Prec nethack_decoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    NethackDecoderWeights* dw = (NethackDecoderWeights*)w;
    NethackDecoderActivations* a = (NethackDecoderActivations*)activations;
    int B = input.shape[0];
    NethackEncoderActivations* ea = a->enc;
    if (a->saved_input.data) puf_copy(&a->saved_input, &input, stream);
    puf_mm(&input, &dw->lin_w, &a->tmp, stream);
    puf_mm(&input, &dw->q_w, &a->q, stream);
    Prec sflat = {.data = ea->inv_out.data, .shape = {B * NH_INV, NH_INV_HID}};
    Prec kflat = {.data = a->kmat.data, .shape = {B * NH_INV, NH_INV_HID}};
    puf_mm(&sflat, &dw->k_w, &kflat, stream);
    nh_ptr_rownorm_kernel<<<grid_size(B * NH_QHEADS), BLOCK_SIZE, 0, stream>>>(
        a->qn.data, a->q.data, B * NH_QHEADS);
    nh_ptr_rownorm_kernel<<<grid_size(B * NH_INV), BLOCK_SIZE, 0, stream>>>(
        a->kn.data, a->kmat.data, B * NH_INV);
    nh_ptr3_cos_kernel<<<grid_size(B * NH_SLOT_OD), BLOCK_SIZE, 0, stream>>>(
        a->slot_logits.data, a->q.data, a->qn.data, a->kmat.data, a->kn.data, dw->tau.data, B);
    nh_dec_assemble_kernel<<<grid_size(B * (NH_DEC_OD + 1)), BLOCK_SIZE, 0, stream>>>(
        a->out.data, a->tmp.data, a->slot_logits.data, a->q.data, ea->spk_keys.data, B);
    return a->out;
}

static Prec nethack_decoder_backward(void* w, void* activations,
    Float grad_logits, Float grad_logstd, Float grad_value, cudaStream_t stream) {
    NethackDecoderWeights* dw = (NethackDecoderWeights*)w;
    NethackDecoderActivations* a = (NethackDecoderActivations*)activations;
    int B = a->saved_input.shape[0];
    NethackEncoderActivations* ea = a->enc;
    assemble_decoder_grad<<<grid_size(B * (NH_DEC_OD + 1)), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data, B, NH_DEC_OD, NH_DEC_OD + 1);
    nh_dec_dtmp_kernel<<<grid_size(B * NH_DEC_PAD), BLOCK_SIZE, 0, stream>>>(
        a->dtmp.data, a->grad_out.data, B);
    cudaMemsetAsync(a->tau_acc.data, 0, NH_TAU_PAD * sizeof(long long), stream);
    nh_ptr3_dq_kernel<<<grid_size(B * NH_HEADS), BLOCK_SIZE, 0, stream>>>(
        a->dq.data, (long long*)a->tau_acc.data, a->grad_out.data, a->out.data,
        a->q.data, a->qn.data, a->kmat.data, a->kn.data, dw->tau.data, B);
    nh_fxp_to_precision_kernel<<<1, 32, 0, stream>>>(
        a->tau_grad.data, (long long*)a->tau_acc.data, NH_TAU_PAD);
    nh_spq_bwd_kernel<<<grid_size(B * NH_SPKEY), BLOCK_SIZE, 0, stream>>>(
        a->dq.data, a->spdk.data, a->grad_out.data, a->q.data, ea->spk_keys.data, B);
    nh_ptr3_dkmat_kernel<<<grid_size((int64_t)B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->dkmat.data, a->grad_out.data, a->out.data, a->q.data, a->qn.data,
        a->kmat.data, a->kn.data, dw->tau.data, B);
    // dK = dkmat^T @ s ; keygrad (ds, into the encoder inv slice) = dkmat @ K
    Prec dkflat = {.data = a->dkmat.data, .shape = {B * NH_INV, NH_INV_HID}};
    Prec sflat = {.data = ea->inv_out.data, .shape = {B * NH_INV, NH_INV_HID}};
    Prec kgflat = {.data = a->keygrad.data, .shape = {B * NH_INV, NH_INV_HID}};
    puf_mm_tn(&dkflat, &sflat, &a->k_wgrad, stream);
    puf_mm_nn(&dkflat, &dw->k_w, &kgflat, stream);
    puf_mm_tn(&a->dtmp, &a->saved_input, &a->lin_wgrad, stream);
    puf_mm_tn(&a->dq, &a->saved_input, &a->q_wgrad, stream);
    puf_mm_nn(&a->dtmp, &dw->lin_w, &a->grad_input, stream);
    puf_mm_nn(&a->dq, &dw->q_w, &a->grad_input2, stream);
    nh_add_inplace_kernel<<<grid_size(B * dw->hidden_dim), BLOCK_SIZE, 0, stream>>>(
        a->grad_input.data, a->grad_input2.data, B * dw->hidden_dim);
    return a->grad_input;
}

__global__ void nh_fill_kernel(precision_t* p, float v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = from_float(v);
}

static void nethack_decoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NethackDecoderWeights* dw = (NethackDecoderWeights*)w;
    puf_kaiming_init(&dw->lin_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&dw->q_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&dw->k_w, 1.0f, (*seed)++, stream);
    nh_fill_kernel<<<1, 32, 0, stream>>>(dw->tau.data, logf(10.0f), NH_TAU_PAD);
}

static void nethack_decoder_reg_params(void* w, Allocator* alloc) {
    NethackDecoderWeights* dw = (NethackDecoderWeights*)w;
    dw->lin_w = {.shape = {NH_DEC_PAD, dw->hidden_dim}};
    dw->q_w = {.shape = {NH_QDIM, dw->hidden_dim}};
    dw->k_w = {.shape = {NH_INV_HID, NH_INV_HID}};
    dw->tau = {.shape = {NH_TAU_PAD}};
    alloc_register(alloc,&dw->lin_w);
    alloc_register(alloc,&dw->q_w);
    alloc_register(alloc,&dw->k_w);
    alloc_register(alloc,&dw->tau);
}

static void nethack_decoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NethackDecoderWeights* dw = (NethackDecoderWeights*)w;
    NethackDecoderActivations* a = (NethackDecoderActivations*)activations;
    *a = {};
    a->out = {.shape = {B_TT, NH_DEC_OD + 1}};
    a->tmp = {.shape = {B_TT, NH_DEC_PAD}};
    a->q = {.shape = {B_TT, NH_QDIM}};
    a->saved_input = {.shape = {B_TT, dw->hidden_dim}};
    a->grad_input = {.shape = {B_TT, dw->hidden_dim}};
    a->grad_input2 = {.shape = {B_TT, dw->hidden_dim}};
    a->grad_out = {.shape = {B_TT, NH_DEC_OD + 1}};
    a->dtmp = {.shape = {B_TT, NH_DEC_PAD}};
    a->dq = {.shape = {B_TT, NH_QDIM}};
    a->keygrad = {.shape = {B_TT, NH_INV_FLAT}};
    a->kmat = {.shape = {B_TT, NH_INV_FLAT}};
    a->kn = {.shape = {B_TT, NH_INV}};
    a->qn = {.shape = {B_TT, NH_QHEADS}};
    a->slot_logits = {.shape = {B_TT, NH_SLOT_OD}};
    a->dkmat = {.shape = {B_TT, NH_INV_FLAT}};
    a->spdk = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SPKEY}};
    a->tau_acc = {.shape = {NH_TAU_PAD}};
    a->lin_wgrad = {.shape = {NH_DEC_PAD, dw->hidden_dim}};
    a->q_wgrad = {.shape = {NH_QDIM, dw->hidden_dim}};
    a->k_wgrad = {.shape = {NH_INV_HID, NH_INV_HID}};
    a->tau_grad = {.shape = {NH_TAU_PAD}};
    alloc_register(acts,&a->out);         alloc_register(acts,&a->tmp);
    alloc_register(acts,&a->q);           alloc_register(acts,&a->saved_input);
    alloc_register(acts,&a->grad_input);  alloc_register(acts,&a->grad_input2);
    alloc_register(acts,&a->grad_out);    alloc_register(acts,&a->dtmp);
    alloc_register(acts,&a->dq);          alloc_register(acts,&a->keygrad);
    alloc_register(acts,&a->kmat);        alloc_register(acts,&a->kn);
    alloc_register(acts,&a->qn);          alloc_register(acts,&a->slot_logits);
    alloc_register(acts,&a->dkmat);       alloc_register(acts,&a->tau_acc);
    alloc_register(acts,&a->spdk);
    alloc_register(grads,&a->lin_wgrad);  alloc_register(grads,&a->q_wgrad);
    alloc_register(grads,&a->k_wgrad);    alloc_register(grads,&a->tau_grad);
    a->enc = nh_enc_last;
    nh_ptr_keygrad = &a->keygrad;
    nh_ptr_spkeygrad = &a->spdk;
}

static void nethack_decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NethackDecoderActivations* a = (NethackDecoderActivations*)activations;
    a->enc = nh_enc_last;
    a->out = {.shape = {B, NH_DEC_OD + 1}};
    a->tmp = {.shape = {B, NH_DEC_PAD}};
    a->q = {.shape = {B, NH_QDIM}};
    a->kmat = {.shape = {B, NH_INV_FLAT}};
    a->kn = {.shape = {B, NH_INV}};
    a->qn = {.shape = {B, NH_QHEADS}};
    a->slot_logits = {.shape = {B, NH_SLOT_OD}};
    alloc_register(alloc,&a->out);
    alloc_register(alloc,&a->tmp);
    alloc_register(alloc,&a->q);
    alloc_register(alloc,&a->kmat);
    alloc_register(alloc,&a->kn);
    alloc_register(alloc,&a->qn);
    alloc_register(alloc,&a->slot_logits);
}

// The framework casts decoder weights to DecoderWeights to read
// .continuous/.logstd (sampling + train sites). Any custom decoder's weights
// struct MUST lead with an identical header — enforce it at compile time.
static_assert(offsetof(NethackDecoderWeights, logstd) == offsetof(DecoderWeights, logstd),
              "NethackDecoderWeights header must mirror DecoderWeights (logstd)");
static_assert(offsetof(NethackDecoderWeights, continuous) == offsetof(DecoderWeights, continuous),
              "NethackDecoderWeights header must mirror DecoderWeights (continuous)");

static void* nethack_decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    if (d->output_dim != NH_DEC_OD) {
        fprintf(stderr, "nethack decoder: output_dim %d != expected %d\n",
                d->output_dim, NH_DEC_OD);
        exit(1);
    }
    NethackDecoderWeights* dw = (NethackDecoderWeights*)calloc(1, sizeof(NethackDecoderWeights));
    dw->hidden_dim = d->hidden_dim;
    dw->output_dim = d->output_dim;
    dw->continuous = false;
    return dw;
}

static void create_nethack_decoder(Decoder* dec) {
    *dec = Decoder{
        .forward = nethack_decoder_forward,
        .backward = nethack_decoder_backward,
        .init_weights = nethack_decoder_init_weights,
        .reg_params = nethack_decoder_reg_params,
        .reg_train = nethack_decoder_reg_train,
        .reg_rollout = nethack_decoder_reg_rollout,
        .create_weights = nethack_decoder_create_weights,
        .hidden_dim = dec->hidden_dim, .output_dim = dec->output_dim,
        .continuous = dec->continuous,
        .activation_size = (int)sizeof(NethackDecoderActivations),
    };
}
