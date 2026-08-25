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
static constexpr int NH_TERRC_DIM = 8; // v3 crop terrain code width (own tiny table)
static constexpr int NH_RING = 9; // v3 3x3 adjacency ring, full-width monster rows
static constexpr int NH_CROP = 9, NH_CHALF = 4; // NETHACK_CROP, egocentric
static constexpr int NH_CGRID = NH_CROP * NH_CROP;
static constexpr int NH_PW = 5, NH_PH = 5; // patch size (cells)
static constexpr int NH_PX = 16, NH_PY = 5; // patch grid (ceil 79/5, 21/5)
static constexpr int NH_TOK = NH_PX * NH_PY; // 80 global tokens
static constexpr int NH_PCELLS = NH_PW * NH_PH; // cells per patch (off-map -> pad glyph)
// v5 champion-candidate (ENCODER_V5.md): NH_LAB base + five gated changes,
// each independently settable for the per-change screen.
// NH_LOC3: 9-class semantic crop (croplab: danger .998, esc-dir .83;
// classes ~= depth, both help; width flat 128-512 -> 256)
static constexpr int NH_LOCC_DIM = 8;
static constexpr int NH_LOCC_CLASSES = 9; // wall floor door hazard mon item stone other offmap
// v3 local: terrain field (81x8, own table) + adjacency ring (9x32, T_mon)
// + underfoot item (32, T_item). Positions share "what", loc_w keeps "where".
static constexpr int NH_LOC_IN = NH_CGRID * NH_LOCC_DIM;
static constexpr int NH_LOC_HID = 256;
// two-layer local branch (crop -> NH_LOC_H1 -> NH_LOC_HID) so bearing
// at 2-4 tiles has a nonlinearity to be represented in (audit: 0.63 decodable
// from the single linear+relu, 0.39 in the direction logits). NH_LOC3 forces
// the two-layer path at hidden 256.
static constexpr int NH_LOC_H1 = 256; // hidden width; loc_w is (NH_LOC_H1, NH_LOC_IN)
// NH_TERR: landmark table (12x4) + sector radar (8x4x17) -> MLP (terrain lab)
static constexpr int NH_TERRF = 12 * 4 + 8 * 4 * 17; // 592 features
static constexpr int NH_TERR_H1 = 256;
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
// typed token streams (supervised-lab locked design, 2026-08). Typed K-nearest token
// streams (monsters, items; lean NH_TOK_OBS lists env-side) with deep values
// (2-layer MLP), 8-head attention pools scoring from raw token features, and
// a rank feature; inventory gets the 8-query attention pool (NH_INV_ATTN);
// aux mega-spec heads regress ~26 engine-truth targets off the concat
// (training-time only shaping; targets ride the NH_AUX_OBS extras block and
// are never featurized). All additive beside the champion branches.
// component toggles for leave-one-out ablation
static constexpr int NH_BL_RAW = 27; // NLE_BLSTATS_SIZE
static constexpr int NH_BL_HUNGER = 21, NH_BL_CONDITION = 25;
static constexpr int NH_BL_HP = 10, NH_BL_ENE = 14; // hp/hpmax at 10/11, ene/enemax at 14/15
static constexpr int NH_ACTIONS = 26; // NETHACK_NUM_ACTIONS
static constexpr int NH_OCLASSES = 18; // MAXOCLASSES
static constexpr int NH_EXTRA_SHOP = 2 + NH_OCLASSES; // extra[] index of the shop pair
static constexpr int NH_SPELL_SLOTS = 8; // NETHACK_SPELL_SLOTS
static constexpr int NH_EX_ROLEOH = 2 + NH_OCLASSES + 2 + 1 + 4 * NH_SPELL_SLOTS + 2;
static constexpr int NH_EX_RAW = NH_EX_ROLEOH + 13 + 5 + 2
                               + 1
; // NETHACK_EXTRA_INTS (+role/race/gender one-hots)
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
static constexpr int NH_F_INTRINS = NH_F_ROLE + 20; // 8 intrinsic bits
// threat block: visible bit, proximity, bearing 8-onehot, adj/near3/vis
// counts, nearest difficulty, difficulty-vs-xplvl, nearest speed
static constexpr int NH_F_THREAT = NH_F_INTRINS + 8;
static constexpr int NH_BL_FEAT = NH_F_THREAT;
static constexpr int NH_EX_INTRINS = NH_EX_ROLEOH + 20;
static constexpr int NH_EX_THREAT = NH_EX_INTRINS + 1;
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
// NH_INV2 drops the max-pool trunk summary (half-dead in production);
// its slice leaves the concat entirely.
static constexpr int NH_INVP_DIM = 0;
static constexpr int NH_MSG_CONCAT_OFF = NH_LOC_HID + NH_GLB_HID + NH_INVP_DIM + NH_BL_HID + NH_BL_FEAT;
// spell-key path (v3 pointer): per slot, key = spk_w . [e_eff(book glyph) |
// known, lev/7, fail/100, know/20000]; keys feed the CAST pointer head and a
// sum-pooled 16-dim trunk summary. Empty slots are exact zeros end to end.
static constexpr int NH_SPKEY = NH_INV_HID; // 16, shared key width
static constexpr int NH_SPIN = NH_EMBED_DIM + 4; // 36 key inputs/slot
// NH_SPELL2 (v5.1): trunk spell summary = isum32 over the keys (masked sum
// x0.2) + 4 exact doorstep scalars [min_fail, max_lev, n/8, min_retention];
// replaces the spk2 max-pool (proven to destroy spell info: identity AUC
// .18-.24 pooled vs .97-1.0 in the keys). Keys + CAST pointer untouched.
static constexpr int NH_SP2_DIM = 32;
static constexpr int NH_SPELL_SLICE = NH_SP2_DIM + 4;
static constexpr int NH_SPELL_CONCAT_OFF = NH_MSG_CONCAT_OFF + NH_MSG_HID;
// identity embeddings: explicit role/race/gend/align tables in a direct concat channel
static constexpr int NH_IDE_ROLE = 16, NH_IDE_RACE = 8;
static constexpr int NH_IDE_GEND = 8, NH_IDE_ALGN = 8;
static constexpr int NH_IDE_DIM = NH_IDE_ROLE + NH_IDE_RACE + NH_IDE_GEND + NH_IDE_ALGN;
static constexpr int NH_IDE_CONCAT_OFF = NH_SPELL_CONCAT_OFF + NH_SPELL_SLICE;
static constexpr int NH_GMEAN_CONCAT_OFF = NH_IDE_CONCAT_OFF + NH_IDE_DIM;
static constexpr int NH_TERR_VOCAB = 128; // cmap_index+1; 0 unseen; 127 off-map pad
static constexpr int NH_ENT_K = 32;
static constexpr int NH_ENT_F = 6;
static constexpr int NH_ENT_IN = 40;
static constexpr int NH_ENT_HID = 32;
static constexpr int NH_ENT_TAIL = 36;
// v3 typed-level: glyph ranges, tables, two K-nearest token lists
static constexpr int NH_NUMMONS = 381, NH_PET_OFF = 381, NH_DET_OFF = 762;
static constexpr int NH_BODY_OFF = 1144, NH_OBJ_LO = 1906;
static constexpr int NH_MON_ROWS = 384;   // species+1 (0 = pad/empty)
static constexpr int NH_ITEM_ROWS = 840;  // objects 1..453, bodies 454..834, 0 = pad
static constexpr int NH_V3K = 16;
static constexpr int NH_V3_MONF = 8, NH_V3_ITEMF = 8;
static constexpr int NH_V3_IN = 40, NH_V3_HID = 32, NH_V3_TAIL = 36;
static constexpr int NH_ITBL = NH_GLYPH_VOCAB; // inv/discovery table rows
__global__ void nh_scale_kernel(precision_t* p, float a, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = from_float(to_float(p[i]) * a);
}
static constexpr int NH_AP_DIM = NH_P1 + 2;
static constexpr int NH_TK_K = 8;
static constexpr int NH_TK_DIM = NH_P1 + 2; // gated rep + (dx,dy)
static constexpr int NH_ENT_CONCAT_OFF = NH_GMEAN_CONCAT_OFF;
static constexpr int NH_IVA_M = 8; // invattn query heads (invattn8)
static constexpr int NH_IVA_CONCAT_OFF = NH_ENT_CONCAT_OFF;
// lab stream dims: 16 tokens x 48 inputs (emb 32 + dx dy cheb rank + flags +
// diff speed), deep values 48 -> 64 -> 64, 8 heads x 8 dims
static constexpr int NH_LABK = 16;   // NETHACK_V3_K
static constexpr int NH_LAB_IN = 48;
static constexpr int NH_LAB_HID = 64;
static constexpr int NH_LAB_HEADS = 8;
static constexpr int NH_AUXH = 32;   // aux head rows (26 used, %8 pad)
static constexpr float NH_AUX_COEF = 0.05f;
static constexpr int NH_LABM_CONCAT_OFF = NH_IVA_CONCAT_OFF + NH_IVA_M * NH_INV_HID;
static constexpr int NH_LABI_CONCAT_OFF = NH_LABM_CONCAT_OFF + NH_LAB_HID;
// NH_INV2 tail: hard wield readout (16, parameterless) + sum channel (64)
static constexpr int NH_WLD_CONCAT_OFF = NH_LABI_CONCAT_OFF + NH_LAB_HID;
static constexpr int NH_ISUM_DIM = 64;
static constexpr int NH_ISUM_CONCAT_OFF = NH_WLD_CONCAT_OFF + NH_INV_HID;
static constexpr int NH_CONCAT = NH_ISUM_CONCAT_OFF + NH_ISUM_DIM;
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
static constexpr int NH_OBS_SIZE = NH_MSG_OFF + NH_MSG_LEN

                                 + NH_V3K * (NH_V3_MONF + NH_V3_ITEMF);
static constexpr int NH_TERR_OFF = NH_MSG_OFF + NH_MSG_LEN;
static constexpr int NH_ENTL_OFF = NH_TERR_OFF + NH_MGRID;
static constexpr int NH_OBJM_OFF = NH_TERR_OFF + NH_MGRID;      // v3 (aliases ENTL; exclusive)
static constexpr int NH_VMON_OFF = NH_OBJM_OFF + NH_MGRID * 2;
static constexpr int NH_VITEM_OFF = NH_VMON_OFF + NH_V3K * NH_V3_MONF;
// lean token lists (NH_LAB / env NH_TOK_OBS): appended after every other
// optional block, no dense planes (netlib.h NETHACK_OFF_TOKM mirror)
static constexpr int NH_TOKM_OFF = NH_MSG_OFF + NH_MSG_LEN
;
static constexpr int NH_TOKI_OFF = NH_TOKM_OFF + NH_V3K * NH_V3_MONF;
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
// v5 semantic-class LUTs (host-built once at create, before graph capture)
static unsigned char* nh_locc_lut_dev = NULL;  // glyph -> 9-class local id
static unsigned char* nh_terrc_lut_dev = NULL; // glyph -> 17-class terrain id
static void nh_v5_luts_init(void) {
    if (nh_locc_lut_dev) return;
    unsigned char* loc = (unsigned char*)malloc(NH_GLYPH_VOCAB);
    unsigned char* ter = (unsigned char*)malloc(NH_GLYPH_VOCAB);
    for (int g = 0; g < NH_GLYPH_VOCAB; g++) {
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
        if (g == NH_PAD_GLYPH) lc = 8;                                  // off-map
        loc[g] = lc;
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
        ter[g] = tc;
    }
    cudaMalloc(&nh_locc_lut_dev, NH_GLYPH_VOCAB);
    cudaMalloc(&nh_terrc_lut_dev, NH_GLYPH_VOCAB);
    cudaMemcpy(nh_locc_lut_dev, loc, NH_GLYPH_VOCAB, cudaMemcpyHostToDevice);
    cudaMemcpy(nh_terrc_lut_dev, ter, NH_GLYPH_VOCAB, cudaMemcpyHostToDevice);
    free(loc); free(ter);
}
static unsigned char* nh_haz_lut_dev = NULL; // species -> hazard bits
static void nh_haz_lut_init(void) {
    if (nh_haz_lut_dev) return;
    cudaMalloc(&nh_haz_lut_dev, NH_MONS_STATIC_N);
    cudaMemcpy(nh_haz_lut_dev, NH_MON_HAZ, NH_MONS_STATIC_N, cudaMemcpyHostToDevice);
}
// semantic crop gather: crop glyph ids -> class -> tiny class embedding.
// Hero cell (crop center) is class 7 (mirrors croplab).
__global__ void nh_loc3_gather_kernel(precision_t* __restrict__ x,
    const precision_t* __restrict__ locc_w, const float* __restrict__ crop_glyph,
    const unsigned char* __restrict__ lut, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LOC_IN) return;
    int b = t / NH_LOC_IN, j = t % NH_LOC_IN;
    int p = j / NH_LOCC_DIM, d = j % NH_LOCC_DIM;
    int g = (int)crop_glyph[(int64_t)b * NH_CGRID + p];
    int cls = p == (NH_CGRID / 2) ? 7 : (int)lut[g];
    x[t] = locc_w[cls * NH_LOCC_DIM + d];
}
__global__ void nh_loc3_scatter_kernel(long long* __restrict__ acc,
    const precision_t* __restrict__ dx, const float* __restrict__ crop_glyph,
    const unsigned char* __restrict__ lut, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LOC_IN) return;
    int b = t / NH_LOC_IN, j = t % NH_LOC_IN;
    int p = j / NH_LOCC_DIM, d = j % NH_LOCC_DIM;
    int g = (int)crop_glyph[(int64_t)b * NH_CGRID + p];
    int cls = p == (NH_CGRID / 2) ? 7 : (int)lut[g];
    float v = to_float(dx[t]);
    if (v != 0.0f) nh_fxp_atomic_add(&acc[cls * NH_LOCC_DIM + d], v);
}
// terrain featurize (forward-only; features carry no gradients): landmark
// table (12x[seen,dx/78,dy/20,min(d,30)/30]) + sector radar (8x4x17,
// log1p(c)/log1p(1660)). Mirrors terrlab2 exactly: bands digitize([3,7,15]),
// hero cell counts as floor, all 17 classes counted incl. landmarks.
__global__ void nh_terr_feat_kernel(precision_t* __restrict__ tf,
    const float* __restrict__ gidx, const precision_t* __restrict__ obs,
    const unsigned char* __restrict__ lut, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const precision_t* bl = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF;
    int hx = nh_bl_read_i32(bl), hy = nh_bl_read_i32(bl + 4);
    int hcell = hy * NH_MAPW + hx;
    float lm[48];
    for (int i = 0; i < 48; i++) lm[i] = 0.0f;
    int lmd[12];
    for (int i = 0; i < 12; i++) lmd[i] = 1 << 30;
    float sec[8 * 4 * 17];
    for (int i = 0; i < 8 * 4 * 17; i++) sec[i] = 0.0f;
    for (int cell = 0; cell < NH_MGRID; cell++) {
        int g = (int)gidx[(int64_t)b * NH_MGRID + cell];
        int tc = cell == hcell ? 13 : (int)lut[g];
        if (tc == 255) continue;
        int dy = cell / NH_MAPW - hy, dx = cell % NH_MAPW - hx;
        int ady = dy < 0 ? -dy : dy, adx = dx < 0 ? -dx : dx;
        int cheb = adx > ady ? adx : ady;
        if (tc < 12 && cheb < lmd[tc]) {
            lmd[tc] = cheb;
            lm[tc * 4 + 0] = 1.0f;
            lm[tc * 4 + 1] = (float)dx * (1.0f / 78.0f);
            lm[tc * 4 + 2] = (float)dy * (1.0f / 20.0f);
            lm[tc * 4 + 3] = (float)(cheb < 30 ? cheb : 30) * (1.0f / 30.0f);
        }
        float a = atan2f((float)dy, (float)dx) + 3.14159265358979f;
        int s = ((int)(a / 0.78539816339745f)) & 7;
        int band = cheb < 3 ? 0 : cheb < 7 ? 1 : cheb < 15 ? 2 : 3;
        sec[(s * 4 + band) * 17 + tc] += 1.0f;
    }
    precision_t* o = tf + (int64_t)b * NH_TERRF;
    for (int i = 0; i < 48; i++) o[i] = from_float(lm[i]);
    float inv_log = 1.0f / logf(1660.0f);
    for (int i = 0; i < 8 * 4 * 17; i++) o[48 + i] = from_float(log1pf(sec[i]) * inv_log);
}
// hard wield readout: sum of slot vectors gated by the wielded state bit
// (inv_sfeat flag bit1 = feature index 10). Parameterless.
__global__ void nh_wld_kernel(precision_t* __restrict__ concat,
    const precision_t* __restrict__ inv_out, const precision_t* __restrict__ sfeat, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV_HID) return;
    int b = t / NH_INV_HID, d = t % NH_INV_HID;
    float acc = 0.0f;
    for (int s = 0; s < NH_INV; s++) {
        float w = to_float(sfeat[((int64_t)b * NH_INV + s) * NH_SFEAT + 10]);
        if (w > 0.5f) acc += to_float(inv_out[((int64_t)b * NH_INV + s) * NH_INV_HID + d]);
    }
    concat[(int64_t)b * NH_CONCAT + NH_WLD_CONCAT_OFF + d] = from_float(acc);
}
__global__ void nh_wld_bwd_kernel(precision_t* __restrict__ dinv,
    const precision_t* __restrict__ grad_concat, const precision_t* __restrict__ sfeat, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV * NH_INV_HID) return;
    int b = t / (NH_INV * NH_INV_HID);
    int s = (t / NH_INV_HID) % NH_INV, d = t % NH_INV_HID;
    float w = to_float(sfeat[((int64_t)b * NH_INV + s) * NH_SFEAT + 10]);
    if (w > 0.5f) {
        float g = to_float(grad_concat[(int64_t)b * NH_CONCAT + NH_WLD_CONCAT_OFF + d]);
        dinv[t] = from_float(to_float(dinv[t]) + g);
    }
}
// sum channel: relu(isum_w . slot + b) summed over occupied slots x 0.2
__global__ void nh_isum_pool_kernel(precision_t* __restrict__ concat,
    const precision_t* __restrict__ ih, const float* __restrict__ inv_idx, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_ISUM_DIM) return;
    int b = t / NH_ISUM_DIM, d = t % NH_ISUM_DIM;
    float acc = 0.0f;
    for (int s = 0; s < NH_INV; s++)
        if ((int)inv_idx[(int64_t)b * NH_INV + s] != NH_PAD_GLYPH)
            acc += to_float(ih[((int64_t)b * NH_INV + s) * NH_ISUM_DIM + d]);
    concat[(int64_t)b * NH_CONCAT + NH_ISUM_CONCAT_OFF + d] = from_float(acc * 0.2f);
}
__global__ void nh_isum_dh_kernel(precision_t* __restrict__ dih,
    const precision_t* __restrict__ grad_concat, const float* __restrict__ inv_idx, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV * NH_ISUM_DIM) return;
    int b = t / (NH_INV * NH_ISUM_DIM);
    int s = (t / NH_ISUM_DIM) % NH_INV, d = t % NH_ISUM_DIM;
    float g = 0.0f;
    if ((int)inv_idx[(int64_t)b * NH_INV + s] != NH_PAD_GLYPH)
        g = 0.2f * to_float(grad_concat[(int64_t)b * NH_CONCAT + NH_ISUM_CONCAT_OFF + d]);
    dih[t] = from_float(g);
}
// ---- lab arm kernels: typed streams (deep values + 8-head pools) ----
// token builder: 48-dim = [e_eff(glyph) 32 | dx dy cheb rank | type flags |
// diff speed | pad]. gid stores the mapped GLYPH id (shared embed table).
__global__ void nh_lab_tok_kernel(precision_t* __restrict__ tok, float* __restrict__ gid,
    const precision_t* __restrict__ obs, const precision_t* __restrict__ e_eff,
    const unsigned char* __restrict__ haz, int list_off, int is_mon, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LABK) return;
    int b = t / NH_LABK, k = t % NH_LABK;
    const precision_t* e = obs + (int64_t)b * NH_OBS_SIZE + list_off + k * NH_V3_MONF;
    int row = (int)to_float(e[0]) | ((int)to_float(e[1]) << 8);
    int dx = (int)to_float(e[2]); if (dx >= 128) dx -= 256;
    int dy = (int)to_float(e[3]); if (dy >= 128) dy -= 256;
    int f4 = (int)to_float(e[4]), f5 = (int)to_float(e[5]), f6 = (int)to_float(e[6]);
    int g = row <= 0 ? -1
          : is_mon ? row - 1
          : (row < 454 ? NH_OBJ_LO + row - 1 : NH_BODY_OFF + row - 454);
    gid[t] = (float)g;
    precision_t* o = tok + (int64_t)t * NH_LAB_IN;
    for (int d = 0; d < NH_EMBED_DIM; d++)
        o[d] = g >= 0 ? e_eff[(int64_t)g * NH_EMBED_DIM + d] : from_float(0.0f);
    int cheb = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    // strict [-1,1]: record-validated scaling with rare tails clamped
    o[32] = from_float(g >= 0 ? fmaxf(fminf((float)dx * (1.0f / 40.0f), 1.0f), -1.0f) : 0.0f);
    o[33] = from_float(g >= 0 ? fmaxf(fminf((float)dy * (1.0f / 11.0f), 1.0f), -1.0f) : 0.0f);
    o[34] = from_float(g >= 0 ? fminf((float)cheb, 15.0f) * (1.0f / 15.0f) : 0.0f);
    o[35] = from_float(g >= 0 ? (float)k * (1.0f / 15.0f) : 0.0f); // rank
    if (is_mon) { // e4 flags: bit0 hostile, bit2 detected, bit3 pet
        o[36] = from_float(g >= 0 && (f4 & 1) ? 1.0f : 0.0f);
        o[37] = from_float(g >= 0 && (f4 & 8) ? 1.0f : 0.0f);
        o[38] = from_float(g >= 0 && (f4 & 4) ? 1.0f : 0.0f);
        o[39] = from_float(g >= 0 && cheb <= 1 ? 1.0f : 0.0f);
        o[40] = from_float(g >= 0 ? fminf((float)f5 * 0.04f, 1.0f) : 0.0f);
        o[41] = from_float(g >= 0 ? fminf((float)f6 * (1.0f / 24.0f), 1.0f) : 0.0f);
    } else { // e5 flags: bit0 underfoot, bit1 body
        o[36] = from_float(g >= 0 && (f5 & 1) ? 1.0f : 0.0f);
        o[37] = from_float(g >= 0 && (f5 & 2) ? 1.0f : 0.0f);
        o[38] = o[39] = o[40] = o[41] = from_float(0.0f);
        (void)f4; (void)f6;
    }
    { // hazard bits (species LUT): passive, engulf, explosive, poisonous
        int hb = is_mon && row > 0 && haz != NULL ? (int)haz[(row - 1) % 381] : 0;
        o[42] = from_float((hb & 1) ? 1.0f : 0.0f);
        o[43] = from_float((hb & 2) ? 1.0f : 0.0f);
        o[44] = from_float((hb & 4) ? 1.0f : 0.0f);
        o[45] = from_float((hb & 8) ? 1.0f : 0.0f);
    }
    for (int d = 46; d < NH_LAB_IN; d++) o[d] = from_float(0.0f);
}
// 8-head attention pool: scores from RAW token features (the lab winning
// form: a = Linear(vin, heads)), per-head softmax over valid tokens, output
// slice h = sum_k attn * h2[k, h*8..]. One thread per (sample, head).
__global__ void nh_lab_pool_kernel(precision_t* __restrict__ concat, precision_t* __restrict__ attn,
    const precision_t* __restrict__ h2, const precision_t* __restrict__ tok,
    const float* __restrict__ gid, const precision_t* __restrict__ aw,
    const precision_t* __restrict__ ab, int concat_off, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LAB_HEADS) return;
    int b = t / NH_LAB_HEADS, hh = t % NH_LAB_HEADS;
    const int D = NH_LAB_HID / NH_LAB_HEADS;
    float sc[NH_LABK], mx = -1e30f;
    int any = 0;
    for (int k = 0; k < NH_LABK; k++) {
        if (gid[(int64_t)b * NH_LABK + k] < 0.0f) { sc[k] = -1e30f; continue; }
        any = 1;
        const precision_t* tk = tok + ((int64_t)b * NH_LABK + k) * NH_LAB_IN;
        float s = to_float(ab[hh]);
        for (int j = 0; j < NH_LAB_IN; j++)
            s += to_float(aw[hh * NH_LAB_IN + j]) * to_float(tk[j]);
        sc[k] = s;
        if (s > mx) mx = s;
    }
    float z = 0.0f;
    for (int k = 0; k < NH_LABK; k++) {
        sc[k] = any && sc[k] > -1e29f ? expf(sc[k] - mx) : 0.0f;
        z += sc[k];
    }
    float out[NH_LAB_HID / NH_LAB_HEADS];
    for (int d = 0; d < D; d++) out[d] = 0.0f;
    for (int k = 0; k < NH_LABK; k++) {
        float a = z > 0.0f ? sc[k] / z : 0.0f;
        attn[((int64_t)b * NH_LABK + k) * NH_LAB_HEADS + hh] = from_float(a);
        if (a > 0.0f) {
            const precision_t* hk = h2 + ((int64_t)b * NH_LABK + k) * NH_LAB_HID + hh * D;
            for (int d = 0; d < D; d++) out[d] += a * to_float(hk[d]);
        }
    }
    precision_t* dst = concat + (int64_t)b * NH_CONCAT + concat_off + hh * D;
    for (int d = 0; d < D; d++) dst[d] = from_float(out[d]);
}
// pool backward: dh2 (per-head slice, exclusive) and the softmax jacobian's
// per-token score grads ds (consumed by GEMMs for daw / dtok-score / dab).
// Invalid tokens have attn 0 so their ds and dh2 are exact zeros.
__global__ void nh_lab_pool_bwd_kernel(precision_t* __restrict__ dh2, precision_t* __restrict__ ds,
    const precision_t* __restrict__ grad_concat, const precision_t* __restrict__ attn,
    const precision_t* __restrict__ h2, int concat_off, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LAB_HEADS) return;
    int b = t / NH_LAB_HEADS, hh = t % NH_LAB_HEADS;
    const int D = NH_LAB_HID / NH_LAB_HEADS;
    const precision_t* gd = grad_concat + (int64_t)b * NH_CONCAT + concat_off + hh * D;
    float gv[NH_LABK], mean_gv = 0.0f;
    for (int k = 0; k < NH_LABK; k++) {
        float a = to_float(attn[((int64_t)b * NH_LABK + k) * NH_LAB_HEADS + hh]);
        const precision_t* hk = h2 + ((int64_t)b * NH_LABK + k) * NH_LAB_HID + hh * D;
        float v = 0.0f;
        for (int d = 0; d < D; d++) v += to_float(gd[d]) * to_float(hk[d]);
        gv[k] = v;
        mean_gv += a * v;
        precision_t* dhk = dh2 + ((int64_t)b * NH_LABK + k) * NH_LAB_HID + hh * D;
        for (int d = 0; d < D; d++) dhk[d] = from_float(a * to_float(gd[d]));
    }
    for (int k = 0; k < NH_LABK; k++) {
        float a = to_float(attn[((int64_t)b * NH_LABK + k) * NH_LAB_HEADS + hh]);
        ds[((int64_t)b * NH_LABK + k) * NH_LAB_HEADS + hh] = from_float(a * (gv[k] - mean_gv));
    }
}
// per-head score bias grad: deterministic fixed-point column sum of ds
__global__ void nh_lab_dab_kernel(long long* __restrict__ acc,
    const precision_t* __restrict__ ds, int64_t total) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    float v = to_float(ds[t]);
    if (v != 0.0f) nh_fxp_atomic_add(&acc[t % NH_LAB_HEADS], v);
}
// embed-table scatter for a stream's dtok buffer (first 32 dims only)
__global__ void nh_lab_dE_scatter_kernel(long long* __restrict__ dE_i,
    const precision_t* __restrict__ dtok, const float* __restrict__ gid, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LABK * NH_EMBED_DIM) return;
    int bk = t / NH_EMBED_DIM, d = t % NH_EMBED_DIM;
    int g = (int)gid[bk];
    if (g < 0) return;
    float v = to_float(dtok[(int64_t)bk * NH_LAB_IN + d]);
    if (v != 0.0f) nh_fxp_atomic_add(&dE_i[(int64_t)g * NH_EMBED_DIM + d], v);
}

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
    const precision_t* __restrict__ obs, int pad, int B) {
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
            int g = (r < NH_MAPH && c < NH_MAPW) ? (int)gi[r * NH_MAPW + c] : pad;
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

__global__ void nh_fill_kernel(precision_t* p, float v, int n); // defined with the decoder below


// inventory attention tail: NH_IVA_M learned queries softmax over the 55
// post-relu slot reps (empty slots attended too — their shared pad rep is
// learnable to suppress). 4x16 tail beside the existing max pool.
__global__ void nh_iva_kernel(precision_t* __restrict__ concat, precision_t* __restrict__ attn,
    const precision_t* __restrict__ inv_out, const precision_t* __restrict__ q, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_IVA_M) return;
    int b = t / NH_IVA_M, m = t % NH_IVA_M;
    const precision_t* K = inv_out + (int64_t)b * NH_INV_FLAT;
    float sc[NH_INV], mx = -1e30f;
    for (int s = 0; s < NH_INV; s++) {
        float d = 0.0f;
        for (int k = 0; k < NH_INV_HID; k++)
            d += to_float(q[m * NH_INV_HID + k]) * to_float(K[s * NH_INV_HID + k]);
        sc[s] = d;
        if (d > mx) mx = d;
    }
    float z = 0.0f;
    for (int s = 0; s < NH_INV; s++) { sc[s] = expf(sc[s] - mx); z += sc[s]; }
    float out[NH_INV_HID];
    for (int k = 0; k < NH_INV_HID; k++) out[k] = 0.0f;
    for (int s = 0; s < NH_INV; s++) {
        float aw = sc[s] / z;
        attn[((int64_t)b * NH_IVA_M + m) * NH_INV + s] = from_float(aw);
        for (int k = 0; k < NH_INV_HID; k++) out[k] += aw * to_float(K[s * NH_INV_HID + k]);
    }
    precision_t* dst = concat + (int64_t)b * NH_CONCAT + NH_IVA_CONCAT_OFF + m * NH_INV_HID;
    for (int k = 0; k < NH_INV_HID; k++) dst[k] = from_float(out[k]);
}
// backward: adds dK into inv_grad (after the max-pool bwd overwrote it, before
// the relu mask — same slot as the pointer key grads), dq in fixed point. One
// thread per sample keeps the += race-free and deterministic.
__global__ void nh_iva_bwd_kernel(precision_t* __restrict__ inv_grad, long long* __restrict__ dq_acc,
    const precision_t* __restrict__ grad_concat, const precision_t* __restrict__ attn,
    const precision_t* __restrict__ inv_out, const precision_t* __restrict__ q, int B) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const precision_t* K = inv_out + (int64_t)b * NH_INV_FLAT;
    precision_t* dK = inv_grad + (int64_t)b * NH_INV_FLAT;
    float dq[NH_IVA_M * NH_INV_HID];
    for (int i = 0; i < NH_IVA_M * NH_INV_HID; i++) dq[i] = 0.0f;
    for (int m = 0; m < NH_IVA_M; m++) {
        const precision_t* g = grad_concat + (int64_t)b * NH_CONCAT + NH_IVA_CONCAT_OFF + m * NH_INV_HID;
        const precision_t* a = attn + ((int64_t)b * NH_IVA_M + m) * NH_INV;
        float gv[NH_INV], mean_gv = 0.0f;
        for (int s = 0; s < NH_INV; s++) {
            float v = 0.0f;
            for (int k = 0; k < NH_INV_HID; k++) v += to_float(g[k]) * to_float(K[s * NH_INV_HID + k]);
            gv[s] = v;
            mean_gv += to_float(a[s]) * v;
        }
        for (int s = 0; s < NH_INV; s++) {
            float aw = to_float(a[s]), ds = aw * (gv[s] - mean_gv);
            for (int k = 0; k < NH_INV_HID; k++) {
                dK[s * NH_INV_HID + k] = from_float(to_float(dK[s * NH_INV_HID + k])
                    + aw * to_float(g[k]) + ds * to_float(q[m * NH_INV_HID + k]));
                dq[m * NH_INV_HID + k] += ds * to_float(K[s * NH_INV_HID + k]);
            }
        }
    }
    for (int i = 0; i < NH_IVA_M * NH_INV_HID; i++)
        if (dq[i] != 0.0f) atomicAdd((unsigned long long*)&dq_acc[i],
            (unsigned long long)(long long)__float2ll_rn(dq[i] * NH_FXP));
}


__global__ void nh_patch_max_bwd_kernel(
    precision_t* __restrict__ t16_io, long long* __restrict__ dw2_acc,
    const precision_t* __restrict__ dglb, const precision_t* __restrict__ w2,
    const int* __restrict__ argmax, const precision_t* __restrict__ dmean,
    const precision_t* __restrict__ dt_extra, int B) {
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
        float v = nh_fxp_to_float(dt16s[i]);
        if (dmean) v += to_float(dmean[(int64_t)b * NH_P1 + (i % NH_P1)]) * (1.0f / NH_TOK); // mean path: uniform over tokens
        if (dt_extra) v += to_float(dt_extra[(int64_t)b * (NH_TOK * NH_P1) + i]); // topk path
        t16_io[(int64_t)b * (NH_TOK * NH_P1) + i] = from_float(t16s[i] > 0.0f ? v : 0.0f);
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
        } else if (j < NH_F_INTRINS) {
            // role/race/gender one-hots, already 0/1
            // dead under NH_ID_EMBED: identity flows via the embed channel
            f = 0.0f; // identity flows via the embed channel
        } else {
            f = (float)((nh_bl_read_i32(ex + 4*NH_EX_INTRINS) >> (j - NH_F_INTRINS)) & 1);
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

// spell sum-channel forward: masked sum x0.2 of relu'd key projections + 4
// exact doorstep scalars (all in [0,1]; no spells -> minfail 1, maxlev 0,
// n 0, minret 1).
__global__ void nh_sp2_pool_kernel(precision_t* __restrict__ concat,
    const precision_t* __restrict__ h, const float* __restrict__ spell_idx,
    const precision_t* __restrict__ obs, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLICE) return;
    int b = t / NH_SPELL_SLICE, d = t % NH_SPELL_SLICE;
    float v;
    if (d < NH_SP2_DIM) {
        float acc = 0.0f;
        for (int s = 0; s < NH_SPELL_SLOTS; s++)
            if (spell_idx[(int64_t)b * NH_SPELL_SLOTS + s] >= 0.0f)
                acc += to_float(h[((int64_t)b * NH_SPELL_SLOTS + s) * NH_SP2_DIM + d]);
        v = acc * 0.2f;
    } else {
        const precision_t* ex = obs + (int64_t)b * NH_OBS_SIZE + NH_BL_OFF + NH_BL_RAW * 4;
        int mf = 100, ml = 0, n = 0, mr = 20000;
        for (int s = 0; s < NH_SPELL_SLOTS; s++) {
            const precision_t* q = ex + 4 * (NH_EXTRA_SHOP + 2 + 1 + 4 * s);
            int sid = nh_bl_read_i32(q);
            if (sid <= 0) continue;
            n++;
            int lv = nh_bl_read_i32(q + 4), fl = nh_bl_read_i32(q + 8), kn = nh_bl_read_i32(q + 12);
            if (fl < mf) mf = fl;
            if (lv > ml) ml = lv;
            if (kn < mr) mr = kn;
        }
        int j = d - NH_SP2_DIM;
        v = j == 0 ? (float)mf * 0.01f
          : j == 1 ? (float)ml * (1.0f / 7.0f)
          : j == 2 ? (float)(n > 8 ? 8 : n) * 0.125f
                   : (float)mr * 0.00005f;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
    }
    concat[(int64_t)b * NH_CONCAT + NH_SPELL_CONCAT_OFF + d] = from_float(v);
}
// backward: dih = 0.2 * occ * g (doorstep scalars carry no params)
__global__ void nh_sp2_dh_kernel(precision_t* __restrict__ dih,
    const precision_t* __restrict__ grad_concat, const float* __restrict__ spell_idx, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_SP2_DIM) return;
    int b = t / (NH_SPELL_SLOTS * NH_SP2_DIM);
    int s = (t / NH_SP2_DIM) % NH_SPELL_SLOTS, d = t % NH_SP2_DIM;
    float g = 0.0f;
    if (spell_idx[(int64_t)b * NH_SPELL_SLOTS + s] >= 0.0f)
        g = 0.2f * to_float(grad_concat[(int64_t)b * NH_CONCAT + NH_SPELL_CONCAT_OFF + d]);
    dih[t] = from_float(g);
}
// dkeys under SPELL2 = pointer grads only (the sum channel reads the RAW
// slot inputs, not the keys), gated by the key relu
__global__ void nh_sp2_dk_kernel(precision_t* __restrict__ dkeys,
    const precision_t* __restrict__ ptr_dkeys, const precision_t* __restrict__ keys, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_SPKEY) return;
    float v = ptr_dkeys ? to_float(ptr_dkeys[t]) : 0.0f;
    if (to_float(keys[t]) <= 0.0f) v = 0.0f;
    dkeys[t] = from_float(v);
}
// sum-channel embed grads: d_emb = ss_w[:, :32]^T @ dih per occupied slot,
// scattered into dE by the slot's book glyph (mirror of nh_spkey_dE)
__global__ void nh_sp2_dE_kernel(long long* __restrict__ dE_i,
    const precision_t* __restrict__ dih, const precision_t* __restrict__ ss_w,
    const float* __restrict__ sp_idx, int B) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_EMBED_DIM) return;
    int bs = t / NH_EMBED_DIM;
    int d = t % NH_EMBED_DIM;
    int g = (int)sp_idx[bs];
    if (g < 0) return;
    float acc = 0.0f;
    for (int r = 0; r < NH_SP2_DIM; r++)
        acc += to_float(ss_w[r * NH_SPIN + d]) * to_float(dih[(int64_t)bs * NH_SP2_DIM + r]);
    if (acc != 0.0f) nh_fxp_atomic_add(&dE_i[(int64_t)g * NH_EMBED_DIM + d], acc);
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
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INVP_DIM)
        val = inv[(int64_t)b * NH_INV_POOL + (c - NH_LOC_HID - NH_GLB_HID)];
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INVP_DIM + NH_BL_HID)
        val = bl_out[(int64_t)b * NH_BL_HID + (c - NH_LOC_HID - NH_GLB_HID - NH_INVP_DIM)];
    else if (c < NH_MSG_CONCAT_OFF)
        val = bl_feats[(int64_t)b * NH_BL_FEAT + (c - NH_LOC_HID - NH_GLB_HID - NH_INVP_DIM - NH_BL_HID)];
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
    // strict [-1,1]: spe clamp/7; quantity LINEAR capped (log-squash
    // regressed ammo-count .88->.76 across two seeds — counts want scale)
    f[5] = from_float(spe_known ? fmaxf(fminf((float)st[1], 7.0f), -7.0f) * (1.0f / 7.0f) : 0.0f);
    f[6] = from_float(fminf(fmaxf((float)st[2], 0.0f), 30.0f) * (1.0f / 30.0f));
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
    const int* __restrict__ hot_list, const int* __restrict__ hot_n, int pad, int B) {
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
            int gl = (r < NH_MAPH && c < NH_MAPW) ? (int)gi[r * NH_MAPW + c] : pad;
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
__global__ void nh_count_pad_kernel(int* __restrict__ counts, int pad, int B) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        counts[pad] += NH_PAD_PER_SAMPLE * B;
}

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

// encoder structs

struct NethackEncoderWeights {
    Prec embed_w, ekind_w, esub_w, loc_w, loc_b;
    Prec loc2_w, loc2_b; // (NH_LOC_HID, NH_LOC_H1), (NH_LOC_HID)
    Prec iaq_w; // (NH_IVA_M, NH_INV_HID) inventory attention queries
    Prec terr1_w, terr1_b, terr2_w, terr2_b; // terrain MLP 592->256->128
    Prec locc_w; // (NH_LOCC_CLASSES, NH_LOCC_DIM) local class table
    Prec inv1_w, inv1_b, inv1s_w, invt_w;
    Prec isum_w, isum_b; // (NH_ISUM_DIM, NH_INV_HID) sum-channel projection
    Prec bl_w, bl_b, proj_w, proj_b;
    Prec msg_w; // trigram embedding table (NH_MSG_VOCAB, NH_MSG_HID)
    Prec spk_w; // spell slot-rep projection (NH_SPKEY, NH_SPIN)
    Prec ss_w, ss_b; // sum-channel projection over RAW slot inputs (NH_SP2_DIM, NH_SPIN)
    Prec ide_role_w, ide_race_w, ide_gend_w, ide_algn_w; // identity tables
    Prec lm1_w, lm1_b, lm2_w, lm2_b; // monster stream deep values (48->64->64)
    Prec lma_w, lma_b;               // monster stream score linear (8, 48)
    Prec li1_w, li1_b, li2_w, li2_b; // item stream deep values
    Prec lia_w, lia_b;               // item stream score linear
    int obs_size, hidden;
};

struct NethackEncoderActivations {
    Float glyph_idx, crop_glyph; // decoded grid + crop glyph ids
    Prec e_eff; // materialized E_res + E_kind + E_sub
    Prec x_local; // crop embeds (grad aliases it)
    Prec terr_tf; // (B, NH_TERRF) featurized terrain (fwd-only input)
    Prec terr_h, terr_dh; // relu'd hidden (B, NH_TERR_H1) + its grad
    Long terr1b_acc; // fixed-point terr1_b accumulator
    Prec terr1_wgrad, terr1_bgrad, terr2_wgrad, terr2_bgrad;
    Long locc_acc; // fixed-point class-table accumulator
    Prec locc_wgrad;
    Prec isum_h, isum_dh; // (B, 55*NH_ISUM_DIM) sum-channel hidden + grad
    Long isumb_acc; Prec isum_wgrad, isum_bgrad;
    Float inv_idx; // inventory slot glyph ids
    Float spell_idx; // per-slot book glyphs (-1 = empty slot)
    Prec spk_in, spk_keys; // spell-key inputs (B, 8*36) + relu'd reps (B, 8*16)
    Prec spk_dkeys; // per-slot key grads (pointer; +pool under !SPELL2)
    Prec sp2_h, sp2_dh; // (B, 8*NH_SP2_DIM) sum-channel hidden + grad
    Long ssb_acc; Prec ss_wgrad, ss_bgrad;
    Float invt_idx; // discovered-type glyph ids (pad = unknown)
    Prec inv_sfeat; // per-slot state features (B, 55*NH_SFEAT)
    Prec inv_T, inv_out; // fused inv table + relu'd flat slots
    Prec invt_T; // fused discovered-type table
    Prec loc_out, glb_out;
    Prec bl_feats, bl_out;
    Float msg_ids; // per-position trigram bucket ids (-1 pad)
    Prec msg_out; // normalized trigram-bag summary (B, NH_MSG_HID)
    Prec concat, out;
    Prec loc_grad, glb_grad, inv_grad, bl_grad; // contiguous concat slices
    Prec iva_attn; // (B, NH_IVA_M*NH_INV) attention weights
    Long diaq_acc; Prec iaq_wgrad;
    Prec dTinv, dE_tmp; // inv-table grad + its dE staging
    Prec dTtrue; // discovered-type table grad
    Long dTinv_i, dTtrue_i; // fixed-point dT scatter staging
    Long dE_i; // fixed-point local embed-grad staging
    Long dmsg_acc; // fixed-point trigram-table wgrad staging
    Long bias_acc; // fixed-point bias grads: proj | loc | glb2 | bl | glb1 | inv1 | inv2
    Prec embed_wgrad, ekind_wgrad, esub_wgrad, loc_wgrad, loc_bgrad;
    Prec loc2_wgrad, loc2_bgrad;
    Prec loc_h1, loc_h1_grad; // relu'd hidden (B, NH_LOC_H1) and its grad
    Long loc1b_acc; // fixed-point bias accumulator for loc_b
    Prec inv1_wgrad, inv1_bgrad, inv1s_wgrad, invt_wgrad;
    Prec bl_wgrad, bl_bgrad, proj_wgrad, proj_bgrad;
    Prec msg_wgrad, spk_wgrad;
    Float ide_idx; // per-sample [role, race, gend, align] saved for backward
    Prec ide_role_wgrad, ide_race_wgrad, ide_gend_wgrad, ide_algn_wgrad;
    Prec lm_tok, lm_h1, lm_h2, lm_attn; Float lm_gid; // monster stream fwd
    Prec li_tok, li_h1, li_h2, li_attn; Float li_gid; // item stream fwd
    Prec lm_dh2, lm_dh1, lm_ds, lm_dts; // bwd: dh2, dh1, score grads, score-path dtok
    Prec li_dh2, li_dh1, li_ds, li_dts;
    Long lm1b_acc, lm2b_acc, lmab_acc; // fixed-point bias accs
    Long li1b_acc, li2b_acc, liab_acc;
    Prec lm1_wgrad, lm1_bgrad, lm2_wgrad, lm2_bgrad, lma_wgrad, lma_bgrad;
    Prec li1_wgrad, li1_bgrad, li2_wgrad, li2_bgrad, lia_wgrad, lia_bgrad;
};

static NethackEncoderWeights* nethack_encoder_create(int obs_size, int hidden) {
    nh_v5_luts_init();
    nh_haz_lut_init();
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
    nh_loc3_gather_kernel<<<grid_size(B * NH_LOC_IN), BLOCK_SIZE, 0, stream>>>(
        a->x_local.data, ew->locc_w.data, a->crop_glyph.data, nh_locc_lut_dev, B);
    puf_mm(&a->x_local, &ew->loc_w, &a->loc_h1, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LOC_H1), BLOCK_SIZE, 0, stream>>>(
        a->loc_h1.data, ew->loc_b.data, B * NH_LOC_H1, NH_LOC_H1);
    puf_mm(&a->loc_h1, &ew->loc2_w, &a->loc_out, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LOC_HID), BLOCK_SIZE, 0, stream>>>(
        a->loc_out.data, ew->loc2_b.data, B * NH_LOC_HID, NH_LOC_HID);

    // terrain branch replaces the patch encoder: featurize (fwd-only, no
    // input grads) -> 592 -> 256 -> 128 into the glb slot
    nh_terr_feat_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->terr_tf.data, a->glyph_idx.data, input.data, nh_terrc_lut_dev, B);
    puf_mm(&a->terr_tf, &ew->terr1_w, &a->terr_h, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_TERR_H1), BLOCK_SIZE, 0, stream>>>(
        a->terr_h.data, ew->terr1_b.data, B * NH_TERR_H1, NH_TERR_H1);
    puf_mm(&a->terr_h, &ew->terr2_w, &a->glb_out, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_GLB_HID), BLOCK_SIZE, 0, stream>>>(
        a->glb_out.data, ew->terr2_b.data, B * NH_GLB_HID, NH_GLB_HID);

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
    nh_wld_kernel<<<grid_size(B * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->inv_out.data, a->inv_sfeat.data, B);
    { Prec invf = {.data = a->inv_out.data, .shape = {B * NH_INV, NH_INV_HID}};
      Prec ihf = {.data = a->isum_h.data, .shape = {B * NH_INV, NH_ISUM_DIM}};
      puf_mm(&invf, &ew->isum_w, &ihf, stream); }
    nh_bias_relu_kernel<<<grid_size(B * NH_INV * NH_ISUM_DIM), BLOCK_SIZE, 0, stream>>>(
        a->isum_h.data, ew->isum_b.data, (int64_t)B * NH_INV * NH_ISUM_DIM, NH_ISUM_DIM);
    nh_isum_pool_kernel<<<grid_size(B * NH_ISUM_DIM), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->isum_h.data, a->inv_idx.data, B);
    nh_iva_kernel<<<grid_size(B * NH_IVA_M), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->iva_attn.data, a->inv_out.data, ew->iaq_w.data, B);

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
        a->concat.data, a->loc_out.data, a->glb_out.data,
        (const precision_t*)NULL, // invpool slice is width-0
        a->bl_out.data, a->bl_feats.data, a->msg_out.data, B);
    { // typed streams: tok -> deep values (48->64->64) -> 8-head pool
    nh_lab_tok_kernel<<<grid_size(B * NH_LABK), BLOCK_SIZE, 0, stream>>>(
        a->lm_tok.data, a->lm_gid.data, input.data, a->e_eff.data,
        nh_haz_lut_dev,
        NH_TOKM_OFF, 1, B);
    Prec mtokf = {.data = a->lm_tok.data, .shape = {B * NH_LABK, NH_LAB_IN}};
    Prec mh1f = {.data = a->lm_h1.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm(&mtokf, &ew->lm1_w, &mh1f, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LABK * NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->lm_h1.data, ew->lm1_b.data, B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    Prec mh2f = {.data = a->lm_h2.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm(&mh1f, &ew->lm2_w, &mh2f, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LABK * NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->lm_h2.data, ew->lm2_b.data, B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    nh_lab_pool_kernel<<<grid_size(B * NH_LAB_HEADS), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->lm_attn.data, a->lm_h2.data, a->lm_tok.data, a->lm_gid.data,
        ew->lma_w.data, ew->lma_b.data, NH_LABM_CONCAT_OFF, B);
    nh_lab_tok_kernel<<<grid_size(B * NH_LABK), BLOCK_SIZE, 0, stream>>>(
        a->li_tok.data, a->li_gid.data, input.data, a->e_eff.data, NULL, NH_TOKI_OFF, 0, B);
    Prec itokf = {.data = a->li_tok.data, .shape = {B * NH_LABK, NH_LAB_IN}};
    Prec ih1f = {.data = a->li_h1.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm(&itokf, &ew->li1_w, &ih1f, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LABK * NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->li_h1.data, ew->li1_b.data, B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    Prec ih2f = {.data = a->li_h2.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm(&ih1f, &ew->li2_w, &ih2f, stream);
    nh_bias_relu_kernel<<<grid_size(B * NH_LABK * NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->li_h2.data, ew->li2_b.data, B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    nh_lab_pool_kernel<<<grid_size(B * NH_LAB_HEADS), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->li_attn.data, a->li_h2.data, a->li_tok.data, a->li_gid.data,
        ew->lia_w.data, ew->lia_b.data, NH_LABI_CONCAT_OFF, B);
    }
    nh_spkey_kernel<<<grid_size(B * NH_SPELL_SLOTS), BLOCK_SIZE, 0, stream>>>(
        a->spk_keys.data, a->spk_in.data, a->spell_idx.data,
        ew->spk_w.data, a->e_eff.data, input.data, B);
    { Prec inf = {.data = a->spk_in.data, .shape = {B * NH_SPELL_SLOTS, NH_SPIN}};
      Prec hf = {.data = a->sp2_h.data, .shape = {B * NH_SPELL_SLOTS, NH_SP2_DIM}};
      puf_mm(&inf, &ew->ss_w, &hf, stream); }
    nh_bias_relu_kernel<<<grid_size(B * NH_SPELL_SLOTS * NH_SP2_DIM), BLOCK_SIZE, 0, stream>>>(
        a->sp2_h.data, ew->ss_b.data, B * NH_SPELL_SLOTS * NH_SP2_DIM, NH_SP2_DIM);
    nh_sp2_pool_kernel<<<grid_size(B * NH_SPELL_SLICE), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->sp2_h.data, a->spell_idx.data, input.data, B);
    nh_idemb_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->ide_idx.data, input.data,
        ew->ide_role_w.data, ew->ide_race_w.data,
        ew->ide_gend_w.data, ew->ide_algn_w.data, B);
    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    nh_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

// packed bias-acc slot widths: glb1 slot vanishes under TERR (terr1_b has its
// own acc), inv2 slot vanishes under INV2 (max-pool path deleted)
#define NH_BACC_GLB1 0
#define NH_BACC_INVP 0
static void nethack_encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    // fixed-point bias-grad accumulators: [proj H | loc 256 | glb2/terr2 128 | bl 64 | glb1 16? | inv1 | inv2 128?]
    long long* bacc = (long long*)a->bias_acc.data;
    cudaMemsetAsync(bacc, 0, (H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_BACC_GLB1 + NH_INV_HID + NH_BACC_INVP) * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * H, H), BLOCK_SIZE, H * sizeof(long long), stream>>>(
        grad.data, a->out.data, bacc, (int64_t)B * H, H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);


    Prec grad_concat = {.data = a->concat.data, .shape = {B, NH_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    nh_idemb_grad_kernel<<<grid_size(13 * NH_IDE_ROLE + 5 * NH_IDE_RACE
        + 2 * NH_IDE_GEND + 3 * NH_IDE_ALGN), BLOCK_SIZE, 0, stream>>>(
        a->ide_role_wgrad.data, a->ide_race_wgrad.data, a->ide_gend_wgrad.data,
        a->ide_algn_wgrad.data, grad_concat.data, a->ide_idx.data, B);

    // Local view: wgrad against saved x_local, then the input grad overwrites
    // x_local in place before scattering into the embed table.
    nh_slice_kernel<<<grid_size(B * NH_LOC_HID), BLOCK_SIZE, 0, stream>>>(
        a->loc_grad.data, grad_concat.data, B, NH_CONCAT, 0, NH_LOC_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LOC_HID, NH_LOC_HID), BLOCK_SIZE, NH_LOC_HID * sizeof(long long), stream>>>(
        a->loc_grad.data, a->loc_out.data, bacc + H, (int64_t)B * NH_LOC_HID, NH_LOC_HID);
    Prec locg = {.data = a->loc_grad.data, .shape = {B, NH_LOC_HID}};
    // second layer: dW2 = dout^T h1, dh1 = dout W2 (in place over h1 after wgrad), relu mask + own bias acc
    puf_mm_tn(&locg, &a->loc_h1, &a->loc2_wgrad, stream);
    Prec dh1 = {.data = a->loc_h1_grad.data, .shape = {B, NH_LOC_H1}};
    puf_mm_nn(&locg, &ew->loc2_w, &dh1, stream);
    cudaMemsetAsync(a->loc1b_acc.data, 0, NH_LOC_H1 * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LOC_H1, NH_LOC_H1), BLOCK_SIZE, NH_LOC_H1 * sizeof(long long), stream>>>(
        dh1.data, a->loc_h1.data, (long long*)a->loc1b_acc.data, (int64_t)B * NH_LOC_H1, NH_LOC_H1);
    nh_fxp_to_precision_kernel<<<grid_size(NH_LOC_H1), BLOCK_SIZE, 0, stream>>>(
        a->loc_bgrad.data, (long long*)a->loc1b_acc.data, NH_LOC_H1);
    puf_mm_tn(&dh1, &a->x_local, &a->loc_wgrad, stream);
    Prec dx_local = {.data = a->x_local.data, .shape = {B, NH_LOC_IN}};
    puf_mm_nn(&dh1, &ew->loc_w, &dx_local, stream);

    // Global view: relu mask + b2 grad, then the fused max backward (dW2 via
    // fixed-point staging, dt16 overwrites t16), then b1's column sum.
    nh_slice_kernel<<<grid_size(B * NH_GLB_HID), BLOCK_SIZE, 0, stream>>>(
        a->glb_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID, NH_GLB_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_GLB_HID, NH_GLB_HID), BLOCK_SIZE, NH_GLB_HID * sizeof(long long), stream>>>(
        a->glb_grad.data, a->glb_out.data, bacc + H + NH_LOC_HID, (int64_t)B * NH_GLB_HID, NH_GLB_HID);
    // terrain MLP backward: features are inputs (no grad past terr_tf); the
    // bacc glb2 slot above carries terr2_b's grad.
    { Prec glbg = {.data = a->glb_grad.data, .shape = {B, NH_GLB_HID}};
      puf_mm_tn(&glbg, &a->terr_h, &a->terr2_wgrad, stream);
      Prec tdh = {.data = a->terr_dh.data, .shape = {B, NH_TERR_H1}};
      puf_mm_nn(&glbg, &ew->terr2_w, &tdh, stream);
      cudaMemsetAsync(a->terr1b_acc.data, 0, NH_TERR_H1 * sizeof(long long), stream);
      nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_TERR_H1, NH_TERR_H1), BLOCK_SIZE, NH_TERR_H1 * sizeof(long long), stream>>>(
          tdh.data, a->terr_h.data, (long long*)a->terr1b_acc.data, (int64_t)B * NH_TERR_H1, NH_TERR_H1);
      nh_fxp_to_precision_kernel<<<grid_size(NH_TERR_H1), BLOCK_SIZE, 0, stream>>>(
          a->terr1_bgrad.data, (long long*)a->terr1b_acc.data, NH_TERR_H1);
      puf_mm_tn(&tdh, &a->terr_tf, &a->terr1_wgrad, stream); }

    // Inventory branch: slice the pooled-summary grad, relu-mask it (inv2
    // bias grad rides along), backprop the fused max into inv_grad (dW2 via
    // fixed-point staging), then the per-slot relu mask + inv1 bias, dT_inv
    // scatter by slot glyph, dinv1_w = dT_inv^T @ E; dE added at the end.
    // sum channel backward: dih = 0.2*occ*g_isum -> relu gate (+isum bias acc)
    // -> isum wgrad; its dinv via mm_nn is the FIRST writer of inv_grad
    // (mirrors the max-bwd's write-all contract); wield readout adds next.
    nh_isum_dh_kernel<<<grid_size((int64_t)B * NH_INV * NH_ISUM_DIM), BLOCK_SIZE, 0, stream>>>(
        a->isum_dh.data, grad_concat.data, a->inv_idx.data, B);
    cudaMemsetAsync(a->isumb_acc.data, 0, NH_ISUM_DIM * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_INV * NH_ISUM_DIM, NH_ISUM_DIM), BLOCK_SIZE, NH_ISUM_DIM * sizeof(long long), stream>>>(
        a->isum_dh.data, a->isum_h.data, (long long*)a->isumb_acc.data,
        (int64_t)B * NH_INV * NH_ISUM_DIM, NH_ISUM_DIM);
    nh_fxp_to_precision_kernel<<<grid_size(NH_ISUM_DIM), BLOCK_SIZE, 0, stream>>>(
        a->isum_bgrad.data, (long long*)a->isumb_acc.data, NH_ISUM_DIM);
    { Prec dihf = {.data = a->isum_dh.data, .shape = {B * NH_INV, NH_ISUM_DIM}};
      Prec invf = {.data = a->inv_out.data, .shape = {B * NH_INV, NH_INV_HID}};
      puf_mm_tn(&dihf, &invf, &a->isum_wgrad, stream);
      Prec dinvf = {.data = a->inv_grad.data, .shape = {B * NH_INV, NH_INV_HID}};
      puf_mm_nn(&dihf, &ew->isum_w, &dinvf, stream); }
    nh_wld_bwd_kernel<<<grid_size((int64_t)B * NH_INV * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->inv_grad.data, grad_concat.data, a->inv_sfeat.data, B);
    // pointer-decoder key grads: second consumer of inv_out, summed before
    // the relu mask (both paths read the post-relu slot vectors)
    if (nh_ptr_keygrad != NULL)
        nh_add_inplace_kernel<<<grid_size(B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
            a->inv_grad.data, nh_ptr_keygrad->data, B * NH_INV_FLAT);
    // attention tail: third consumer of inv_out, also pre-relu-mask
    cudaMemsetAsync(a->diaq_acc.data, 0, NH_IVA_M * NH_INV_HID * sizeof(long long), stream);
    nh_iva_bwd_kernel<<<grid_size(B), BLOCK_SIZE, 0, stream>>>(
        a->inv_grad.data, (long long*)a->diaq_acc.data, grad_concat.data,
        a->iva_attn.data, a->inv_out.data, ew->iaq_w.data, B);
    nh_fxp_to_precision_kernel<<<grid_size(NH_IVA_M * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->iaq_wgrad.data, (long long*)a->diaq_acc.data, NH_IVA_M * NH_INV_HID);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_INV_FLAT, NH_INV_HID), BLOCK_SIZE, NH_INV_HID * sizeof(long long), stream>>>(
        a->inv_grad.data, a->inv_out.data, bacc + H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_BACC_GLB1,
        (int64_t)B * NH_INV_FLAT, NH_INV_HID);
    // state-path weight grad: dW_s = dslot^T @ sfeat over the B*55 slot rows
    Prec dsflat = {.data = a->inv_grad.data, .shape = {B * NH_INV, NH_INV_HID}};
    Prec sfflat = {.data = a->inv_sfeat.data, .shape = {B * NH_INV, NH_SFEAT}};
    puf_mm_tn(&dsflat, &sfflat, &a->inv1s_wgrad, stream);
    cudaMemsetAsync(a->dTinv_i.data, 0, (size_t)NH_ITBL * NH_INV_HID * sizeof(long long), stream);
    nh_dTinv_scatter_kernel<<<grid_size((int64_t)B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dTinv_i.data, a->inv_grad.data, a->inv_idx.data, (int64_t)B * NH_INV_FLAT, 0);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->dTinv.data, (long long*)a->dTinv_i.data, NH_ITBL * NH_INV_HID);
    puf_mm_tn(&a->dTinv, &a->e_eff, &a->inv1_wgrad, stream);
    // discovered-type channel: same fused-table backward keyed by true glyph
    cudaMemsetAsync(a->dTtrue_i.data, 0, (size_t)NH_ITBL * NH_INV_HID * sizeof(long long), stream);
    nh_dTinv_scatter_kernel<<<grid_size((int64_t)B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dTtrue_i.data, a->inv_grad.data, a->invt_idx.data, (int64_t)B * NH_INV_FLAT, 1);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_INV_HID), BLOCK_SIZE, 0, stream>>>(
        a->dTtrue.data, (long long*)a->dTtrue_i.data, NH_ITBL * NH_INV_HID);
    puf_mm_tn(&a->dTtrue, &a->e_eff, &a->invt_wgrad, stream);

    // Blstats branch (raw-feature slice of concat has no upstream params)
    nh_slice_kernel<<<grid_size(B * NH_BL_HID), BLOCK_SIZE, 0, stream>>>(
        a->bl_grad.data, grad_concat.data, B, NH_CONCAT, NH_LOC_HID + NH_GLB_HID + NH_INVP_DIM, NH_BL_HID);
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
    // patch machinery gone: seed the embed grad at zero, downstream adds only
    cudaMemsetAsync(a->embed_wgrad.data, 0, (size_t)NH_GLYPH_VOCAB * NH_EMBED_DIM * sizeof(precision_t), stream);

    int dE_n = NH_GLYPH_VOCAB * NH_EMBED_DIM;
    // class-crop backward: dx_local scatters into the 9x8 class table only
    cudaMemsetAsync(a->locc_acc.data, 0, NH_LOCC_CLASSES * NH_LOCC_DIM * sizeof(long long), stream);
    nh_loc3_scatter_kernel<<<grid_size((int64_t)B * NH_LOC_IN), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->locc_acc.data, dx_local.data, a->crop_glyph.data, nh_locc_lut_dev, B);
    nh_fxp_to_precision_kernel<<<grid_size(NH_LOCC_CLASSES * NH_LOCC_DIM), BLOCK_SIZE, 0, stream>>>(
        a->locc_wgrad.data, (long long*)a->locc_acc.data, NH_LOCC_CLASSES * NH_LOCC_DIM);

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
    // sum channel: dih = 0.2*occ*g -> relu gate (+ss_b acc) -> ss grads;
    // its embed grads scatter directly (nh_sp2_dE); dkeys = pointer only.
    nh_sp2_dh_kernel<<<grid_size((int64_t)B * NH_SPELL_SLOTS * NH_SP2_DIM), BLOCK_SIZE, 0, stream>>>(
        a->sp2_dh.data, grad_concat.data, a->spell_idx.data, B);
    cudaMemsetAsync(a->ssb_acc.data, 0, NH_SP2_DIM * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_SPELL_SLOTS * NH_SP2_DIM, NH_SP2_DIM), BLOCK_SIZE, NH_SP2_DIM * sizeof(long long), stream>>>(
        a->sp2_dh.data, a->sp2_h.data, (long long*)a->ssb_acc.data,
        (int64_t)B * NH_SPELL_SLOTS * NH_SP2_DIM, NH_SP2_DIM);
    nh_fxp_to_precision_kernel<<<grid_size(NH_SP2_DIM), BLOCK_SIZE, 0, stream>>>(
        a->ss_bgrad.data, (long long*)a->ssb_acc.data, NH_SP2_DIM);
    { Prec dhf = {.data = a->sp2_dh.data, .shape = {B * NH_SPELL_SLOTS, NH_SP2_DIM}};
      Prec inf = {.data = a->spk_in.data, .shape = {B * NH_SPELL_SLOTS, NH_SPIN}};
      puf_mm_tn(&dhf, &inf, &a->ss_wgrad, stream); }
    nh_sp2_dE_kernel<<<grid_size(B * NH_SPELL_SLOTS * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dE_i.data, a->sp2_dh.data, ew->ss_w.data, a->spell_idx.data, B);
    nh_sp2_dk_kernel<<<grid_size(B * NH_SPELL_SLOTS * NH_SPKEY), BLOCK_SIZE, 0, stream>>>(
        a->spk_dkeys.data, nh_ptr_spkeygrad != NULL ? nh_ptr_spkeygrad->data : NULL,
        a->spk_keys.data, B);
    nh_spkey_dE_kernel<<<grid_size(B * NH_SPELL_SLOTS * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dE_i.data, a->spk_dkeys.data, ew->spk_w.data, a->spell_idx.data, B);
    { Prec dkf = {.data = a->spk_dkeys.data, .shape = {B * NH_SPELL_SLOTS, NH_SPKEY}};
      Prec inf = {.data = a->spk_in.data, .shape = {B * NH_SPELL_SLOTS, NH_SPIN}};
      puf_mm_tn(&dkf, &inf, &a->spk_wgrad, stream); }
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->dE_tmp.data, (long long*)a->dE_i.data, NH_GLYPH_VOCAB * NH_EMBED_DIM);
    nh_add_inplace_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->dE_tmp.data, dE_n);
    { // typed stream backward: pool -> score/value grads -> shared-embed scatter
    // monster stream
    nh_lab_pool_bwd_kernel<<<grid_size(B * NH_LAB_HEADS), BLOCK_SIZE, 0, stream>>>(
        a->lm_dh2.data, a->lm_ds.data, grad_concat.data, a->lm_attn.data,
        a->lm_h2.data, NH_LABM_CONCAT_OFF, B);
    Prec mdsf = {.data = a->lm_ds.data, .shape = {B * NH_LABK, NH_LAB_HEADS}};
    Prec mtokf = {.data = a->lm_tok.data, .shape = {B * NH_LABK, NH_LAB_IN}};
    puf_mm_tn(&mdsf, &mtokf, &a->lma_wgrad, stream);
    cudaMemsetAsync(a->lmab_acc.data, 0, NH_LAB_HEADS * sizeof(long long), stream);
    nh_lab_dab_kernel<<<grid_size((int64_t)B * NH_LABK * NH_LAB_HEADS), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->lmab_acc.data, a->lm_ds.data, (int64_t)B * NH_LABK * NH_LAB_HEADS);
    nh_fxp_to_precision_kernel<<<1, 32, 0, stream>>>(
        a->lma_bgrad.data, (long long*)a->lmab_acc.data, NH_LAB_HEADS);
    { Prec dtsf = {.data = a->lm_dts.data, .shape = {B * NH_LABK, NH_LAB_IN}};
      puf_mm_nn(&mdsf, &ew->lma_w, &dtsf, stream); }
    cudaMemsetAsync(a->lm2b_acc.data, 0, NH_LAB_HID * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID), BLOCK_SIZE, NH_LAB_HID * sizeof(long long), stream>>>(
        a->lm_dh2.data, a->lm_h2.data, (long long*)a->lm2b_acc.data, (int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    nh_fxp_to_precision_kernel<<<grid_size(NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->lm2_bgrad.data, (long long*)a->lm2b_acc.data, NH_LAB_HID);
    Prec mdh2f = {.data = a->lm_dh2.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    Prec mh1f = {.data = a->lm_h1.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm_tn(&mdh2f, &mh1f, &a->lm2_wgrad, stream);
    Prec mdh1f = {.data = a->lm_dh1.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm_nn(&mdh2f, &ew->lm2_w, &mdh1f, stream);
    cudaMemsetAsync(a->lm1b_acc.data, 0, NH_LAB_HID * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID), BLOCK_SIZE, NH_LAB_HID * sizeof(long long), stream>>>(
        a->lm_dh1.data, a->lm_h1.data, (long long*)a->lm1b_acc.data, (int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    nh_fxp_to_precision_kernel<<<grid_size(NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->lm1_bgrad.data, (long long*)a->lm1b_acc.data, NH_LAB_HID);
    puf_mm_tn(&mdh1f, &mtokf, &a->lm1_wgrad, stream);
    { Prec mdtokf = {.data = a->lm_tok.data, .shape = {B * NH_LABK, NH_LAB_IN}}; // in place; tok done
      puf_mm_nn(&mdh1f, &ew->lm1_w, &mdtokf, stream); }
    // item stream
    nh_lab_pool_bwd_kernel<<<grid_size(B * NH_LAB_HEADS), BLOCK_SIZE, 0, stream>>>(
        a->li_dh2.data, a->li_ds.data, grad_concat.data, a->li_attn.data,
        a->li_h2.data, NH_LABI_CONCAT_OFF, B);
    Prec idsf = {.data = a->li_ds.data, .shape = {B * NH_LABK, NH_LAB_HEADS}};
    Prec itokf = {.data = a->li_tok.data, .shape = {B * NH_LABK, NH_LAB_IN}};
    puf_mm_tn(&idsf, &itokf, &a->lia_wgrad, stream);
    cudaMemsetAsync(a->liab_acc.data, 0, NH_LAB_HEADS * sizeof(long long), stream);
    nh_lab_dab_kernel<<<grid_size((int64_t)B * NH_LABK * NH_LAB_HEADS), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->liab_acc.data, a->li_ds.data, (int64_t)B * NH_LABK * NH_LAB_HEADS);
    nh_fxp_to_precision_kernel<<<1, 32, 0, stream>>>(
        a->lia_bgrad.data, (long long*)a->liab_acc.data, NH_LAB_HEADS);
    { Prec dtsf = {.data = a->li_dts.data, .shape = {B * NH_LABK, NH_LAB_IN}};
      puf_mm_nn(&idsf, &ew->lia_w, &dtsf, stream); }
    cudaMemsetAsync(a->li2b_acc.data, 0, NH_LAB_HID * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID), BLOCK_SIZE, NH_LAB_HID * sizeof(long long), stream>>>(
        a->li_dh2.data, a->li_h2.data, (long long*)a->li2b_acc.data, (int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    nh_fxp_to_precision_kernel<<<grid_size(NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->li2_bgrad.data, (long long*)a->li2b_acc.data, NH_LAB_HID);
    Prec idh2f = {.data = a->li_dh2.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    Prec ih1f = {.data = a->li_h1.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm_tn(&idh2f, &ih1f, &a->li2_wgrad, stream);
    Prec idh1f = {.data = a->li_dh1.data, .shape = {B * NH_LABK, NH_LAB_HID}};
    puf_mm_nn(&idh2f, &ew->li2_w, &idh1f, stream);
    cudaMemsetAsync(a->li1b_acc.data, 0, NH_LAB_HID * sizeof(long long), stream);
    nh_relu_bias_bwd_kernel<<<nh_colsum_grid((int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID), BLOCK_SIZE, NH_LAB_HID * sizeof(long long), stream>>>(
        a->li_dh1.data, a->li_h1.data, (long long*)a->li1b_acc.data, (int64_t)B * NH_LABK * NH_LAB_HID, NH_LAB_HID);
    nh_fxp_to_precision_kernel<<<grid_size(NH_LAB_HID), BLOCK_SIZE, 0, stream>>>(
        a->li1_bgrad.data, (long long*)a->li1b_acc.data, NH_LAB_HID);
    puf_mm_tn(&idh1f, &itokf, &a->li1_wgrad, stream);
    { Prec idtokf = {.data = a->li_tok.data, .shape = {B * NH_LABK, NH_LAB_IN}};
      puf_mm_nn(&idh1f, &ew->li1_w, &idtokf, stream); }
    // shared-embed scatter: value-path dtok (in tok buffers) + score-path
    // dtok (dts buffers) for both streams, then one flush + add
    cudaMemsetAsync(a->dE_i.data, 0, (size_t)NH_GLYPH_VOCAB * NH_EMBED_DIM * sizeof(long long), stream);
    nh_lab_dE_scatter_kernel<<<grid_size(B * NH_LABK * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dE_i.data, a->lm_tok.data, a->lm_gid.data, B);
    nh_lab_dE_scatter_kernel<<<grid_size(B * NH_LABK * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dE_i.data, a->lm_dts.data, a->lm_gid.data, B);
    nh_lab_dE_scatter_kernel<<<grid_size(B * NH_LABK * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dE_i.data, a->li_tok.data, a->li_gid.data, B);
    nh_lab_dE_scatter_kernel<<<grid_size(B * NH_LABK * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        (long long*)a->dE_i.data, a->li_dts.data, a->li_gid.data, B);
    nh_fxp_to_precision_kernel<<<grid_size(NH_GLYPH_VOCAB * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->dE_tmp.data, (long long*)a->dE_i.data, NH_GLYPH_VOCAB * NH_EMBED_DIM);
    nh_add_inplace_kernel<<<grid_size(dE_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->dE_tmp.data, dE_n);
    }


    // dE_eff complete: embed_wgrad doubles as dE_res (identity path); factor
    // tables get deterministic CSR row sums of the same buffer
    nh_ekind_grad_kernel<<<grid_size(NH_NKIND * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->ekind_wgrad.data, a->embed_wgrad.data);
    nh_esub_grad_kernel<<<grid_size(NH_NSUB * NH_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->esub_wgrad.data, a->embed_wgrad.data);

    nh_bias_flush_kernel<<<grid_size(H + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_BACC_GLB1 + NH_INV_HID + NH_BACC_INVP), BLOCK_SIZE, 0, stream>>>(
        bacc, a->proj_bgrad.data, H,
        a->loc2_bgrad.data,
        NH_LOC_HID,
        a->terr2_bgrad.data,
        NH_GLB_HID, a->bl_bgrad.data, NH_BL_HID,
        (precision_t*)NULL, 0,
        a->inv1_bgrad.data, NH_INV_HID,
        (precision_t*)NULL, 0);
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
    puf_kaiming_init(&ew->loc2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->loc2_b.data, 0, numel(ew->loc2_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->iaq_w, 1.0f, (*seed)++, stream);
    puf_kaiming_init(&ew->terr1_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->terr1_b.data, 0, numel(ew->terr1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->terr2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->terr2_b.data, 0, numel(ew->terr2_b.shape) * sizeof(precision_t), stream);
    puf_normal_init(&ew->locc_w, 1.0f, (*seed)++, stream); // class table = embedding idiom
    puf_kaiming_init(&ew->inv1_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->inv1_b.data, 0, numel(ew->inv1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->inv1s_w, 1.0f, (*seed)++, stream);
    // zero: discovered-type channel starts as an exact no-op (ekind_w idiom)
    cudaMemsetAsync(ew->invt_w.data, 0, numel(ew->invt_w.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->isum_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->isum_b.data, 0, numel(ew->isum_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->bl_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->bl_b.data, 0, numel(ew->bl_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->proj_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->proj_b.data, 0, numel(ew->proj_b.shape) * sizeof(precision_t), stream);
    puf_normal_init(&ew->msg_w, 1.0f, (*seed)++, stream); // trigram embedding
    puf_kaiming_init(&ew->spk_w, 1.0f, (*seed)++, stream); // spell slot-rep projection
    puf_kaiming_init(&ew->ss_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->ss_b.data, 0, numel(ew->ss_b.shape) * sizeof(precision_t), stream);
    // zero: the identity channel starts as an exact no-op (ekind_w idiom)
    cudaMemsetAsync(ew->ide_role_w.data, 0, numel(ew->ide_role_w.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->ide_race_w.data, 0, numel(ew->ide_race_w.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->ide_gend_w.data, 0, numel(ew->ide_gend_w.shape) * sizeof(precision_t), stream);
    cudaMemsetAsync(ew->ide_algn_w.data, 0, numel(ew->ide_algn_w.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->lm1_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->lm1_b.data, 0, numel(ew->lm1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->lm2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->lm2_b.data, 0, numel(ew->lm2_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->lma_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->lma_b.data, 0, numel(ew->lma_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->li1_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->li1_b.data, 0, numel(ew->li1_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->li2_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->li2_b.data, 0, numel(ew->li2_b.shape) * sizeof(precision_t), stream);
    puf_kaiming_init(&ew->lia_w, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->lia_b.data, 0, numel(ew->lia_b.shape) * sizeof(precision_t), stream);
}

// Param and grad registration orders must match pairwise (muon walks both flat).
static void nethack_encoder_reg_params(void* w, Allocator* alloc) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    ew->embed_w = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    ew->ekind_w = {.shape = {NH_NKIND, NH_EMBED_DIM}}; // 14x32=448, mult of 8
    ew->esub_w = {.shape = {NH_NSUB, NH_EMBED_DIM}}; // 944x32=30208, mult of 8
    ew->loc_w = {.shape = {NH_LOC_H1, NH_LOC_IN}};
    ew->loc_b = {.shape = {NH_LOC_H1}};
    ew->loc2_w = {.shape = {NH_LOC_HID, NH_LOC_H1}};
    ew->loc2_b = {.shape = {NH_LOC_HID}};
    ew->iaq_w = {.shape = {NH_IVA_M, NH_INV_HID}};
    ew->terr1_w = {.shape = {NH_TERR_H1, NH_TERRF}}; // 256x592=151552, mult of 8
    ew->terr1_b = {.shape = {NH_TERR_H1}};
    ew->terr2_w = {.shape = {NH_GLB_HID, NH_TERR_H1}}; // 128x256, mult of 8
    ew->terr2_b = {.shape = {NH_GLB_HID}};
    ew->locc_w = {.shape = {NH_LOCC_CLASSES, NH_LOCC_DIM}}; // 9x8=72, mult of 8
    ew->inv1_w = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    ew->inv1_b = {.shape = {NH_INV_HID}};
    ew->inv1s_w = {.shape = {NH_INV_HID, NH_SFEAT}};
    ew->invt_w = {.shape = {NH_INV_HID, NH_EMBED_DIM}}; // 16x32=512, mult of 8
    ew->isum_w = {.shape = {NH_ISUM_DIM, NH_INV_HID}}; // 64x16=1024, mult of 8
    ew->isum_b = {.shape = {NH_ISUM_DIM}};
    ew->bl_w = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    ew->bl_b = {.shape = {NH_BL_HID}};
    ew->proj_w = {.shape = {ew->hidden, NH_CONCAT}};
    ew->proj_b = {.shape = {ew->hidden}};
    ew->msg_w = {.shape = {NH_MSG_VOCAB, NH_MSG_HID}}; // 4096x32=131072, mult of 8
    ew->spk_w = {.shape = {NH_SPKEY, NH_SPIN}}; // 16x36=576, mult of 8
    ew->ss_w = {.shape = {NH_SP2_DIM, NH_SPIN}}; // 32x36=1152, mult of 8
    ew->ss_b = {.shape = {NH_SP2_DIM}}; // 32, mult of 8
    ew->ide_role_w = {.shape = {13, NH_IDE_ROLE}}; // 208, mult of 8
    ew->ide_race_w = {.shape = {5, NH_IDE_RACE}}; // 40
    ew->ide_gend_w = {.shape = {2, NH_IDE_GEND}}; // 16
    ew->ide_algn_w = {.shape = {3, NH_IDE_ALGN}}; // 24
    ew->lm1_w = {.shape = {NH_LAB_HID, NH_LAB_IN}}; // 3072, mult of 8
    ew->lm1_b = {.shape = {NH_LAB_HID}};
    ew->lm2_w = {.shape = {NH_LAB_HID, NH_LAB_HID}};
    ew->lm2_b = {.shape = {NH_LAB_HID}};
    ew->lma_w = {.shape = {NH_LAB_HEADS, NH_LAB_IN}}; // 384
    ew->lma_b = {.shape = {NH_LAB_HEADS}};
    ew->li1_w = {.shape = {NH_LAB_HID, NH_LAB_IN}};
    ew->li1_b = {.shape = {NH_LAB_HID}};
    ew->li2_w = {.shape = {NH_LAB_HID, NH_LAB_HID}};
    ew->li2_b = {.shape = {NH_LAB_HID}};
    ew->lia_w = {.shape = {NH_LAB_HEADS, NH_LAB_IN}};
    ew->lia_b = {.shape = {NH_LAB_HEADS}};
    alloc_register(alloc,&ew->embed_w);
    alloc_register(alloc,&ew->ekind_w); alloc_register(alloc,&ew->esub_w);
    alloc_register(alloc,&ew->loc_w);   alloc_register(alloc,&ew->loc_b);
    alloc_register(alloc,&ew->loc2_w);  alloc_register(alloc,&ew->loc2_b);
    alloc_register(alloc,&ew->iaq_w);
    alloc_register(alloc,&ew->terr1_w); alloc_register(alloc,&ew->terr1_b);
    alloc_register(alloc,&ew->terr2_w); alloc_register(alloc,&ew->terr2_b);
    alloc_register(alloc,&ew->locc_w);
    alloc_register(alloc,&ew->inv1_w);  alloc_register(alloc,&ew->inv1_b);
    alloc_register(alloc,&ew->inv1s_w); alloc_register(alloc,&ew->invt_w);
    alloc_register(alloc,&ew->isum_w);  alloc_register(alloc,&ew->isum_b);
    alloc_register(alloc,&ew->bl_w);    alloc_register(alloc,&ew->bl_b);
    alloc_register(alloc,&ew->proj_w);  alloc_register(alloc,&ew->proj_b);
    alloc_register(alloc,&ew->msg_w);
    alloc_register(alloc,&ew->spk_w);
    alloc_register(alloc,&ew->ss_w);    alloc_register(alloc,&ew->ss_b);
    alloc_register(alloc,&ew->ide_role_w); alloc_register(alloc,&ew->ide_race_w);
    alloc_register(alloc,&ew->ide_gend_w); alloc_register(alloc,&ew->ide_algn_w);
    alloc_register(alloc,&ew->lm1_w); alloc_register(alloc,&ew->lm1_b);
    alloc_register(alloc,&ew->lm2_w); alloc_register(alloc,&ew->lm2_b);
    alloc_register(alloc,&ew->lma_w); alloc_register(alloc,&ew->lma_b);
    alloc_register(alloc,&ew->li1_w); alloc_register(alloc,&ew->li1_b);
    alloc_register(alloc,&ew->li2_w); alloc_register(alloc,&ew->li2_b);
    alloc_register(alloc,&ew->lia_w); alloc_register(alloc,&ew->lia_b);
}

static void nethack_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    *a = {};
    a->glyph_idx = {.shape = {B_TT, NH_MGRID}};
    a->crop_glyph = {.shape = {B_TT, NH_CGRID}};
    a->e_eff = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->x_local = {.shape = {B_TT, NH_LOC_IN}};
    a->terr_tf = {.shape = {B_TT, NH_TERRF}};
    a->terr_h = {.shape = {B_TT, NH_TERR_H1}};
    a->terr_dh = {.shape = {B_TT, NH_TERR_H1}};
    a->terr1b_acc = {.shape = {NH_TERR_H1}};
    a->isum_h = {.shape = {B_TT, NH_INV * NH_ISUM_DIM}};
    a->isum_dh = {.shape = {B_TT, NH_INV * NH_ISUM_DIM}};
    a->isumb_acc = {.shape = {NH_ISUM_DIM}};
    a->locc_acc = {.shape = {NH_LOCC_CLASSES * NH_LOCC_DIM}};
    a->inv_idx = {.shape = {B_TT, NH_INV}};
    a->spell_idx = {.shape = {B_TT, 8}};
    a->invt_idx = {.shape = {B_TT, NH_INV}};
    a->inv_sfeat = {.shape = {B_TT, NH_INV * NH_SFEAT}};
    a->inv_T = {.shape = {NH_ITBL, NH_INV_HID}};
    a->invt_T = {.shape = {NH_ITBL, NH_INV_HID}};
    a->inv_out = {.shape = {B_TT, NH_INV_FLAT}};
    a->loc_out = {.shape = {B_TT, NH_LOC_HID}};
    a->loc_h1 = {.shape = {B_TT, NH_LOC_H1}}; a->loc_h1_grad = {.shape = {B_TT, NH_LOC_H1}};
    a->loc1b_acc = {.shape = {NH_LOC_H1}};
    a->glb_out = {.shape = {B_TT, NH_GLB_HID}};
    a->bl_feats = {.shape = {B_TT, NH_BL_FEAT}};
    a->bl_out = {.shape = {B_TT, NH_BL_HID}};
    a->msg_ids = {.shape = {B_TT, NH_MSG_LEN}};
    a->msg_out = {.shape = {B_TT, NH_MSG_HID}};
    a->spk_in = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SPIN}};
    a->spk_keys = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SPKEY}};
    a->spk_dkeys = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SPKEY}};
    a->sp2_h = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SP2_DIM}};
    a->sp2_dh = {.shape = {B_TT, NH_SPELL_SLOTS * NH_SP2_DIM}};
    a->ssb_acc = {.shape = {NH_SP2_DIM}};
    a->ide_idx = {.shape = {B_TT, 4}};
    a->lm_tok = {.shape = {B_TT, NH_LABK * NH_LAB_IN}};
    a->lm_h1 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->lm_h2 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->lm_attn = {.shape = {B_TT, NH_LABK * NH_LAB_HEADS}};
    a->lm_gid = {.shape = {B_TT, NH_LABK}};
    a->li_tok = {.shape = {B_TT, NH_LABK * NH_LAB_IN}};
    a->li_h1 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->li_h2 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->li_attn = {.shape = {B_TT, NH_LABK * NH_LAB_HEADS}};
    a->li_gid = {.shape = {B_TT, NH_LABK}};
    a->concat = {.shape = {B_TT, NH_CONCAT}};
    a->out = {.shape = {B_TT, ew->hidden}};
    alloc_register(acts,&a->glyph_idx); alloc_register(acts,&a->crop_glyph);
    alloc_register(acts,&a->e_eff);
    alloc_register(acts,&a->x_local);
    alloc_register(acts,&a->terr_tf);   alloc_register(acts,&a->terr_h);
    alloc_register(acts,&a->terr_dh);   alloc_register(acts,&a->terr1b_acc);
    alloc_register(acts,&a->isum_h);    alloc_register(acts,&a->isum_dh);
    alloc_register(acts,&a->isumb_acc);
    alloc_register(acts,&a->locc_acc);
    alloc_register(acts,&a->inv_idx);   alloc_register(acts,&a->invt_idx);
    alloc_register(acts,&a->spell_idx);
    alloc_register(acts,&a->spk_in);    alloc_register(acts,&a->spk_keys);
    alloc_register(acts,&a->spk_dkeys);
    alloc_register(acts,&a->sp2_h);     alloc_register(acts,&a->sp2_dh);
    alloc_register(acts,&a->ssb_acc);
    alloc_register(acts,&a->ide_idx);
    alloc_register(acts,&a->inv_sfeat);
    alloc_register(acts,&a->inv_T);     alloc_register(acts,&a->invt_T);
    alloc_register(acts,&a->inv_out);
    alloc_register(acts,&a->loc_out);   alloc_register(acts,&a->glb_out);
    alloc_register(acts,&a->bl_feats);  alloc_register(acts,&a->bl_out);
    alloc_register(acts,&a->msg_ids);   alloc_register(acts,&a->msg_out);
    alloc_register(acts,&a->lm_tok); alloc_register(acts,&a->lm_h1);
    alloc_register(acts,&a->lm_h2);  alloc_register(acts,&a->lm_attn);
    alloc_register(acts,&a->lm_gid);
    alloc_register(acts,&a->li_tok); alloc_register(acts,&a->li_h1);
    alloc_register(acts,&a->li_h2);  alloc_register(acts,&a->li_attn);
    alloc_register(acts,&a->li_gid);
    alloc_register(acts,&a->concat);    alloc_register(acts,&a->out);
    a->loc_grad = {.shape = {B_TT, NH_LOC_HID}};
    a->glb_grad = {.shape = {B_TT, NH_GLB_HID}};
    a->iva_attn = {.shape = {B_TT, NH_IVA_M * NH_INV}};
    a->diaq_acc = {.shape = {NH_IVA_M * NH_INV_HID}}; a->iaq_wgrad = {.shape = {NH_IVA_M, NH_INV_HID}};
    a->inv_grad = {.shape = {B_TT, NH_INV_FLAT}};
    a->bl_grad = {.shape = {B_TT, NH_BL_HID}};
    a->dTinv = {.shape = {NH_ITBL, NH_INV_HID}};
    a->dTinv_i = {.shape = {NH_ITBL, NH_INV_HID}};
    a->dTtrue = {.shape = {NH_ITBL, NH_INV_HID}};
    a->dTtrue_i = {.shape = {NH_ITBL, NH_INV_HID}};
    a->dE_tmp = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->dE_i = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->loc2_wgrad = {.shape = {NH_LOC_HID, NH_LOC_H1}};
    a->loc2_bgrad = {.shape = {NH_LOC_HID}};
    a->dmsg_acc = {.shape = {NH_MSG_VOCAB * NH_MSG_HID}};
    a->lm_dh2 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->lm_dh1 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->lm_ds = {.shape = {B_TT, NH_LABK * NH_LAB_HEADS}};
    a->lm_dts = {.shape = {B_TT, NH_LABK * NH_LAB_IN}};
    a->li_dh2 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->li_dh1 = {.shape = {B_TT, NH_LABK * NH_LAB_HID}};
    a->li_ds = {.shape = {B_TT, NH_LABK * NH_LAB_HEADS}};
    a->li_dts = {.shape = {B_TT, NH_LABK * NH_LAB_IN}};
    a->lm1b_acc = {.shape = {NH_LAB_HID}}; a->lm2b_acc = {.shape = {NH_LAB_HID}};
    a->lmab_acc = {.shape = {NH_LAB_HEADS}};
    a->li1b_acc = {.shape = {NH_LAB_HID}}; a->li2b_acc = {.shape = {NH_LAB_HID}};
    a->liab_acc = {.shape = {NH_LAB_HEADS}};
    a->bias_acc = {.shape = {ew->hidden + NH_LOC_HID + NH_GLB_HID + NH_BL_HID + NH_P1 + NH_INV_HID + NH_INV_POOL}}; // superset of packed slots
    alloc_register(acts,&a->loc_grad);  alloc_register(acts,&a->glb_grad);
    alloc_register(acts,&a->iva_attn); alloc_register(acts,&a->diaq_acc);
    alloc_register(acts,&a->inv_grad);
    alloc_register(acts,&a->bl_grad);
    alloc_register(acts,&a->dTinv);     alloc_register(acts,&a->dTinv_i);
    alloc_register(acts,&a->dTtrue);    alloc_register(acts,&a->dTtrue_i);
    alloc_register(acts,&a->dE_tmp);    alloc_register(acts,&a->dE_i);
    alloc_register(acts,&a->loc_h1);    alloc_register(acts,&a->loc_h1_grad);
    alloc_register(acts,&a->loc1b_acc);
    alloc_register(acts,&a->dmsg_acc);
    alloc_register(acts,&a->bias_acc);
    alloc_register(acts,&a->lm_dh2); alloc_register(acts,&a->lm_dh1);
    alloc_register(acts,&a->lm_ds);  alloc_register(acts,&a->lm_dts);
    alloc_register(acts,&a->li_dh2); alloc_register(acts,&a->li_dh1);
    alloc_register(acts,&a->li_ds);  alloc_register(acts,&a->li_dts);
    alloc_register(acts,&a->lm1b_acc); alloc_register(acts,&a->lm2b_acc);
    alloc_register(acts,&a->lmab_acc);
    alloc_register(acts,&a->li1b_acc); alloc_register(acts,&a->li2b_acc);
    alloc_register(acts,&a->liab_acc);
    a->embed_wgrad = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->ekind_wgrad = {.shape = {NH_NKIND, NH_EMBED_DIM}};
    a->esub_wgrad = {.shape = {NH_NSUB, NH_EMBED_DIM}};
    a->loc_wgrad = {.shape = {NH_LOC_H1, NH_LOC_IN}};
    a->loc_bgrad = {.shape = {NH_LOC_H1}};
    a->terr1_wgrad = {.shape = {NH_TERR_H1, NH_TERRF}};
    a->terr1_bgrad = {.shape = {NH_TERR_H1}};
    a->terr2_wgrad = {.shape = {NH_GLB_HID, NH_TERR_H1}};
    a->terr2_bgrad = {.shape = {NH_GLB_HID}};
    a->locc_wgrad = {.shape = {NH_LOCC_CLASSES, NH_LOCC_DIM}};
    a->inv1_wgrad = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    a->inv1_bgrad = {.shape = {NH_INV_HID}};
    a->inv1s_wgrad = {.shape = {NH_INV_HID, NH_SFEAT}};
    a->invt_wgrad = {.shape = {NH_INV_HID, NH_EMBED_DIM}};
    a->isum_wgrad = {.shape = {NH_ISUM_DIM, NH_INV_HID}};
    a->isum_bgrad = {.shape = {NH_ISUM_DIM}};
    a->bl_wgrad = {.shape = {NH_BL_HID, NH_BL_FEAT}};
    a->bl_bgrad = {.shape = {NH_BL_HID}};
    a->proj_wgrad = {.shape = {ew->hidden, NH_CONCAT}};
    a->proj_bgrad = {.shape = {ew->hidden}};
    a->msg_wgrad = {.shape = {NH_MSG_VOCAB, NH_MSG_HID}};
    a->spk_wgrad = {.shape = {NH_SPKEY, NH_SPIN}};
    a->ss_wgrad = {.shape = {NH_SP2_DIM, NH_SPIN}};
    a->ss_bgrad = {.shape = {NH_SP2_DIM}};
    a->ide_role_wgrad = {.shape = {13, NH_IDE_ROLE}};
    a->ide_race_wgrad = {.shape = {5, NH_IDE_RACE}};
    a->ide_gend_wgrad = {.shape = {2, NH_IDE_GEND}};
    a->ide_algn_wgrad = {.shape = {3, NH_IDE_ALGN}};
    a->lm1_wgrad = {.shape = {NH_LAB_HID, NH_LAB_IN}}; a->lm1_bgrad = {.shape = {NH_LAB_HID}};
    a->lm2_wgrad = {.shape = {NH_LAB_HID, NH_LAB_HID}}; a->lm2_bgrad = {.shape = {NH_LAB_HID}};
    a->lma_wgrad = {.shape = {NH_LAB_HEADS, NH_LAB_IN}}; a->lma_bgrad = {.shape = {NH_LAB_HEADS}};
    a->li1_wgrad = {.shape = {NH_LAB_HID, NH_LAB_IN}}; a->li1_bgrad = {.shape = {NH_LAB_HID}};
    a->li2_wgrad = {.shape = {NH_LAB_HID, NH_LAB_HID}}; a->li2_bgrad = {.shape = {NH_LAB_HID}};
    a->lia_wgrad = {.shape = {NH_LAB_HEADS, NH_LAB_IN}}; a->lia_bgrad = {.shape = {NH_LAB_HEADS}};
    alloc_register(grads,&a->embed_wgrad);
    alloc_register(grads,&a->ekind_wgrad); alloc_register(grads,&a->esub_wgrad);
    alloc_register(grads,&a->loc_wgrad);   alloc_register(grads,&a->loc_bgrad);
    alloc_register(grads,&a->loc2_wgrad);  alloc_register(grads,&a->loc2_bgrad);
    alloc_register(grads,&a->iaq_wgrad);
    alloc_register(grads,&a->terr1_wgrad); alloc_register(grads,&a->terr1_bgrad);
    alloc_register(grads,&a->terr2_wgrad); alloc_register(grads,&a->terr2_bgrad);
    alloc_register(grads,&a->locc_wgrad);
    alloc_register(grads,&a->inv1_wgrad);  alloc_register(grads,&a->inv1_bgrad);
    alloc_register(grads,&a->inv1s_wgrad); alloc_register(grads,&a->invt_wgrad);
    alloc_register(grads,&a->isum_wgrad);  alloc_register(grads,&a->isum_bgrad);
    alloc_register(grads,&a->bl_wgrad);    alloc_register(grads,&a->bl_bgrad);
    alloc_register(grads,&a->proj_wgrad);  alloc_register(grads,&a->proj_bgrad);
    alloc_register(grads,&a->msg_wgrad);
    alloc_register(grads,&a->spk_wgrad);
    alloc_register(grads,&a->ss_wgrad);   alloc_register(grads,&a->ss_bgrad);
    alloc_register(grads,&a->ide_role_wgrad); alloc_register(grads,&a->ide_race_wgrad);
    alloc_register(grads,&a->ide_gend_wgrad); alloc_register(grads,&a->ide_algn_wgrad);
    alloc_register(grads,&a->lm1_wgrad); alloc_register(grads,&a->lm1_bgrad);
    alloc_register(grads,&a->lm2_wgrad); alloc_register(grads,&a->lm2_bgrad);
    alloc_register(grads,&a->lma_wgrad); alloc_register(grads,&a->lma_bgrad);
    alloc_register(grads,&a->li1_wgrad); alloc_register(grads,&a->li1_bgrad);
    alloc_register(grads,&a->li2_wgrad); alloc_register(grads,&a->li2_bgrad);
    alloc_register(grads,&a->lia_wgrad); alloc_register(grads,&a->lia_bgrad);
    nh_enc_last = a;
}

static void nethack_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NethackEncoderWeights* ew = (NethackEncoderWeights*)w;
    NethackEncoderActivations* a = (NethackEncoderActivations*)activations;
    a->glyph_idx = {.shape = {B, NH_MGRID}};
    a->crop_glyph = {.shape = {B, NH_CGRID}};
    a->e_eff = {.shape = {NH_GLYPH_VOCAB, NH_EMBED_DIM}};
    a->x_local = {.shape = {B, NH_LOC_IN}};
    a->terr_tf = {.shape = {B, NH_TERRF}};
    a->terr_h = {.shape = {B, NH_TERR_H1}};
    a->isum_h = {.shape = {B, NH_INV * NH_ISUM_DIM}};
    a->inv_idx = {.shape = {B, NH_INV}};
    a->invt_idx = {.shape = {B, NH_INV}};
    a->spell_idx = {.shape = {B, 8}};
    a->inv_sfeat = {.shape = {B, NH_INV * NH_SFEAT}};
    a->inv_T = {.shape = {NH_ITBL, NH_INV_HID}};
    a->invt_T = {.shape = {NH_ITBL, NH_INV_HID}};
    a->inv_out = {.shape = {B, NH_INV_FLAT}};
    a->loc_out = {.shape = {B, NH_LOC_HID}};
    a->loc_h1 = {.shape = {B, NH_LOC_H1}};
    a->iva_attn = {.shape = {B, NH_IVA_M * NH_INV}};
    a->glb_out = {.shape = {B, NH_GLB_HID}};
    a->bl_feats = {.shape = {B, NH_BL_FEAT}};
    a->bl_out = {.shape = {B, NH_BL_HID}};
    a->msg_ids = {.shape = {B, NH_MSG_LEN}};
    a->msg_out = {.shape = {B, NH_MSG_HID}};
    a->spk_in = {.shape = {B, NH_SPELL_SLOTS * NH_SPIN}};
    a->spk_keys = {.shape = {B, NH_SPELL_SLOTS * NH_SPKEY}};
    a->sp2_h = {.shape = {B, NH_SPELL_SLOTS * NH_SP2_DIM}};
    a->ide_idx = {.shape = {B, 4}};
    a->lm_tok = {.shape = {B, NH_LABK * NH_LAB_IN}};
    a->lm_h1 = {.shape = {B, NH_LABK * NH_LAB_HID}};
    a->lm_h2 = {.shape = {B, NH_LABK * NH_LAB_HID}};
    a->lm_attn = {.shape = {B, NH_LABK * NH_LAB_HEADS}};
    a->lm_gid = {.shape = {B, NH_LABK}};
    a->li_tok = {.shape = {B, NH_LABK * NH_LAB_IN}};
    a->li_h1 = {.shape = {B, NH_LABK * NH_LAB_HID}};
    a->li_h2 = {.shape = {B, NH_LABK * NH_LAB_HID}};
    a->li_attn = {.shape = {B, NH_LABK * NH_LAB_HEADS}};
    a->li_gid = {.shape = {B, NH_LABK}};
    a->concat = {.shape = {B, NH_CONCAT}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc,&a->glyph_idx); alloc_register(alloc,&a->crop_glyph);
    alloc_register(alloc,&a->e_eff);
    alloc_register(alloc,&a->x_local);
    alloc_register(alloc,&a->terr_tf);   alloc_register(alloc,&a->terr_h);
    alloc_register(alloc,&a->isum_h);
    alloc_register(alloc,&a->inv_idx);   alloc_register(alloc,&a->invt_idx);
    alloc_register(alloc,&a->spell_idx);
    alloc_register(alloc,&a->ide_idx);
    alloc_register(alloc,&a->inv_sfeat);
    alloc_register(alloc,&a->inv_T);     alloc_register(alloc,&a->invt_T);
    alloc_register(alloc,&a->inv_out);
    alloc_register(alloc,&a->loc_out);   alloc_register(alloc,&a->glb_out);
    alloc_register(alloc,&a->loc_h1);
    alloc_register(alloc,&a->iva_attn);
    alloc_register(alloc,&a->bl_feats);  alloc_register(alloc,&a->bl_out);
    alloc_register(alloc,&a->msg_ids);   alloc_register(alloc,&a->msg_out);
    alloc_register(alloc,&a->spk_in);    alloc_register(alloc,&a->spk_keys);
    alloc_register(alloc,&a->sp2_h);
    alloc_register(alloc,&a->lm_tok); alloc_register(alloc,&a->lm_h1);
    alloc_register(alloc,&a->lm_h2);  alloc_register(alloc,&a->lm_attn);
    alloc_register(alloc,&a->lm_gid);
    alloc_register(alloc,&a->li_tok); alloc_register(alloc,&a->li_h1);
    alloc_register(alloc,&a->li_h2);  alloc_register(alloc,&a->li_attn);
    alloc_register(alloc,&a->li_gid);
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
    Prec qn; // query norms (B, NH_QHEADS)
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

// slot logit = exp(ltau_h) * (qhat_h . k_i): query-only normalization — key
// magnitude reaches the logit (decoder lab: composite gear selection +3..21pp
// vs full cosine; adopted 2026-08-25, pair +386). One thread per (b, head, slot).
__global__ void nh_ptr3_cos_kernel(precision_t* __restrict__ slot_logits,
    const precision_t* __restrict__ q, const precision_t* __restrict__ qn,
    const precision_t* __restrict__ kmat,
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
        to_float(qn[(int64_t)b * NH_QHEADS + h]));
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
    const precision_t* __restrict__ tau, int B) {
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
        const precision_t* ki = kmat + ((int64_t)b * NH_INV + i) * NH_INV_HID;
        for (int k = 0; k < NH_INV_HID; k++)
            dv[k] += tauv * gi * to_float(ki[k]);
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
    const precision_t* __restrict__ kmat,
    const precision_t* __restrict__ tau, int B) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * NH_INV_FLAT) return;
    int64_t bi = idx / NH_INV_HID;
    int64_t b = bi / NH_INV;
    int i = (int)(bi % NH_INV);
    int k = (int)(idx % NH_INV_HID);
    float acc = 0.0f;
    for (int h = 0; h < NH_HEADS; h++) {
        int64_t gi_idx = b * (NH_DEC_OD + 1) + NH_ACTIONS + h * NH_INV + i;
        float gi = to_float(g[gi_idx]);
        if (gi == 0.0f) continue;
        float tauv = expf(to_float(tau[h]));
        float vk = to_float(q[((int64_t)b * NH_QHEADS + h) * NH_INV_HID + k])
                 / to_float(qn[(int64_t)b * NH_QHEADS + h]);
        acc += tauv * gi * vk;
    }
    dkmat[idx] = from_float(acc);
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
    nh_ptr3_cos_kernel<<<grid_size(B * NH_SLOT_OD), BLOCK_SIZE, 0, stream>>>(
        a->slot_logits.data, a->q.data, a->qn.data, a->kmat.data, dw->tau.data, B);
    nh_dec_assemble_kernel<<<grid_size(B * (NH_DEC_OD + 1)), BLOCK_SIZE, 0, stream>>>(
        a->out.data, a->tmp.data, a->slot_logits.data, a->q.data, ea->spk_keys.data, B);
    return a->out;
}

static Prec nethack_decoder_backward(void* w, void* activations,
    Float grad_logits, Float grad_logstd, Float grad_value, cudaStream_t stream) {
    (void)grad_logstd;
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
        a->q.data, a->qn.data, a->kmat.data, dw->tau.data, B);
    nh_fxp_to_precision_kernel<<<1, 32, 0, stream>>>(
        a->tau_grad.data, (long long*)a->tau_acc.data, NH_TAU_PAD);
    nh_spq_bwd_kernel<<<grid_size(B * NH_SPKEY), BLOCK_SIZE, 0, stream>>>(
        a->dq.data, a->spdk.data, a->grad_out.data, a->q.data, ea->spk_keys.data, B);
    nh_ptr3_dkmat_kernel<<<grid_size((int64_t)B * NH_INV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->dkmat.data, a->grad_out.data, a->out.data, a->q.data, a->qn.data,
        a->kmat.data, dw->tau.data, B);
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
    alloc_register(acts,&a->kmat);
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
    (void)w; // rollout shapes are all compile-time constants
    NethackDecoderActivations* a = (NethackDecoderActivations*)activations;
    a->enc = nh_enc_last;
    a->out = {.shape = {B, NH_DEC_OD + 1}};
    a->tmp = {.shape = {B, NH_DEC_PAD}};
    a->q = {.shape = {B, NH_QDIM}};
    a->kmat = {.shape = {B, NH_INV_FLAT}};
    a->qn = {.shape = {B, NH_QHEADS}};
    a->slot_logits = {.shape = {B, NH_SLOT_OD}};
    alloc_register(alloc,&a->out);
    alloc_register(alloc,&a->tmp);
    alloc_register(alloc,&a->q);
    alloc_register(alloc,&a->kmat);
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
