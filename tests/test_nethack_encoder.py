"""Numeric gradient check for the Nethack CUDA encoder + pointer decoder
(src/nethack.cu).

Builds tests/test_nethack_cuda.cu as a float shared lib and verifies the
analytic gradients from encoder backward() against central finite differences
of a scalar loss L = sum(out * g_out), plus exact float64 torch references for
the whole encoder and the 5-head pointer decoder (values, weight grads,
keygrad, grad_input).

Run: python tests/test_nethack_encoder.py
"""
import ctypes
import os
import glob
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
LIB = os.path.join(HERE, "nethack_test.so")

VP = ctypes.c_void_p

WEIGHT_NAMES = [
    "embed_w", "ekind_w", "esub_w", "bl_w", "bl_b", "proj_w", "proj_b", "loc_w", "loc_b",
    "glb1_w", "glb1_xy", "glb1_b", "glb2_w", "glb2_b",
    "inv1_w", "inv1_b", "inv1s_w", "invt_w", "inv2_w", "inv2_b", "msg_w", "spk_w", "spk2_w", "spk2_b",
    "dec_lin_w", "dec_q_w", "dec_k_w", "dec_tau",
]

# NH_TEST_IDEMB=1: build and check the identity-embedding arm (NH_ID_EMBED)
IDEMB = True
IDE_NAMES = ["ide_role_w", "ide_race_w", "ide_gend_w", "ide_algn_w"]
if IDEMB:
    WEIGHT_NAMES += IDE_NAMES
# NH_TEST_FILM=1: identity FiLM on the encoder output (implies IDEMB)
FILM = False
# NH_TEST_GMEAN=1: token-mean of the patch reps appended to the concat tail
GMEAN = False
# NH_TEST_LOC2=1: two-layer local branch (loc_w -> 512 hidden -> loc2_w -> 256)
LOC2 = True
# NH_TEST_TOPK=1: top-K salient patch tokens with coords appended to the concat
TOPK = False
# NH_TEST_GLBSG=1: no gradient from the global patch path into the embedding table
GLBSG = False
# NH_TEST_DIRPTR=1: direction heads get a pointer term over the 80 patch tokens
DIRPTR = False
# NH_TEST_INTRINS=1: 8 intrinsic bits appended to the blstats features
INTRINS = True
# NH_TEST_ATTNPOOL=1: softmax-attention pooled token + coords appended to the concat
ATTNPOOL = False
# NH_TEST_EFACTOR=1: E_res frozen (zero grad); NH_TEST_GSCALE=<f>: global dE scaled
EFACTOR = False
GSCALE = 1.0
# NH_TEST_SPLIT=1: terrain-global (own 128-row table) + entity-token branch
SPLIT = False
SPLITR = False
if SPLITR: SPLIT = True
# NH_TEST_THREAT=1: 16 nearest-hostile/threat features appended to blstats
THREAT = False
# NH_TEST_GLBPOS=1: each global channel's argmax patch (dx,dy) appended (256 dims)
GLBPOS = False
# NH_TEST_INVATTN=1: 4-query attention tail over the 55 slot reps
INVATTN = True
# NH_TEST_ENTMAX=1: entity pool = channel-wise max (attn query unused)
ENTMAX = False
# NH_TEST_V3=1: full typed-level encoder (implies SPLIT machinery for terrain)
V3 = False
if V3:
    SPLIT = True; SPLITR = True; THREAT = True
# NH_TEST_LAB=1: lab arm — typed token streams (deep values, 8-head pools,
# rank) + invattn8 + aux mega-spec heads (aux MSE folded into the FD loss).
# Leave-one-out toggles: NH_TEST_LAB_TOK/AUX/IVA=0 disable a component.
LAB = True
LAB_TOK = True
LAB_AUX = False
LAB_IVA = True
# NH_TEST_V5=1: finalized v5 encoder (ENCODER_V5.md) — class-crop local (LUT),
# terrain featurizer replacing patch, hazard token bits, intrinsics, inv2
# (wield readout + masked-sum channel replacing the inv max-pool). Aux OFF.
V5 = True
# NH_TEST_SPELL2=1: v5.1 spell fix — sum channel over RAW slot inputs + 4
# doorstep scalars replaces the spk2 max-pool. Implies V5.
SPELL2 = True

GMP = os.environ.get("NH_TEST_GMP") == "1"
GEN = os.environ.get("NH_TEST_GEN") == "1"
MIN = os.environ.get("NH_TEST_MIN", "1") == "1"  # min is the encoder now
APANEL = "-DNH_NO_APANEL" not in os.environ.get("NH_TEST_DEFS", "")  # champion default on
LINPOOL = "-DNH_ENT_LINPOOL" in os.environ.get("NH_TEST_DEFS", "")  # linear last entity layer before sum|max
def act_last(x):
    import torch as _t
    return x if LINPOOL else _t.relu(x)
def pool_max(v, keep, dim):
    import torch as _t
    m = v.masked_fill(keep < .5, -1e9).max(dim=dim).values
    return _t.where((keep > .5).any(dim=dim), m, _t.zeros_like(m)) if LINPOOL else _t.relu(m)
ACCOBS = os.environ.get("NH_TEST_ACC", "1") == "1"  # worn rings/amulets/eyewear + armor in inventory (exercises the panels)
import re as _re0
_ARMOR_OTYPS = np.flatnonzero(np.array([int(x) for x in _re0.findall(r"-?\d+", _re0.search(
    r"nh_obj_armcat\[NH_NUM_OBJECTS\] = \{(.*?)\};", open("ocean/nethack/netlib.h").read(), _re0.S).group(1))]) >= 0)
MSGH = int(os.environ.get("NH_TEST_MSGH", "256"))  # champion msg width
KT = int(os.environ.get("NH_TEST_K", "16"))  # nearest-token cap (NETHACK_V3_K)
def entact(x):
    import torch as _t
    return _t.relu(x)
if SPELL2:
    V5 = True
if V5:
    LAB = True; LAB_TOK = True; LAB_IVA = True; LAB_AUX = False
    INTRINS = True; LOC2 = True
if LAB and LAB_IVA:
    INVATTN = True
if V5:
    for _nm in ["glb1_w", "glb1_xy", "glb1_b", "glb2_w", "glb2_b", "inv2_w", "inv2_b"]:
        WEIGHT_NAMES.remove(_nm)
    WEIGHT_NAMES += ["terr1_w", "terr1_b", "terr2_w", "terr2_b", "locc_w"]
if GMP:
    for _nm in ["isum_w", "isum_b", "iaq_w"]:
        if _nm in WEIGHT_NAMES: WEIGHT_NAMES.remove(_nm)
    WEIGHT_NAMES += ["gws_w", "gv_w", "gv_b", "gtau"]
if GEN:
    for _nm in ["isum_w", "isum_b", "iaq_w"]:
        if _nm in WEIGHT_NAMES: WEIGHT_NAMES.remove(_nm)
    WEIGHT_NAMES += ["gln_g", "gln_b", "gnv_w", "gnv_b", "gns_w"]
if MIN:
    for _nm in ["isum_w", "isum_b", "iaq_w"]:
        if _nm in WEIGHT_NAMES: WEIGHT_NAMES.remove(_nm)
    WEIGHT_NAMES += ["mv1_w", "mv1_b", "mv2_w", "mv2_b"]
if SPELL2:
    WEIGHT_NAMES.remove("spk2_w"); WEIGHT_NAMES.remove("spk2_b")
    WEIGHT_NAMES += ["spm1_w", "spm1_b", "spm2_w", "spm2_b"]
if SPLIT:
    WEIGHT_NAMES += ["eterr_w", "tglb1_w", "tglb1_xy", "tglb1_b", "tglb2_w", "tglb2_b"]
    if not V3:  # mixed-ent branch is disabled under v3 (typed lists replace it)
        WEIGHT_NAMES += ["ent1_w", "ent1_b", "entq_w", "entb"]
if ATTNPOOL:
    WEIGHT_NAMES += ["apq_w", "apb"]
if INVATTN and not (MIN or GMP or GEN):
    WEIGHT_NAMES += ["iaq_w"]
if V3:
    WEIGHT_NAMES += ["emon_w", "eitem_w", "eterrc_w", "mon1_w", "mon1_b", "monq_w", "monb"]
LAB_NAMES = ((["mr_w", "mr_b", "mm1_w", "mm1_b", "mm2_w", "mm2_b",
               "ir_w", "ir_b", "im1_w", "im1_b", "im2_w", "im2_b"] if MIN
              else ["lm1_w", "lm1_b", "lm2_w", "lm2_b", "lma_w", "lma_b",
                    "li1_w", "li1_b", "li2_w", "li2_b", "lia_w", "lia_b"]) if LAB_TOK else []) \
          + (["aux_w"] if LAB_AUX else [])
if LAB:
    WEIGHT_NAMES += LAB_NAMES
if DIRPTR:
    WEIGHT_NAMES += ["dec_qd_w", "dec_kd_w", "dec_taud"]
_G = {}
if TOPK:
    WEIGHT_NAMES += ["sal_w", "sal_b"]
if LOC2:
    WEIGHT_NAMES += ["loc2_w", "loc2_b"]
if FILM:
    assert IDEMB, "NH_TEST_FILM needs NH_TEST_IDEMB=1"
    WEIGHT_NAMES += ["film_g_w", "film_b_w"]


def build():
    root = os.path.dirname(HERE)
    cuda = os.path.dirname(os.path.dirname(shutil.which("nvcc")))
    raylib = glob.glob(os.path.join(root, "raylib-*"))[0]
    cmd = [
        "nvcc", "-shared", "-o", LIB, os.path.join(HERE, "test_nethack_cuda.cu"),
        "-std=c++17", "-arch=native",
        "-I" + root, "-I" + os.path.join(root, "src"),
        "-I" + os.path.join(root, "ocean", "nethack"),
        "-I" + os.path.join(root, "vendor"),
        "-I" + os.path.join(root, "vendor", "fast-nle", "include"),
        "-I" + os.path.join(root, "vendor", "fast-nle", "build",
                            "_deps", "deboost_context-src", "include"),
        "-I" + os.path.join(cuda, "include"),
        "-I" + os.path.join(cuda, "include", "cccl"),
        "-I" + os.path.join(raylib, "include"),
        '-DENV_HEADER="ocean/nethack/nethack.h"',
        "-DPUFFER_NETHACK", "-DENV_NAME=nethack", '-DPUFFER_ENV_NAME="nethack"',
    ] + [d for d in os.environ.get("NH_TEST_DEFS", "").split() if d] + [
        "-Xcompiler=-DPLATFORM_DESKTOP", "-Xcompiler=-fPIC",
        "-Xcompiler=-fopenmp", "-O2",
        "-L" + os.path.join(root, "vendor", "fast-nle", "build"), "-lnethack",
        "-Xlinker", "-rpath", "-Xlinker",
        os.path.join(root, "vendor", "fast-nle", "build"),
        "-L" + os.path.join(raylib, "lib"), "-lraylib",
        "-Xlinker", "-rpath", "-Xlinker", os.path.join(raylib, "lib"),
        "-lcublas", "-lcublasLt", "-lcusolver", "-lcurand", "-lnvidia-ml", "-lcudart",
    ]
    try:
        import nvidia.nccl
        nccl = nvidia.nccl.__path__[0]
        cmd += ["-I" + os.path.join(nccl, "include"),
                "-L" + os.path.join(nccl, "lib"), "-lnccl"]
    except ImportError:
        cmd += ["-lnccl"]
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if os.environ.get("NH_TEST_BUILD_ONLY"):
        print("BUILD ONLY OK"); sys.exit(0)


def load():
    lib = ctypes.CDLL(LIB)
    for name in WEIGHT_NAMES:
        for fn in [f"nh_get_{name}", f"nh_set_{name}"]:
            getattr(lib, fn).argtypes = [VP]
            getattr(lib, fn).restype = None
        if name.startswith("dec_"):
            getattr(lib, f"nh_grad_{name}").argtypes = [VP]
        getattr(lib, f"nh_numel_{name}").restype = ctypes.c_int
    for name in WEIGHT_NAMES:
        if not name.startswith("dec_"):
            getattr(lib, f"nh_grad_{name}").argtypes = [VP]
            getattr(lib, f"nh_grad_{name}").restype = None
    lib.nh_init.argtypes = [ctypes.c_int, ctypes.c_int]
    for fn in ["nh_obs_size", "nh_bl_feat", "nh_glyph_vocab", "nh_embed_dim",
               "nh_concat", "nh_grid", "nh_dec_od", "nh_heads"]:
        getattr(lib, fn).restype = ctypes.c_int
    lib.nh_forward.argtypes = [VP, VP, ctypes.c_int]
    lib.nh_backward.argtypes = [VP, ctypes.c_int]
    lib.nh_dec_forward.argtypes = [VP, VP, ctypes.c_int]
    lib.nh_dec_backward.argtypes = [VP, VP, VP, ctypes.c_int]
    lib.nh_dec_keygrad.argtypes = [VP, ctypes.c_int]
    if DIRPTR:
        lib.nh_dec_tokgrad.argtypes = [VP, ctypes.c_int]
    return lib


def make_obs(B, obs_size, grid, max_glyph_used):
    """Build a valid packed obs: glyphs int16 LE, then blstats int32 LE, as
    byte-valued float32 (matching cast_dispatch's per-byte float cast).
    Returns (obs, glyphs, bl_vals, ex_vals) so the torch reference can rebuild
    the exact inputs."""
    obs = np.zeros((B, obs_size), dtype=np.float32)
    rng = np.random.default_rng(0)
    # glyphs @0: grid cells, 2 bytes each (restrict to a small glyph set so many
    # embedding rows receive gradient and are individually checkable).
    glyphs = rng.integers(0, max_glyph_used, size=(B, grid)).astype(np.int32)
    lo = (glyphs & 0xFF).astype(np.float32)
    hi = ((glyphs >> 8) & 0xFF).astype(np.float32)
    obs[:, 0:2 * grid:2] = lo
    obs[:, 1:2 * grid:2] = hi
    # blstats @ 2*grid: 27 int32, mixed magnitudes incl. negatives (AC/align).
    bl_off = 2 * grid
    vals = rng.integers(-5, 500, size=(B, 27)).astype(np.int64)
    vals[:, 0] = rng.integers(0, 79, size=B)     # hero x: crop center
    vals[:, 1] = rng.integers(0, 21, size=B)     # hero y
    vals[:, 25] = rng.integers(0, 8192, size=B)  # CONDITION bitmask
    u = vals.astype(np.uint32)
    for k in range(4):
        obs[:, bl_off + k::4][:, :27] = ((u >> (8 * k)) & 0xFF).astype(np.float32)
    # extra stats @ +27*4: engraving state, prev action (-1..21 valid; sampled
    # -1..13 to preserve the original FD test batch — the higher onehot columns
    # are linear and covered by the analytic torch check), 18 class counts
    spell_cols = []
    for _ in range(8):
        spell_cols += [
            rng.integers(0, 500, size=(B, 1)),    # slot id (otyp; 0 = empty)
            rng.integers(0, 8, size=(B, 1)),      # slot level
            rng.integers(0, 101, size=(B, 1)),    # slot fail%
            rng.integers(0, 20001, size=(B, 1)),  # slot retention turns
        ]
    # identity one-hots (challenge layout): exactly one bit per block
    oh = np.zeros((B, 20), dtype=np.int64)
    oh[np.arange(B), rng.integers(0, 13, size=B)] = 1
    oh[np.arange(B), 13 + rng.integers(0, 5, size=B)] = 1
    oh[np.arange(B), 18 + rng.integers(0, 2, size=B)] = 1
    ex = np.concatenate([
        rng.integers(0, 3, size=(B, 1)),      # engraving state 0/1/2
        rng.integers(-1, 14, size=(B, 1)),
        rng.integers(0, 6, size=(B, 18)),
        rng.integers(0, 2, size=(B, 1)),      # in-shop bit
        rng.integers(0, 101, size=(B, 1)),    # affordability percent
        rng.integers(0, 9, size=(B, 1)),      # known-spell count
    ] + spell_cols + [
        rng.integers(0, 320, size=(B, 1)),    # encumbrance percent (unclipped)
        rng.integers(50, 1001, size=(B, 1)),  # carry capacity
        oh,
    ] + ([rng.integers(0, 256, size=(B, 1))] if INTRINS else [])
      + ([np.concatenate([rng.integers(0, 16, size=(B, 1)),   # dist (15 = none)
                          rng.integers(0, 9, size=(B, 1)),    # bearing (8 = none)
                          rng.integers(0, 9, size=(B, 1)),    # adjacent count
                          rng.integers(0, 9, size=(B, 1)),    # within-3 count
                          rng.integers(0, 30, size=(B, 1)),   # visible count
                          rng.integers(0, 58, size=(B, 1)),   # difficulty
                          rng.integers(0, 37, size=(B, 1))], axis=1)] if THREAT else [])
      + ([np.concatenate([rng.integers(0, 2, size=(B, 1)),     # aux: ds_seen
                          rng.integers(-39, 40, size=(B, 1)),  # ds_dx
                          rng.integers(-10, 11, size=(B, 1)),  # ds_dy
                          rng.integers(0, 16, size=(B, 1)),    # nh_d
                          rng.integers(-39, 40, size=(B, 1)),  # nh_dx
                          rng.integers(-10, 11, size=(B, 1)),  # nh_dy
                          rng.integers(0, 30, size=(B, 1)),    # nh vis
                          rng.integers(0, 26, size=(B, 1)),    # n_items
                          rng.integers(0, 2, size=(B, 1)),     # has_food
                          rng.integers(-1, 18, size=(B, 1)),   # wield_class
                          rng.integers(-39, 40, size=(B, 1)),  # nm1_dx
                          rng.integers(-10, 11, size=(B, 1)),  # nm1_dy
                          np.where(rng.random((B, 1)) < 0.2, 127,
                                   rng.integers(0, 40, size=(B, 1))),  # nm1_d (127 = none)
                          rng.integers(-39, 40, size=(B, 1)),  # nm2_dx
                          rng.integers(-10, 11, size=(B, 1)),  # nm2_dy
                          np.where(rng.random((B, 1)) < 0.3, 127,
                                   rng.integers(0, 40, size=(B, 1))),  # nm2_d
                          rng.integers(0, 30, size=(B, 1)),    # mcnt
                          rng.integers(0, 12, size=(B, 1)),    # qNW
                          rng.integers(0, 12, size=(B, 1)),    # qNE
                          rng.integers(0, 12, size=(B, 1)),    # qSW
                          rng.integers(0, 12, size=(B, 1)),    # qSE
                          rng.integers(-39, 40, size=(B, 1)),  # ni_dx
                          rng.integers(-10, 11, size=(B, 1)),  # ni_dy
                          np.where(rng.random((B, 1)) < 0.3, 127,
                                   rng.integers(0, 40, size=(B, 1))),  # ni_d
                          rng.integers(0, 30, size=(B, 1)),    # icnt
                          rng.integers(0, 8, size=(B, 1))], axis=1)] if LAB and LAB_AUX else []),  # lnc bits
        axis=1).astype(np.int64).astype(np.uint32)
    NEX = 27 + ex.shape[1]
    for k in range(4):
        obs[:, bl_off + k::4][:, 27:NEX] = ((ex >> (8 * k)) & 0xFF).astype(np.float32)
    # inventory entities: 55 slot glyphs int16 LE, tail padded (5976)
    inv_off = bl_off + NEX * 4
    inv = rng.integers(0, max_glyph_used, size=(B, 55)).astype(np.int32)
    inv[:, ::2] = rng.integers(1906, 2359, size=(B, 28))  # object glyphs: armcat coverage
    n_items = rng.integers(3, 12, size=B)
    for b in range(B):
        inv[b, n_items[b]:] = 5976
        if ACCOBS:  # accessory otyps (rings 150-177, amulets 178-188, eyewear 207-209) in some slots
            for k in range(n_items[b]):
                if rng.random() < 0.25:
                    inv[b, k] = 1906 + int(rng.choice(np.r_[150:189, 207:210]))
                elif rng.random() < 0.25:  # armor otyps (any subclass in the baked armcat table)
                    inv[b, k] = 1906 + int(rng.choice(_ARMOR_OTYPS))
    obs[:, inv_off + 0::2][:, :55] = (inv & 0xFF).astype(np.float32)
    obs[:, inv_off + 1::2][:, :55] = ((inv >> 8) & 0xFF).astype(np.float32)
    # per-slot item state @ +55*2: 8 int8 fields, incl. the -128 spe sentinel
    st_off = inv_off + 55 * 2
    st = np.zeros((B, 55, 8), dtype=np.int64)
    st[:, :, 0] = rng.integers(0, 4, size=(B, 55))
    spe_known = rng.integers(0, 2, size=(B, 55)).astype(bool)
    st[:, :, 1] = np.where(spe_known, rng.integers(-3, 6, size=(B, 55)), -128)
    st[:, :, 2] = rng.integers(1, 20, size=(B, 55))
    st[:, :, 3] = rng.integers(0, 4, size=(B, 55))
    st[:, :, 4] = rng.integers(0, 4, size=(B, 55))
    st[:, :, 5] = rng.integers(0, 128, size=(B, 55))
    st[:, :, 6] = rng.integers(0, 2, size=(B, 55))
    obs[:, st_off:st_off + 55 * 8] = (st.reshape(B, -1) & 0xFF).astype(np.float32)
    # discovered-type glyphs @ +55*8: true otyp glyph on a random identified
    # subset, pad (5976) elsewhere and past the item tail
    itr_off = st_off + 55 * 8
    itr = np.full((B, 55), 5976, dtype=np.int32)
    known = rng.integers(0, 2, size=(B, 55)).astype(bool)
    itr[known] = rng.integers(1906, 2359, size=int(known.sum())).astype(np.int32)
    for b in range(B):
        itr[b, n_items[b]:] = 5976
    obs[:, itr_off + 0::2][:, :55] = (itr & 0xFF).astype(np.float32)
    obs[:, itr_off + 1::2][:, :55] = ((itr >> 8) & 0xFF).astype(np.float32)
    # trigram message @ msg_off: raw topline chars (null-padded). Random
    # lowercase words so the char-trigram bag hits many buckets.
    msg_off = itr_off + 55 * 2
    msg_len = 128  # fixed NETHACK_MSG_LEN; the split planes sit after it
    if V3:
        # planes: terr (GRID) | objm (GRID u16) | vmon 16x8 | vitem 16x8
        terr_off = obs_size - (21 * 79 * 3 + 2 * KT * 8)
        tb = rng.integers(0, 97, size=(B, 21 * 79))
        obs[:, terr_off:terr_off + 21 * 79] = tb.astype(np.float32)
        om_off = terr_off + 21 * 79
        objm = np.where(rng.random(size=(B, 21 * 79)) < 0.03,
                        rng.integers(1, 835, size=(B, 21 * 79)), 0).astype(np.int64)
        obs[:, om_off + 0:om_off + 2 * 21 * 79:2] = (objm & 0xFF).astype(np.float32)
        obs[:, om_off + 1:om_off + 2 * 21 * 79:2] = ((objm >> 8) & 0xFF).astype(np.float32)
        vm_off = om_off + 2 * 21 * 79
        vmon = np.zeros((B, KT, 8), dtype=np.int64)
        vitem = np.zeros((B, KT, 8), dtype=np.int64)
        for b in range(B):
            for k in range(int(rng.integers(0, KT + 1))):
                sp = int(rng.integers(1, 382))
                dx = int(rng.integers(-39, 40)); dy = int(rng.integers(-10, 11))
                fl = int(rng.integers(0, 16))
                vmon[b, k] = [sp & 0xFF, (sp >> 8) & 0xFF, dx & 0xFF, dy & 0xFF, fl,
                              int(rng.integers(0, 58)), int(rng.integers(0, 37)), 0]
            for k in range(int(rng.integers(0, KT + 1))):
                row = int(rng.integers(1, 835))
                dx = int(rng.integers(-39, 40)); dy = int(rng.integers(-10, 11))
                vitem[b, k] = [row & 0xFF, (row >> 8) & 0xFF, dx & 0xFF, dy & 0xFF, 0,
                               int(rng.integers(0, 4)), 0, 0]
        obs[:, vm_off:vm_off + KT * 8] = vmon.reshape(B, -1).astype(np.float32)
        obs[:, vm_off + KT * 8:vm_off + 2 * KT * 8] = vitem.reshape(B, -1).astype(np.float32)
        make_obs.terr = tb; make_obs.objm = objm
        make_obs.vmon = vmon; make_obs.vitem = vitem
    elif SPLIT:
        terr_off = obs_size - (21 * 79 + 32 * 6)
        tb = rng.integers(0, 97, size=(B, 21 * 79))
        obs[:, terr_off:terr_off + 21 * 79] = tb.astype(np.float32)
        eoff = terr_off + 21 * 79
        ents = np.zeros((B, 32, 6), dtype=np.int64)
        for b in range(B):
            ne = rng.integers(0, 33)
            for k in range(ne):
                g = int(rng.integers(1, 381)) if rng.random() < 0.6 else int(rng.integers(1906, 2359))
                dx = int(rng.integers(-39, 40)); dy = int(rng.integers(-10, 11))
                fl = int(rng.integers(0, 8))
                ents[b, k] = [g & 0xFF, (g >> 8) & 0xFF, dx & 0xFF, dy & 0xFF, fl, 0]
        obs[:, eoff:eoff + 32 * 6] = ents.reshape(B, -1).astype(np.float32)
        make_obs.terr = tb; make_obs.ents = ents
    if LAB:
        # lean token lists at the obs tail: vmon 16x8 | vitem 16x8
        tm_off = obs_size - 2 * KT * 8
        labm = np.zeros((B, KT, 8), dtype=np.int64)
        labi = np.zeros((B, KT, 8), dtype=np.int64)
        DXR, DYR = (61, 16) if V5 else (40, 11)  # V5 widened: exercise the clamps
        for b in range(B):
            for k in range(int(rng.integers(0, KT + 1))):
                sp = int(rng.integers(1, 382))
                dx = int(rng.integers(-DXR + 1, DXR)); dy = int(rng.integers(-DYR + 1, DYR))
                fl = int(rng.integers(0, 16))
                labm[b, k] = [sp & 0xFF, (sp >> 8) & 0xFF, dx & 0xFF, dy & 0xFF, fl,
                              int(rng.integers(0, 58)), int(rng.integers(0, 37)), 0]
            for k in range(int(rng.integers(0, KT + 1))):
                row = int(rng.integers(1, 835))
                dx = int(rng.integers(-DXR + 1, DXR)); dy = int(rng.integers(-DYR + 1, DYR))
                labi[b, k] = [row & 0xFF, (row >> 8) & 0xFF, dx & 0xFF, dy & 0xFF, 0,
                              int(rng.integers(0, 4)), 0, 0]
        obs[:, tm_off:tm_off + KT * 8] = labm.reshape(B, -1).astype(np.float32)
        obs[:, tm_off + KT * 8:tm_off + 2 * KT * 8] = labi.reshape(B, -1).astype(np.float32)
        make_obs.labm = labm; make_obs.labi = labi
    msg = np.zeros((B, msg_len), dtype=np.int64)
    alpha = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz ", dtype=np.uint8).astype(np.int64)
    for b in range(B):
        ln = int(rng.integers(6, min(40, msg_len)))
        msg[b, :ln] = alpha[rng.integers(0, len(alpha), size=ln)]
    obs[:, msg_off:msg_off + msg_len] = msg.astype(np.float32)
    return obs, glyphs, vals, ex.astype(np.int64).astype(np.int32), inv, st, itr, msg


def dev(nbytes):
    import ctypes
    p = VP()
    _cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes))
    return p


def h2d(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    p = dev(arr.nbytes)
    _cudart.cudaMemcpy(p, arr.ctypes.data_as(VP), ctypes.c_size_t(arr.nbytes), 1)  # H2D
    return p, arr.nbytes


def d2h(p, n):
    out = np.empty(n, dtype=np.float32)
    _cudart.cudaMemcpy(out.ctypes.data_as(VP), p, ctypes.c_size_t(n * 4), 2)  # D2H
    return out


_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaMalloc.argtypes = [VP, ctypes.c_size_t]
_cudart.cudaMemcpy.argtypes = [VP, VP, ctypes.c_size_t, ctypes.c_int]


def glyph_map():
    """Parse the generated (kind, sub) mapping straight from the header."""
    import re
    txt = open(os.path.join(HERE, "..", "ocean", "nethack", "glyph_map.h")).read()
    def arr(name):
        m = re.search(name + r"\[\d+\] = \{([0-9,\-]+)\};", txt)
        return np.array([int(x) for x in m.group(1).split(",") if x], dtype=np.int64)
    return arr("nh_glyph_kind"), arr("nh_glyph_sub")


def getw(lib, name, shape):
    import torch
    fn = getattr(lib, f"nh_get_{name}")
    n = getattr(lib, f"nh_numel_{name}")()
    a = np.empty(n, dtype=np.float32)
    fn(a.ctypes.data_as(VP))
    return torch.tensor(a.astype(np.float64).reshape(shape), requires_grad=True)


def torch_encoder(lib, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, H):
    """float64 torch replica of the encoder forward. Returns (out, invh, w)
    where invh is the (B,55,16) post-relu slot features (the decoder's keys)
    and w maps weight names to the torch leaf tensors."""
    import torch
    B = glyphs.shape[0]
    ROWS, COLS, CROP, PAD = 21, 79, 9, 5976
    PW, PH, PX, PY = 5, 5, 16, 5
    BL_SCALE = np.array([
        1/79, 1/21, 1/25, 1/125, 1/25, 1/25, 1/25, 1/25, 1/25, 0.1,
        1/200, 1/200, 1/50, 0.1, 1/100, 1/100, 1/10, 1/10, 1/30,
        0.1, 0.1, 0.0, 1/4, 0.0, 1/50, 0.0, 1.0], dtype=np.float64)
    BL_ISLOG = np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0])

    w = {}
    # widths from the built lib, so width arms (loc/glb/inv/bl/embed) check as-is
    D = lib.nh_embed_dim()
    LH = lib.nh_numel_loc_b()
    if V5:
        P1 = 16; GH = lib.nh_numel_terr2_b(); IP = 0
    else:
        P1 = lib.nh_numel_glb1_b(); GH = lib.nh_numel_glb2_b(); IP = lib.nh_numel_inv2_b()
    IH = lib.nh_numel_inv1_b(); BH = lib.nh_numel_bl_b()
    SK = 16; SI = lib.nh_numel_spk_w() // SK
    E_res  = w["embed_w"] = getw(lib, "embed_w", (5977, D))
    K_w    = w["ekind_w"] = getw(lib, "ekind_w", (14, D))
    S_w    = w["esub_w"]  = getw(lib, "esub_w", (getattr(lib, "nh_numel_esub_w")() // D, D))
    kind_map, sub_map = glyph_map()
    if EFACTOR:
        E_res = E_res.detach()  # residual frozen
    E = E_res + K_w[torch.tensor(kind_map)] + S_w[torch.tensor(sub_map)]   # E_eff
    loc_w  = w["loc_w"]   = getw(lib, "loc_w", (LH, CROP * CROP * 8 + 10 * D if V3
                                                else CROP * CROP * 8 if V5 else CROP * CROP * D))
    loc_b  = w["loc_b"]   = getw(lib, "loc_b", (LH,))
    if not V5:
        g1_w   = w["glb1_w"]  = getw(lib, "glb1_w", (P1, PW * PH * D))
        g1_b   = w["glb1_b"]  = getw(lib, "glb1_b", (P1,))
        g1_xy  = w["glb1_xy"] = getw(lib, "glb1_xy", (P1, 2))
        g2_w   = w["glb2_w"]  = getw(lib, "glb2_w", (GH, P1))
        g2_b   = w["glb2_b"]  = getw(lib, "glb2_b", (GH,))
    else:
        TH1 = lib.nh_numel_terr1_b()
        t1_w = w["terr1_w"] = getw(lib, "terr1_w", (TH1, lib.nh_terrf()))
        t1_b = w["terr1_b"] = getw(lib, "terr1_b", (TH1,))
        t2_w = w["terr2_w"] = getw(lib, "terr2_w", (GH, TH1))
        t2_b = w["terr2_b"] = getw(lib, "terr2_b", (GH,))
        locc = w["locc_w"] = getw(lib, "locc_w", (lib.nh_numel_locc_w() // 8, 8))
    inv1_w = w["inv1_w"]  = getw(lib, "inv1_w", (IH, D))
    inv1_b = w["inv1_b"]  = getw(lib, "inv1_b", (IH,))
    inv1s_w = w["inv1s_w"] = getw(lib, "inv1s_w", (IH, 24))
    invt_w = w["invt_w"] = getw(lib, "invt_w", (IH, D))
    if not V5:
        inv2_w = w["inv2_w"]  = getw(lib, "inv2_w", (IP, IH))
        inv2_b = w["inv2_b"]  = getw(lib, "inv2_b", (IP,))
    elif not MIN:
        isum_w = w["isum_w"] = getw(lib, "isum_w", (64, IH))
        isum_b = w["isum_b"] = getw(lib, "isum_b", (64,))
    bl_w   = w["bl_w"]    = getw(lib, "bl_w", (BH, lib.nh_bl_feat()))
    bl_b   = w["bl_b"]    = getw(lib, "bl_b", (BH,))
    proj_w = w["proj_w"]  = getw(lib, "proj_w", (H, lib.nh_concat()))
    proj_b = w["proj_b"]  = getw(lib, "proj_b", (H,))
    msg_w  = w["msg_w"]   = getw(lib, "msg_w", (lib.nh_numel_msg_w() // MSGH, MSGH))
    spk_w  = w["spk_w"]   = getw(lib, "spk_w", (SK, SI))
    if not SPELL2:
        spk2_w = w["spk2_w"]  = getw(lib, "spk2_w", (SK, SK))
        spk2_b = w["spk2_b"]  = getw(lib, "spk2_b", (SK,))
    else:
        spm1_w = w["spm1_w"] = getw(lib, "spm1_w", (64, 16))
        spm1_b = w["spm1_b"] = getw(lib, "spm1_b", (64,))
        spm2_w = w["spm2_w"] = getw(lib, "spm2_w", (64, 64))
        spm2_b = w["spm2_b"] = getw(lib, "spm2_b", (64,))

    # local: crop glyph ids with pad off-map
    hx, hy = bl_vals[:, 0], bl_vals[:, 1]
    crop_idx = np.full((B, CROP * CROP), PAD, dtype=np.int64)
    for b in range(B):
        for p in range(CROP * CROP):
            r, c = hy[b] - 4 + p // CROP, hx[b] - 4 + p % CROP
            if 0 <= r < ROWS and 0 <= c < COLS:
                crop_idx[b, p] = glyphs[b, r * COLS + c]
    if V3:  # v3.2 local: terrain field (81x8) + adjacency ring (9x32) + underfoot item
        emon_res = w["emon_w"] = getw(lib, "emon_w", (384, D))
        eitem_res = w["eitem_w"] = getw(lib, "eitem_w", (840, D))
        eterrc = w["eterrc_w"] = getw(lib, "eterrc_w", (128, 8))
        # factorized eff tables (champion scheme, pad row hard zero)
        mon_g = np.arange(384) - 1; mon_g[0] = 0
        item_g = np.zeros(840, dtype=np.int64)
        item_g[1:454] = 1906 + np.arange(453); item_g[454:835] = 1144 + np.arange(381)
        def eff(res, gmap):
            e = res + K_w[torch.tensor(kind_map)[torch.tensor(gmap)]] \
                    + S_w[torch.tensor(sub_map)[torch.tensor(gmap)]]
            return torch.cat([torch.zeros(1, D, dtype=torch.float64), e[1:]], dim=0)
        emon = eff(emon_res, mon_g)
        eitem = eff(eitem_res, item_g)
        w["_emon_eff"] = emon; w["_eitem_eff"] = eitem
        ct = np.full((B, CROP * CROP), 127, dtype=np.int64)
        cm = np.zeros((B, 9), dtype=np.int64)
        ci = np.zeros((B,), dtype=np.int64)
        for b in range(B):
            for p in range(CROP * CROP):
                r, c = hy[b] - 4 + p // CROP, hx[b] - 4 + p % CROP
                if 0 <= r < ROWS and 0 <= c < COLS:
                    cell = r * COLS + c
                    ct[b, p] = make_obs.terr[b, cell]
                    rr, rc = p // CROP - 4, p % CROP - 4
                    if abs(rr) <= 1 and abs(rc) <= 1:
                        g = glyphs[b, cell]
                        m = 0
                        if 0 <= g < 381: m = g + 1
                        elif 381 <= g < 762: m = g - 381 + 1
                        elif 762 <= g < 1144: m = (g - 762) % 381 + 1
                        cm[b, (rr + 1) * 3 + (rc + 1)] = m
                        if rr == 0 and rc == 0:
                            ci[b] = make_obs.objm[b, cell]
        ring = torch.where(torch.tensor(cm > 0)[:, :, None], emon[torch.tensor(cm)],
                           torch.zeros(1, dtype=torch.float64))
        uf = torch.where(torch.tensor(ci > 0)[:, None], eitem[torch.tensor(ci)],
                         torch.zeros(1, dtype=torch.float64))
        x_local = torch.cat([eterrc[torch.tensor(ct)].reshape(B, -1),
                             ring.reshape(B, -1), uf], dim=1)
    elif V5:  # class crop: LUT glyph -> 9 classes, center forced to class 7
        lib.nh_get_locc_lut.argtypes = [VP]
        loclut = np.empty(5977, dtype=np.uint8)
        lib.nh_get_locc_lut(loclut.ctypes.data_as(VP))
        cls = loclut[crop_idx].astype(np.int64)
        cls[:, (CROP * CROP) // 2] = 7
        x_local = locc[torch.tensor(cls)].reshape(B, -1)
    else:
        x_local = E[torch.tensor(crop_idx)].reshape(B, -1)
    loc = torch.relu(x_local @ loc_w.T + loc_b)
    if LOC2:
        w["loc2_w"] = getw(lib, "loc2_w", (lib.nh_numel_loc2_b(), LH))
        w["loc2_b"] = getw(lib, "loc2_b", (lib.nh_numel_loc2_b(),))
        loc = torch.relu(loc @ w["loc2_w"].T + w["loc2_b"])
    if V5:
        # terrain branch: featurize in numpy (mirrors nh_terr_feat_kernel —
        # integer-exact octants), parity-check against the kernel's terr_tf,
        # then the 592 -> 256 -> 128 MLP into the glb slot
        lib.nh_get_terrc_lut.argtypes = [VP]
        lib.nh_get_terr_tf.argtypes = [VP, ctypes.c_int]
        tlut = np.empty(5977, dtype=np.uint8)
        lib.nh_get_terrc_lut(tlut.ctypes.data_as(VP))
        tf = np.zeros((B, 592))
        for b in range(B):
            lmd = np.full(12, 1 << 30); lmt = np.zeros(48); sec = np.zeros((8, 4, 17))
            hcell = hy[b] * COLS + hx[b]
            for cell in range(ROWS * COLS):
                tc = 13 if cell == hcell else int(tlut[glyphs[b, cell]])
                if tc == 255:
                    continue
                dyc, dxc = cell // COLS - hy[b], cell % COLS - hx[b]
                adx, ady = abs(dxc), abs(dyc); cheb = max(adx, ady)
                if tc < 12 and cheb < lmd[tc]:
                    lmd[tc] = cheb
                    lmt[tc*4:tc*4+4] = [1.0, dxc / 78.0, dyc / 20.0, min(cheb, 30) / 30.0]
                a = np.float32(np.arctan2(np.float32(dyc), np.float32(dxc))) + np.float32(3.14159265358979)
                s = int(a / np.float32(0.78539816339745)) & 7
                band = 0 if cheb < 3 else 1 if cheb < 7 else 2 if cheb < 15 else 3
                sec[s, band, tc] += 1.0
            tf[b] = np.concatenate([lmt, np.log1p(sec.reshape(-1)) / np.log(1660.0)])
        tf_cuda = np.empty(B * 592, dtype=np.float32)
        lib.nh_get_terr_tf(tf_cuda.ctypes.data_as(VP), B)
        diff = np.abs(tf - tf_cuda.reshape(B, 592).astype(np.float64))
        lm_err = diff[:, :48].max()  # landmark block: exact
        sec_bad = (diff[:, 48:] > 2e-4).mean()  # sectors: atan2f boundary ULPs may
        # flip exact-diagonal cells between adjacent sectors — tolerate rare flips
        ok_par = lm_err < 2e-4 and sec_bad < 0.02
        print(f"  [{'OK ' if ok_par else 'FAIL'}] terr featurize parity lm_max={lm_err:.2e} sector_flip_frac={sec_bad:.4f}")
        assert ok_par, "terrain featurize parity"
        terr_h = torch.relu(torch.tensor(tf) @ t1_w.T + t1_b)
        glb = torch.relu(terr_h @ t2_w.T + t2_b)
        dxy = None
    else:
        # global: per-patch flatten -> 16 -> 128, max over tokens
        pat_idx = np.full((B, PX * PY, PW * PH), PAD, dtype=np.int64)
        dxy = np.zeros((B, PX * PY, 2), dtype=np.float64)
        for tk in range(PX * PY):
            r0, c0 = (tk // PX) * PH, (tk % PX) * PW
            dxy[:, tk, 0] = (c0 + 0.5 * (PW - 1) - hx) / COLS
            dxy[:, tk, 1] = (r0 + 0.5 * (PH - 1) - hy) / ROWS
            for pos in range(PW * PH):
                r, c = r0 + pos // PW, c0 + pos % PW
                if r < ROWS and c < COLS:
                    pat_idx[:, tk, pos] = glyphs[:, r * COLS + c]
        xp = E[torch.tensor(pat_idx)].reshape(B, PX * PY, -1)
        if GLBSG:  # global path reads E without gradient
            xp = xp.detach()
        elif GSCALE != 1.0:  # global path's gradient into E scaled
            xp = GSCALE * xp + (1.0 - GSCALE) * xp.detach()
        t16 = torch.relu(xp @ g1_w.T + torch.tensor(dxy) @ g1_xy.T + g1_b)
        if SPLITR:  # old global silenced: slice zeroed, no grad through it
            t16 = t16.detach()
        _G['t16'] = t16; _G['dxy'] = dxy
        t128 = t16 @ g2_w.T
        glb = torch.relu(t128.max(dim=1).values + g2_b)
        if SPLITR:
            glb = torch.zeros_like(glb).detach()
    # blstats features
    f = np.zeros((B, lib.nh_bl_feat()), dtype=np.float64)
    j = 0
    for i in range(27):
        if i in (21, 25):
            continue
        v = bl_vals[:, i].astype(np.float64)
        f[:, j] = np.log1p(np.maximum(v, 0)) * BL_SCALE[i] if BL_ISLOG[i] else v * BL_SCALE[i]
        j += 1
    hunger = np.clip(bl_vals[:, 21], 0, 6)
    for h in range(7):
        f[:, j] = (hunger == h); j += 1
    for k in range(13):
        f[:, j] = (bl_vals[:, 25].astype(np.uint32) >> k) & 1; j += 1
    for h in range(lib.nh_num_actions()):
        f[:, j] = (ex_vals[:, 1] == h); j += 1
    for k in range(18):
        f[:, j] = ex_vals[:, 2 + k] * 0.125; j += 1
    # hp_frac (hp/hpmax), ene_frac (ene/enemax), clamped to [0,1]
    hp = bl_vals[:, 10].astype(np.float64); hpmax = bl_vals[:, 11].astype(np.float64)
    ene = bl_vals[:, 14].astype(np.float64); enemax = bl_vals[:, 15].astype(np.float64)
    f[:, j] = np.clip(hp / np.maximum(hpmax, 1), 0, 1); j += 1
    f[:, j] = np.clip(ene / np.maximum(enemax, 1), 0, 1); j += 1
    # dnum one-hot (nominal dungeon branch; scalar scale zeroed)
    dnum = np.clip(bl_vals[:, 23], 0, 7)
    for h in range(8):
        f[:, j] = (dnum == h); j += 1
    # underfoot engraving bits: any-engraving, active-Elbereth
    f[:, j] = (ex_vals[:, 0] >= 1); j += 1
    f[:, j] = (ex_vals[:, 0] >= 2); j += 1
    # shop: standing on goods, and gold/price capped at 1
    f[:, j] = ex_vals[:, 20]; j += 1
    f[:, j] = ex_vals[:, 21] * 0.01; j += 1
    # spell scalar, mirrors NH_F_SPELL: known count/8 only (per-slot content
    # rides the spell-key path)
    f[:, j] = ex_vals[:, 22] * 0.125; j += 1
    # encumbrance pair, mirrors NH_F_WEIGHT: softsign(ratio-1), cap/1000
    d = ex_vals[:, 55] * 0.01 - 1.0
    f[:, j] = d / (1.0 + np.abs(d)); j += 1
    f[:, j] = ex_vals[:, 56] * 0.001; j += 1
    if not IDEMB:  # identity one-hots (dead features under NH_ID_EMBED)
        f[:, j:j + 20] = ex_vals[:, 57:77]
    j += 20
    if INTRINS:
        for k in range(8):
            f[:, j] = (ex_vals[:, 77].astype(np.int64) >> k) & 1; j += 1
    if THREAT:
        tx = 77 + (1 if INTRINS else 0)
        dist = ex_vals[:, tx].astype(np.int64)
        near = dist < 15
        f[:, j] = near; j += 1
        f[:, j] = np.where(near, 1.0 - dist * np.float64(np.float32(1.0 / 15.0)), 0.0); j += 1
        for k in range(8):
            f[:, j] = (ex_vals[:, tx + 1].astype(np.int64) == k); j += 1
        f[:, j] = ex_vals[:, tx + 2] * 0.125; j += 1
        f[:, j] = ex_vals[:, tx + 3] * 0.125; j += 1
        f[:, j] = ex_vals[:, tx + 4] * 0.0625; j += 1
        f[:, j] = ex_vals[:, tx + 5] * np.float64(np.float32(0.04)); j += 1
        f[:, j] = np.where(near, (ex_vals[:, tx + 5].astype(np.float64)
                                  - bl_vals[:, 18]) * np.float64(np.float32(0.1)), 0.0); j += 1
        f[:, j] = ex_vals[:, tx + 6] * np.float64(np.float32(1.0 / 24.0)); j += 1
    f = np.clip(f, -1.0, 1.0)   # strict clamp, mirrors the kernel
    fb = torch.tensor(f)
    blh = torch.relu(fb @ bl_w.T + bl_b)
    # inventory entities: per-slot embed + gated state features -> 32, relu
    # (the decoder's keys), then pooled 32 -> 128 with max over slots
    sf = np.zeros(st_vals.shape[:2] + (24,), dtype=np.float64)
    for c in range(4):
        sf[:, :, c] = (st_vals[:, :, 0] == c)
    sk = st_vals[:, :, 1] != -128
    sf[:, :, 4] = sk
    if V5:  # strict [-1,1]: spe clamp/7, quantity linear capped at 30
        sf[:, :, 5] = np.where(sk, np.clip(st_vals[:, :, 1], -7, 7) / 7.0, 0.0)
        sf[:, :, 6] = np.minimum(np.maximum(st_vals[:, :, 2], 0), 30) / 30.0
    else:
        sf[:, :, 5] = np.where(sk, st_vals[:, :, 1] * np.float64(np.float32(0.1)), 0.0)
        sf[:, :, 6] = np.log1p(np.maximum(st_vals[:, :, 2], 0)) * 0.5
    sf[:, :, 7] = st_vals[:, :, 3] / 3.0
    sf[:, :, 8] = st_vals[:, :, 4] / 3.0
    for c in range(7):
        sf[:, :, 9 + c] = (st_vals[:, :, 5] >> c) & 1
    sf[:, :, 16] = st_vals[:, :, 6]
    # armor slot category one-hot from the slot glyph (baked otyp->ARM_* table)
    import re as _re
    _src = open("ocean/nethack/netlib.h").read()
    _body = _re.search(r"nh_obj_armcat\[NH_NUM_OBJECTS\] = \{(.*?)\};", _src, _re.S).group(1)
    _tbl = np.array([int(x) for x in _re.findall(r"-?\d+", _body)], dtype=np.int64)
    ot = inv_vals.astype(np.int64) - 1906
    cat = np.where((ot >= 0) & (ot < len(_tbl)), _tbl[np.clip(ot, 0, len(_tbl) - 1)], -1)
    for c in range(7):
        sf[:, :, 17 + c] = (cat == c)
    def _itemrow(v):
        v = v.astype(np.int64)
        row = np.zeros_like(v)
        o = (v >= 1906) & (v < 2359); row[o] = v[o] - 1906 + 1
        bdy = (v >= 1144) & (v < 1525); row[bdy] = v[bdy] - 1144 + 454
        return row
    if V3:
        xi = w["_eitem_eff"][torch.tensor(_itemrow(inv_vals))]
        xt = w["_eitem_eff"][torch.tensor(_itemrow(itr_vals))]
        kt = torch.tensor((_itemrow(itr_vals) != 0).astype(np.float64))[:, :, None]
    else:
        xi = E[torch.tensor(inv_vals.astype(np.int64))]
        # discovered-type channel: pad (5976) slots contribute hard zero
        xt = E[torch.tensor(itr_vals.astype(np.int64))]
        kt = torch.tensor((itr_vals != 5976).astype(np.float64))[:, :, None]
    invh = torch.relu(xi @ inv1_w.T + kt * (xt @ invt_w.T)
                      + torch.tensor(sf) @ inv1s_w.T + inv1_b)  # (B,55,16)
    if not V5:
        invp = torch.relu((invh @ inv2_w.T).max(dim=1).values + inv2_b) # (B,128)
    # trigram message bag: hash char-trigrams (matching nh_msg_hash), sum the
    # embed rows, scale by 1/sqrt(count+1). Concatenated raw (no relu).
    def _lc(c): return c + 32 if 65 <= c <= 90 else c
    rows = []
    for b in range(B):
        row = msg[b]
        ids = []
        for t in range(len(row) - 2):
            c0, c1, c2 = int(row[t]), int(row[t + 1]), int(row[t + 2])
            if c0 == 0 or c1 == 0 or c2 == 0:
                break
            key = (_lc(c0) << 16) | (_lc(c1) << 8) | _lc(c2)
            ids.append(((key * 2654435761) & 0xFFFFFFFF) >> (32 - (msg_w.shape[0].bit_length() - 1)))
        cnt = len(ids)
        s = msg_w[torch.tensor(ids, dtype=torch.long)].sum(dim=0) if cnt else torch.zeros(MSGH, dtype=torch.float64)
        rows.append(s / np.sqrt(cnt + 1))
    msg_sum = torch.stack(rows, dim=0)   # (B, 32); grad flows to msg_w
    # spell-key path: per slot, key = spk_w . [e_eff(book glyph) | known,
    # lev/7, fail/100, know/20000]; sum-pool feeds the trunk
    spk_w = w["spk_w"]
    spkeys, spxs, spocc = [], [], []
    for s in range(8):
        c = 23 + 4 * s
        sid = torch.tensor(ex_vals[:, c].astype(np.int64))
        if V3:
            sg = torch.clamp(sid + 1, max=839)
            emb = torch.where((sid > 0)[:, None], w["_eitem_eff"][sg], torch.zeros_like(w["_eitem_eff"][sg]))
        else:
            sg = torch.clamp(sid + 1906, max=5975)
            emb = torch.where((sid > 0)[:, None], E[sg], torch.zeros_like(E[sg]))
        sc = torch.stack([
            (sid > 0).double(),
            torch.clamp(torch.tensor(ex_vals[:, c + 1]) * 0.142857, max=1.0),
            torch.clamp(torch.tensor(ex_vals[:, c + 2]) * 0.01, max=1.0),
            torch.clamp(torch.tensor(ex_vals[:, c + 3]) * 0.00005, max=1.0),
        ], dim=1)
        xs = torch.cat([emb, sc], dim=1)                       # (B,36)
        spxs.append(xs); spocc.append((sid > 0).double())
        spkeys.append(torch.relu(xs @ spk_w.T))  # (B,16)
    sk = torch.stack(spkeys, dim=1)                            # (B,8,16)
    if SPELL2:
        occ = torch.stack(spocc, dim=1)                        # (B,8)
        sph1 = entact(sk @ spm1_w.T + spm1_b)
        spv = act_last(sph1 @ spm2_w.T + spm2_b)  # (B,8,64)
        ssum = 0.25 * (spv * occ[:, :, None]).sum(dim=1)       # (B,64)
        smax = pool_max(spv, occ[:, :, None], 1)
        # doorstep scalars (exact; empty book -> [1,0,0,1])
        sidm = ex_vals[:, 23:23+32:4].astype(np.int64)
        levm = ex_vals[:, 24:24+32:4].astype(np.int64)
        failm = ex_vals[:, 25:25+32:4].astype(np.int64)
        knowm = ex_vals[:, 26:26+32:4].astype(np.int64)
        occn = sidm > 0
        mf = np.where(occn.any(1), np.min(np.where(occn, failm, 999), 1), 100)
        ml = np.where(occn.any(1), np.max(np.where(occn, levm, 0), 1), 0)
        nn = occn.sum(1)
        mr = np.where(occn.any(1), np.min(np.where(occn, knowm, 99999), 1), 20000)
        eng = np.stack([np.clip(mf * 0.01, 0, 1), np.clip(ml / 7.0, 0, 1),
                        np.clip(np.minimum(nn, 8) * 0.125, 0, 1),
                        np.clip(mr * 0.00005, 0, 1)], 1)
        spool = torch.cat([ssum, smax, torch.tensor(eng, dtype=torch.float64)], dim=1)  # (B,132)
    else:
        spool = torch.relu((sk @ spk2_w.T).max(dim=1).values + spk2_b)
    parts = [loc, glb] + ([] if V5 else [invp]) + [blh, fb, msg_sum, spool]
    if IDEMB:
        # identity embeddings: direct table rows, indices per the kernel
        for nm, rows, dims in [("ide_role_w", 13, 16), ("ide_race_w", 5, 8),
                               ("ide_gend_w", 2, 8), ("ide_algn_w", 3, 8)]:
            w[nm] = getw(lib, nm, (rows, dims))
        role = ex_vals[:, 57:70].argmax(axis=1)
        race = ex_vals[:, 70:75].argmax(axis=1)
        gend = ex_vals[:, 75:77].argmax(axis=1)
        al = np.clip(1 - bl_vals[:, 26], 0, 2)
        parts += [w["ide_role_w"][torch.tensor(role)],
                  w["ide_race_w"][torch.tensor(race)],
                  w["ide_gend_w"][torch.tensor(gend)],
                  w["ide_algn_w"][torch.tensor(al)]]
    if SPLIT:
        et_w = w["eterr_w"] = getw(lib, "eterr_w", (128, 32))
        tg1_w = w["tglb1_w"] = getw(lib, "tglb1_w", (16, PW * PH * 32))
        tg1_xy = w["tglb1_xy"] = getw(lib, "tglb1_xy", (16, 2))
        tg1_b = w["tglb1_b"] = getw(lib, "tglb1_b", (16,))
        tg2_w = w["tglb2_w"] = getw(lib, "tglb2_w", (128, 16))
        tg2_b = w["tglb2_b"] = getw(lib, "tglb2_b", (128,))
        tarr = make_obs.terr
        tpat = np.full((B, PX * PY, PW * PH), 127, dtype=np.int64)
        for tk in range(PX * PY):
            r0, c0 = (tk // PX) * PH, (tk % PX) * PW
            for pos in range(PW * PH):
                r, c = r0 + pos // PW, c0 + pos % PW
                if r < ROWS and c < COLS:
                    tpat[:, tk, pos] = tarr[:, r * COLS + c]
        txp = et_w[torch.tensor(tpat)].reshape(B, PX * PY, -1)
        tt16 = torch.relu(txp @ tg1_w.T + torch.tensor(dxy) @ tg1_xy.T + tg1_b)
        tglb = torch.relu((tt16 @ tg2_w.T).max(dim=1).values + tg2_b)
    if GMEAN:  # token-mean of the relu'd patch reps, after the identity tail
        parts.append(t16.mean(dim=1))
    if ATTNPOOL:  # softmax over tokens of beta*(qa.t16), pooled [t16 | dxy]
        w["apq_w"] = getw(lib, "apq_w", (P1,)); w["apb"] = getw(lib, "apb", (8,))
        att = torch.softmax(w["apb"][0] * (t16 @ w["apq_w"]), dim=1)          # (B,80)
        vtok = torch.cat([t16, torch.tensor(dxy)], dim=2)                       # (B,80,18)
        parts.append(torch.einsum('bt,btk->bk', att, vtok))
    if TOPK:  # K most salient tokens (sigmoid gate), gated rep + (dx,dy); stable top-k by index
        w["sal_w"] = getw(lib, "sal_w", (P1,)); w["sal_b"] = getw(lib, "sal_b", (8,))  # sal_w stored (1,P1)
        sc = torch.sigmoid(t16 @ w["sal_w"] + w["sal_b"][0])           # (B,80)
        K = 8; rows = []
        for b in range(B):
            order = sorted(range(t16.shape[1]), key=lambda tk: (-float(sc[b, tk]), tk))[:K]
            rows.append(torch.cat([torch.cat([sc[b, tk] * t16[b, tk], torch.tensor(dxy[b, tk])]) for tk in order]))
        parts.append(torch.stack(rows, dim=0))
    if V3:
        parts.append(tglb)
        for nm, lst, ismon in [("mon", make_obs.vmon, True)]:
            tbl = w["_emon_eff"]
            w1 = w[nm + "1_w"] = getw(lib, nm + "1_w", (32, 40))
            b1v = w[nm + "1_b"] = getw(lib, nm + "1_b", (32,))
            qv = w[nm + "q_w"] = getw(lib, nm + "q_w", (32,))
            bv = w[nm + ("b" if ismon else "b")] = getw(lib, nm + "b", (8,))
            rows = []
            for b in range(B):
                toks, valid = [], []
                cnt, ndist, ndx, ndy = 0, 1.0, 0.0, 0.0
                for k in range(16):
                    row = int(lst[b, k, 0]) | (int(lst[b, k, 1]) << 8)
                    dx = int(lst[b, k, 2]); dx = dx - 256 if dx >= 128 else dx
                    dy = int(lst[b, k, 3]); dy = dy - 256 if dy >= 128 else dy
                    f4, f5, f6 = int(lst[b, k, 4]), int(lst[b, k, 5]), int(lst[b, k, 6])
                    if row > 0:
                        if ismon:
                            tail8 = [dx / 40.0, dy / 11.0, 1.0 if f4 & 1 else 0.0,
                                     1.0 if f4 & 4 else 0.0, min(f5 * np.float64(np.float32(0.04)), 1.0),
                                     min(f6 * np.float64(np.float32(1.0 / 24.0)), 1.0),
                                     1.0 if abs(dx) <= 1 and abs(dy) <= 1 else 0.0,
                                     1.0 if f4 & 8 else 0.0]
                        else:
                            tail8 = [dx / 40.0, dy / 11.0, 1.0 if f5 & 1 else 0.0,
                                     1.0 if f5 & 2 else 0.0, 0.0, 0.0, 0.0, 0.0]
                        tk = torch.cat([tbl[row], torch.tensor(tail8, dtype=torch.float64)])
                        valid.append(True)
                        hit = (f4 & 1) if ismon else 1
                        if hit:
                            cnt += 1
                            dd = max(abs(dx / 40.0) * 40.0, abs(dy / 11.0) * 11.0) * 0.125
                            if dd < ndist:
                                ndist = dd; ndx = dx / 40.0; ndy = dy / 11.0
                    else:
                        tk = torch.zeros(40, dtype=torch.float64); valid.append(False)
                    toks.append(tk)
                tokm = torch.stack(toks)
                h = torch.relu(tokm @ w1.T + b1v)
                if ENTMAX:  # channel-wise max; (dx,dy) = nearest counted entity, as data
                    if any(valid):
                        hm = h + torch.tensor([0.0 if v else -1e30 for v in valid],
                                              dtype=torch.float64).unsqueeze(1)
                        pooled = hm.max(dim=0).values
                    else:
                        pooled = torch.zeros(32, dtype=torch.float64)
                    rows.append(torch.cat([pooled, torch.tensor([ndx, ndy], dtype=torch.float64),
                                           torch.tensor([min(cnt * 0.125, 1.0), min(ndist, 1.0)], dtype=torch.float64)]))
                else:
                    sc = bv[0] * (h @ qv)
                    mask = torch.tensor([0.0 if v else -1e30 for v in valid], dtype=torch.float64)
                    att = torch.softmax(sc + mask, dim=0) if any(valid) else torch.zeros(16, dtype=torch.float64)
                    pooled = att @ h
                    pdx = (att * tokm[:, 32]).sum(); pdy = (att * tokm[:, 33]).sum()
                    rows.append(torch.cat([pooled, pdx.reshape(1), pdy.reshape(1),
                                           torch.tensor([min(cnt * 0.125, 1.0), min(ndist, 1.0)], dtype=torch.float64)]))
            parts.append(torch.stack(rows))
    elif SPLIT:
        parts.append(tglb)
        e1_w = w["ent1_w"] = getw(lib, "ent1_w", (32, 40))
        e1_b = w["ent1_b"] = getw(lib, "ent1_b", (32,))
        eq_w = w["entq_w"] = getw(lib, "entq_w", (32,))
        eb = w["entb"] = getw(lib, "entb", (8,))
        ents = make_obs.ents
        rows = []
        for b in range(B):
            toks, valid, hostd = [], [], []
            nhost, ndist, ndx, ndy = 0, 1.0, 0.0, 0.0
            for k in range(32):
                g = int(ents[b, k, 0]) | (int(ents[b, k, 1]) << 8)
                dx = int(ents[b, k, 2]); dx = dx - 256 if dx >= 128 else dx
                dy = int(ents[b, k, 3]); dy = dy - 256 if dy >= 128 else dy
                fl = int(ents[b, k, 4])
                if g > 0:
                    t = torch.cat([E[min(g, 5976)], torch.tensor([dx / 40.0, dy / 11.0,
                        1.0 if fl & 1 else 0.0, 1.0 if fl & 2 else 0.0, 1.0 if fl & 4 else 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)])
                    valid.append(True)
                    if (fl & 1) and not (fl & 2):
                        nhost += 1
                        dd = max(abs(dx), abs(dy)) * 0.125
                        if dd < ndist:
                            ndist = dd; ndx = dx / 40.0; ndy = dy / 11.0
                else:
                    t = torch.zeros(40, dtype=torch.float64); valid.append(False)
                toks.append(t)
            tokm = torch.stack(toks)                        # (32, 40)
            h = torch.relu(tokm @ e1_w.T + e1_b)            # (32, 32)
            if ENTMAX:  # channel-wise max over valid tokens; (dx,dy) = nearest hostile, as data
                if any(valid):
                    hm = h + torch.tensor([0.0 if v else -1e30 for v in valid],
                                          dtype=torch.float64).unsqueeze(1)
                    pooled = hm.max(dim=0).values
                else:
                    pooled = torch.zeros(32, dtype=torch.float64)
                tail = torch.cat([pooled, torch.tensor([ndx, ndy], dtype=torch.float64),
                                  torch.tensor([min(nhost * 0.125, 1.0), min(ndist, 1.0)], dtype=torch.float64)])
            else:
                sc = eb[0] * (h @ eq_w)
                mask = torch.tensor([0.0 if v else -1e30 for v in valid], dtype=torch.float64)
                att = torch.softmax(sc + mask, dim=0) if any(valid) else torch.zeros(32, dtype=torch.float64)
                pooled = att @ h
                pdx = (att * tokm[:, 32]).sum(); pdy = (att * tokm[:, 33]).sum()
                tail = torch.cat([pooled, pdx.reshape(1), pdy.reshape(1),
                                  torch.tensor([min(nhost * 0.125, 1.0), min(ndist, 1.0)], dtype=torch.float64)])
            rows.append(tail)
        parts.append(torch.stack(rows))
    if GLBPOS:  # argmax patch (dx,dy) per global channel; data selection, no grad
        idxs = (t16 @ g2_w.T).argmax(dim=1)                       # (B,128)
        dxyt = torch.tensor(dxy)                                  # (B,80,2)
        parts.append(dxyt[torch.arange(idxs.shape[0])[:, None], idxs].reshape(idxs.shape[0], -1))
    if MIN:  # V6-min: per-slot deep MLP -> masked sum|max + pass-throughs; no attention
        mv1 = w["mv1_w"] = getw(lib, "mv1_w", (64, IH))
        mv1b = w["mv1_b"] = getw(lib, "mv1_b", (64,))
        mv2 = w["mv2_w"] = getw(lib, "mv2_w", (64, 64))
        mv2b = w["mv2_b"] = getw(lib, "mv2_b", (64,))
        occb = torch.tensor((inv_vals.astype(np.int64) != 5976).astype(np.float64))
        h1 = entact(invh @ mv1.T + mv1b)
        v = act_last(h1 @ mv2.T + mv2b)   # (B,55,64)
        sm = 0.2 * (v * occb[:, :, None]).sum(dim=1)
        mx = pool_max(v, occb[:, :, None], 1)
        wldb = torch.tensor(sf[:, :, 10]) * occb
        qvb = torch.tensor(sf[:, :, 12]) * occb
        wrnb = torch.tensor(sf[:, :, 9]) * occb
        sft = torch.tensor(sf)
        pw = torch.cat([(invh * wldb[:, :, None]).sum(1), (sft * wldb[:, :, None]).sum(1)], 1)
        pq = torch.cat([(invh * qvb[:, :, None]).sum(1), (sft * qvb[:, :, None]).sum(1)], 1)
        nw = wrnb.sum(1).clamp(min=1.0)
        pworn = (invh * wrnb[:, :, None]).sum(1) / nw[:, None]
        if APANEL:  # 4 single-occupant accessory slots [amulet | ring A | ring B | eyewear], first owner wins
            ot = inv_vals.astype(np.int64) - 1906
            worn = (sf[:, :, 9] > 0.5) & (occb.numpy() > 0.5)
            isam = worn & (ot >= 178) & (ot <= 188); isey = worn & (ot >= 207) & (ot <= 209)
            isr = worn & (ot >= 150) & (ot <= 177); rr = np.cumsum(isr, 1) - 1
            masks = [isam & (np.cumsum(isam, 1) == 1), isr & (rr == 0), isr & (rr == 1), isey & (np.cumsum(isey, 1) == 1)]
            if os.environ.get("NH_TEST_ACC_CANARY") == "1": masks[1], masks[2] = masks[2], masks[1]  # must FAIL if exercised
            pacc = torch.cat([torch.cat([(invh * m[:, :, None]).sum(1), (sft * m[:, :, None]).sum(1)], 1)
                              for m in [torch.tensor(mm.astype(np.float64)) for mm in masks]], 1)
            parts.append(torch.cat([sm, mx, pw, pq, pworn, pacc], 1))
        else:
            parts.append(torch.cat([sm, mx, pw, pq, pworn], 1))
    elif GEN:  # final arch: LN + 1/sqrt(d) unit (attn slices|cnt|sum|max) + pass-throughs
        gln_g = w["gln_g"] = getw(lib, "gln_g", (IH,))
        gln_b = w["gln_b"] = getw(lib, "gln_b", (IH,))
        gnv_w = w["gnv_w"] = getw(lib, "gnv_w", (64, IH))
        gnv_b = w["gnv_b"] = getw(lib, "gnv_b", (64,))
        gns_w = w["gns_w"] = getw(lib, "gns_w", (8, IH))
        occb = torch.tensor((inv_vals.astype(np.int64) != 5976).astype(np.float64))
        mu = invh.mean(dim=2, keepdim=True)
        var = invh.var(dim=2, unbiased=False, keepdim=True)
        lnr = (invh - mu) / torch.sqrt(var + 1e-5) * gln_g + gln_b
        S = (lnr @ gns_w.T)                                       # (B,55,8)
        maskb = torch.where(occb > 0, 0.0, float("-inf"))[:, :, None]
        A = torch.softmax(0.25 * S + maskb, dim=1)
        A = torch.where(torch.isnan(A), torch.zeros_like(A), A)
        v = torch.relu(invh @ gnv_w.T + gnv_b)                    # (B,55,64)
        Bn = invh.shape[0]
        att = torch.zeros(Bn, 64, dtype=torch.float64)
        cnt = torch.zeros(Bn, 64, dtype=torch.float64)
        for hh in range(8):
            sl = v[:, :, hh*8:(hh+1)*8]
            att[:, hh*8:(hh+1)*8] = torch.einsum('bs,bsd->bd', A[:, :, hh], sl)
            cnt[:, hh*8:(hh+1)*8] = 0.2 * torch.einsum('bs,bsd->bd',
                torch.sigmoid(S[:, :, hh]) * occb, sl)
        vm = v * occb[:, :, None]
        sm = 0.2 * vm.sum(dim=1)
        mx = torch.relu(v.masked_fill(occb[:, :, None] < .5, -1e9).max(dim=1).values)
        wldb = torch.tensor(sf[:, :, 10]) * occb
        qvb = torch.tensor(sf[:, :, 12]) * occb
        wrnb = torch.tensor(sf[:, :, 9]) * occb
        sft = torch.tensor(sf)
        NOPASS = int(os.environ.get("NH_GEN_NOPASS", "0"))
        pw = torch.cat([(invh * wldb[:, :, None]).sum(1), (sft * wldb[:, :, None]).sum(1)], 1)
        if NOPASS & 1: pw = pw * 0
        pq = torch.cat([(invh * qvb[:, :, None]).sum(1), (sft * qvb[:, :, None]).sum(1)], 1)
        if NOPASS & 2: pq = pq * 0
        nw = wrnb.sum(1).clamp(min=1.0)
        pworn = (invh * wrnb[:, :, None]).sum(1) / nw[:, None]
        if NOPASS & 4: pworn = pworn * 0
        parts.append(torch.cat([att, cnt, sm, mx, pw, pq, pworn], 1))
    elif GMP:  # generalized pooling: K heads x (softmax(tau S) att | 0.2 sigmoid(S) count)
        gws = w["gws_w"] = getw(lib, "gws_w", (8, IH + 24))
        gv_w = w["gv_w"] = getw(lib, "gv_w", (16, IH))
        gv_b = w["gv_b"] = getw(lib, "gv_b", (16,))
        gtau = w["gtau"] = getw(lib, "gtau", (8,))
        occb = torch.tensor((inv_vals.astype(np.int64) != 5976).astype(np.float64))
        xs = torch.cat([invh, torch.tensor(sf)], dim=2)           # (B,55,40)
        S = xs @ gws.T                                            # (B,55,8)
        maskb = torch.where(occb > 0, 0.0, float("-inf"))[:, :, None]
        A = torch.softmax(torch.exp(gtau)[None, None, :] * S + maskb, dim=1)
        A = torch.where(torch.isnan(A), torch.zeros_like(A), A)   # empty inv rows
        v = torch.relu(invh @ gv_w.T + gv_b)                      # (B,55,16)
        att = torch.einsum('bsk,bsd->bkd', A, v)                  # (B,8,16)
        sig = torch.einsum('bsk,bsd->bkd', torch.sigmoid(S) * occb[:, :, None], v) * 0.2
        parts.append(torch.cat([att, sig], dim=2).reshape(invh.shape[0], -1))
    elif INVATTN:  # M-query softmax attention over the 55 post-relu slot reps
        M = lib.nh_numel_iaq_w() // IH
        iaq = w["iaq_w"] = getw(lib, "iaq_w", (M, IH))
        att = torch.softmax(invh @ iaq.T, dim=1)                  # (B,55,M)
        parts.append(torch.einsum('bsm,bsk->bmk', att, invh).reshape(invh.shape[0], -1))
    if LAB and LAB_TOK:  # typed streams: 48-dim tokens -> deep values -> 8-head pools
        if V5:
            lib.nh_get_haz_lut.argtypes = [VP]
            hazlut = np.empty(381, dtype=np.uint8)
            lib.nh_get_haz_lut(hazlut.ctypes.data_as(VP))
        for nm, lst, ismon in [("lm", make_obs.labm, True), ("li", make_obs.labi, False)]:
            if MIN:
                rn = "mr" if ismon else "ir"; mn = "mm" if ismon else "im"
                mrw = w[rn + "_w"] = getw(lib, rn + "_w", (16, 48))
                mrb = w[rn + "_b"] = getw(lib, rn + "_b", (16,))
                m1w = w[mn + "1_w"] = getw(lib, mn + "1_w", (64, 16))
                m1b = w[mn + "1_b"] = getw(lib, mn + "1_b", (64,))
                m2w = w[mn + "2_w"] = getw(lib, mn + "2_w", (64, 64))
                m2b = w[mn + "2_b"] = getw(lib, mn + "2_b", (64,))
            else:
                w1 = w[nm + "1_w"] = getw(lib, nm + "1_w", (64, 48))
                b1v = w[nm + "1_b"] = getw(lib, nm + "1_b", (64,))
                w2 = w[nm + "2_w"] = getw(lib, nm + "2_w", (64, 64))
                b2v = w[nm + "2_b"] = getw(lib, nm + "2_b", (64,))
                aw = w[nm + "a_w"] = getw(lib, nm + "a_w", (8, 48))
                abv = w[nm + "a_b"] = getw(lib, nm + "a_b", (8,))
            rows = []
            for b in range(B):
                toks, valid = [], []
                for k in range(KT):
                    row = int(lst[b, k, 0]) | (int(lst[b, k, 1]) << 8)
                    dx = int(lst[b, k, 2]); dx = dx - 256 if dx >= 128 else dx
                    dy = int(lst[b, k, 3]); dy = dy - 256 if dy >= 128 else dy
                    f4, f5, f6 = int(lst[b, k, 4]), int(lst[b, k, 5]), int(lst[b, k, 6])
                    if row > 0:
                        g = row - 1 if ismon else (1906 + row - 1 if row < 454 else 1144 + row - 454)
                        cheb = max(abs(dx), abs(dy))
                        if V5:  # strict [-1,1]: rare geometry tails clamped
                            base = [np.clip(dx / 40.0, -1.0, 1.0), np.clip(dy / 11.0, -1.0, 1.0),
                                    min(cheb, 15) * np.float64(np.float32(1.0 / 15.0)),
                                    k * np.float64(np.float32(1.0 / 15.0))]
                        else:
                            base = [dx / 40.0, dy / 11.0,
                                    min(cheb, 15) * np.float64(np.float32(1.0 / 15.0)),
                                    k * np.float64(np.float32(1.0 / 15.0))]
                        if ismon:
                            fl = [1.0 if f4 & 1 else 0.0, 1.0 if f4 & 8 else 0.0,
                                  1.0 if f4 & 4 else 0.0, 1.0 if cheb <= 1 else 0.0,
                                  min(f5 * np.float64(np.float32(0.04)), 1.0),
                                  min(f6 * np.float64(np.float32(1.0 / 24.0)), 1.0)]
                        else:
                            fl = [1.0 if f5 & 1 else 0.0, 1.0 if f5 & 2 else 0.0,
                                  0.0, 0.0, 0.0, 0.0]
                        if V5 and ismon:  # hazard LUT bits at dims 42-45
                            hb = int(hazlut[(row - 1) % 381])
                            tail6 = [float((hb >> j) & 1) for j in range(4)] + [0.0, 0.0]
                        else:
                            tail6 = [0.0] * 6
                        tk = torch.cat([E[g], torch.tensor(base + fl + tail6,
                                                           dtype=torch.float64)])
                        valid.append(True)
                    else:
                        tk = torch.zeros(48, dtype=torch.float64); valid.append(False)
                    toks.append(tk)
                tokm = torch.stack(toks)                       # (16, 48)
                if MIN:
                    vb = torch.tensor([1.0 if v else 0.0 for v in valid],
                                      dtype=torch.float64)
                    rp = torch.relu(tokm @ mrw.T + mrb)        # (16, 16)
                    hv1 = entact(rp @ m1w.T + m1b)
                    hv = act_last(hv1 @ m2w.T + m2b)
                    sm = 0.25 * (hv * vb[:, None]).sum(0)
                    mx = pool_max(hv, vb[:, None], 0)
                    g0 = rp[0] * vb[0]
                    if ismon:
                        rows.append(torch.cat([sm, mx, g0]))
                    else:
                        uf = (rp * (vb * (tokm[:, 36] > 0.5).double())[:, None]).sum(0)
                        rows.append(torch.cat([sm, mx, g0, uf]))
                    continue
                h2 = torch.relu(torch.relu(tokm @ w1.T + b1v) @ w2.T + b2v)  # (16, 64)
                sc = tokm @ aw.T + abv                         # (16, 8)
                mask = torch.tensor([0.0 if v else -1e30 for v in valid],
                                    dtype=torch.float64)[:, None]
                if any(valid):
                    att = torch.softmax(sc + mask, dim=0)      # (16, 8)
                    pooled = torch.cat([att[:, hh] @ h2[:, hh * 8:(hh + 1) * 8]
                                        for hh in range(8)])
                else:
                    pooled = torch.zeros(64, dtype=torch.float64)
                rows.append(pooled)
            parts.append(torch.stack(rows))
    if V5 and not GMP and not GEN and not MIN:  # inv2 tail: parameterless wield readout + masked-sum channel
        wldg = torch.tensor(sf[:, :, 10])                # wielded state bit
        parts.append((invh * wldg[:, :, None]).sum(dim=1))       # (B,16)
        ih = torch.relu(invh @ isum_w.T + isum_b)                # (B,55,64)
        occ = torch.tensor((inv_vals.astype(np.int64) != 5976).astype(np.float64))
        parts.append(0.2 * (ih * occ[:, :, None]).sum(dim=1))    # (B,64)
    concat = torch.cat(parts, dim=1)
    if LAB and LAB_AUX:  # aux heads: linear preds + targets/masks mirroring the kernel
        aux_w = w["aux_w"] = getw(lib, "aux_w", (32, lib.nh_concat()))
        ap = concat @ aux_w.T                                  # (B, 32)
        ax0 = ex_vals.shape[1] - 26
        av = ex_vals[:, ax0:].astype(np.int64)
        AT = np.zeros((B, 32)); AM = np.zeros((B, 32))
        dsm = (av[:, 0] != 0).astype(np.float64)
        AT[:, 0] = av[:, 1] / 20.0; AM[:, 0] = dsm
        AT[:, 1] = av[:, 2] / 10.0; AM[:, 1] = dsm
        nhm = (av[:, 6] > 0).astype(np.float64)
        AT[:, 2] = av[:, 4] / 20.0; AM[:, 2] = nhm
        AT[:, 3] = av[:, 5] / 10.0; AM[:, 3] = nhm
        AT[:, 4] = av[:, 3] / 15.0; AM[:, 4] = nhm
        AT[:, 5] = np.minimum(av[:, 6], 16) / 16.0; AM[:, 5] = 1
        m1m = (av[:, 12] < 127).astype(np.float64)
        AT[:, 6] = av[:, 10] / 20.0; AM[:, 6] = m1m
        AT[:, 7] = av[:, 11] / 10.0; AM[:, 7] = m1m
        AT[:, 8] = np.minimum(av[:, 12], 15) / 15.0; AM[:, 8] = m1m
        m2m = (av[:, 15] < 127).astype(np.float64)
        AT[:, 9] = av[:, 13] / 20.0; AM[:, 9] = m2m
        AT[:, 10] = av[:, 14] / 10.0; AM[:, 10] = m2m
        AT[:, 11] = np.minimum(av[:, 15], 15) / 15.0; AM[:, 11] = m2m
        AT[:, 12] = np.minimum(av[:, 16], 16) / 16.0; AM[:, 12] = 1
        for qq in range(4):
            AT[:, 13 + qq] = np.minimum(av[:, 17 + qq], 8) * 0.125; AM[:, 13 + qq] = 1
        i1m = (av[:, 23] < 127).astype(np.float64)
        AT[:, 17] = av[:, 21] / 20.0; AM[:, 17] = i1m
        AT[:, 18] = av[:, 22] / 10.0; AM[:, 18] = i1m
        AT[:, 19] = np.minimum(av[:, 23], 15) / 15.0; AM[:, 19] = i1m
        AT[:, 20] = np.minimum(av[:, 24], 16) / 16.0; AM[:, 20] = 1
        AT[:, 21] = np.minimum(av[:, 7], 20) / 20.0; AM[:, 21] = 1
        AT[:, 22] = (av[:, 8] != 0); AM[:, 22] = 1
        AT[:, 23] = (av[:, 9] >= 0); AM[:, 23] = 1
        AT[:, 24] = ((av[:, 25] & 1) != 0); AM[:, 24] = 1
        AT[:, 25] = ((av[:, 25] & 4) != 0); AM[:, 25] = 1
        w["_aux"] = (ap, torch.tensor(AT), torch.tensor(AM))
    out = torch.relu(concat @ proj_w.T + proj_b)
    if FILM:
        ide = concat[:, -40:]  # the identity tail nh_idemb_kernel wrote
        w["film_g_w"] = getw(lib, "film_g_w", (40, H))
        w["film_b_w"] = getw(lib, "film_b_w", (40, H))
        out = out * (1 + ide @ w["film_g_w"]) + ide @ w["film_b_w"]
    return out, invh, w, torch.stack(spkeys, dim=1)


def run(lib):
    B, hidden = 4, 24
    lib.nh_init(B, hidden)
    obs_size = lib.nh_obs_size()
    grid = lib.nh_grid()
    vocab = lib.nh_glyph_vocab()
    print(f"obs_size={obs_size} grid={grid} vocab={vocab} "
          f"bl_feat={lib.nh_bl_feat()} concat={lib.nh_concat()} dec_od={lib.nh_dec_od()}")

    if not V5:
        # glb1_xy zero-inits; randomize it so a broken dx,dy forward term is visible
        wxy = np.random.default_rng(5).standard_normal(lib.nh_numel_glb1_xy()).astype(np.float32)
        lib.nh_set_glb1_xy(wxy.ctypes.data_as(VP))
    if IDEMB:  # same idiom: zero-init tables would hide forward bugs
        for i, nm in enumerate(IDE_NAMES):
            wv = np.random.default_rng(6 + i).standard_normal(
                getattr(lib, f"nh_numel_{nm}")()).astype(np.float32)
            getattr(lib, f"nh_set_{nm}")(wv.ctypes.data_as(VP))
    if FILM:  # zero-init scale/shift tables would make FiLM an invisible no-op
        for i, nm in enumerate(["film_g_w", "film_b_w"]):
            wv = (0.3 * np.random.default_rng(12 + i).standard_normal(
                getattr(lib, f"nh_numel_{nm}")())).astype(np.float32)
            getattr(lib, f"nh_set_{nm}")(wv.ctypes.data_as(VP))
    if LAB and LAB_AUX:  # zero-init aux heads would leave the concat injection untested
        lib.nh_auxh.restype = ctypes.c_int
        lib.nh_aux_coef.restype = ctypes.c_float
        for fn in ["nh_get_auxp", "nh_get_auxt", "nh_get_auxm"]:
            getattr(lib, fn).argtypes = [VP, ctypes.c_int]
        wv = (0.1 * np.random.default_rng(21).standard_normal(
            lib.nh_numel_aux_w())).astype(np.float32)
        lib.nh_set_aux_w(wv.ctypes.data_as(VP))

    max_glyph_used = 40  # keep embedding usage dense & checkable
    obs, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg = make_obs(B, obs_size, grid, max_glyph_used)
    obs_d, _ = h2d(obs)
    out_d = dev(B * hidden * 4)

    # Fixed upstream grad g_out; loss L = sum(out * g_out).
    rng = np.random.default_rng(7)
    g_out = rng.standard_normal((B, hidden)).astype(np.float32)

    def forward_loss():
        lib.nh_forward(out_d, obs_d, B)
        out = d2h(out_d, B * hidden).reshape(B, hidden)
        L = float((out * g_out).sum())
        if LAB and LAB_AUX:  # total loss includes the aux MSE the backward differentiates
            AH = lib.nh_auxh()
            bufs = []
            for fn in ["nh_get_auxp", "nh_get_auxt", "nh_get_auxm"]:
                d = dev(B * AH * 4)
                getattr(lib, fn)(d, B)
                bufs.append(d2h(d, B * AH).reshape(B, AH).astype(np.float64))
            p, t, m = bufs
            L += 0.5 * float(lib.nh_aux_coef()) / B * float((m * (p - t) ** 2).sum())
        return L

    # Analytic grads: forward then backward with dL/dout = g_out.
    L0 = forward_loss()
    grad_d, _ = h2d(g_out)  # backward mutates grad in place
    lib.nh_backward(grad_d, B)

    enc_names = ["ekind_w", "esub_w", "proj_w", "proj_b", "bl_w", "bl_b", "loc_w", "loc_b"] \
        + (["terr1_w", "terr1_b", "terr2_w", "terr2_b", "locc_w"] if V5
           else ["glb1_w", "glb1_xy", "glb1_b", "glb2_w", "glb2_b"]) \
        + ["inv1_w", "inv1_b", "inv1s_w", "invt_w"] \
        + ([] if (GMP or GEN or MIN) else (["isum_w", "isum_b"] if V5 else ["inv2_w", "inv2_b"])) \
        + ["embed_w", "msg_w"]
    if IDEMB:
        enc_names += IDE_NAMES
    if FILM:
        enc_names += ["film_g_w", "film_b_w"]
    if LOC2:
        enc_names += ["loc2_w", "loc2_b"]
    if TOPK:
        enc_names += ["sal_w", "sal_b"]
    if ATTNPOOL:
        enc_names += ["apq_w", "apb"]
    if SPLIT:
        enc_names += ["eterr_w", "tglb2_w", "tglb2_b"]
        if not V3:
            enc_names += ["ent1_w", "ent1_b", "entq_w", "entb"]
    if V3:
        enc_names += ["emon_w", "eitem_w", "eterrc_w", "mon1_w", "mon1_b", "monq_w", "monb"]
    if INVATTN:
        enc_names += (["mv1_w", "mv1_b", "mv2_w", "mv2_b"] if MIN
                      else ["gws_w", "gv_w", "gv_b", "gtau"] if GMP
                      else (["gln_g", "gln_b", "gnv_w", "gnv_b", "gns_w"] if GEN else ["iaq_w"]))
    if LAB:
        enc_names += LAB_NAMES
    enc_names += ["spk_w"] if "spk_w" not in enc_names else []
    if SPELL2:
        enc_names += ["spm1_w", "spm1_b", "spm2_w", "spm2_b"]
    else:
        enc_names += ["spk2_w", "spk2_b"]

    # Central finite differences of L = sum(out*g_out). The encoder ends in a
    # ReLU (and the blstats branch has its own), so a perturbation that flips a
    # unit's sign makes the FD non-smooth and disagree with the (correct)
    # subgradient. We detect such kink crossings via the second difference
    # (|L+ + L- - 2*L0| is O(eps) at a kink vs O(eps^2) on a smooth region)
    # and skip those entries. eps is small so smooth curvature stays negligible.
    eps = 1e-3
    kink_tol = 2e-5      # |Lp+Lm-2*L0| above this ⇒ a ReLU flipped; skip entry.
    rel_tol = 1.5e-2
    rng = np.random.default_rng(123)
    # glb1/inv2/inv1 are FD-unverifiable: their weights feed the max-pool, so
    # perturbations flip near-tied argmax winners under the kink detector's
    # radar and bias the quotient (worse at the 16-dim inv bottleneck, where
    # ties are denser). The exact float64 torch reference covers them.
    # v5 removes both max-pools (patch, inv) — everything is FD-checkable
    fd_skip = set() if V5 else {"glb1_w", "glb1_xy", "glb1_b", "inv2_w", "inv1_w", "inv1_b", "invt_w"}
    all_ok = True
    for name in enc_names:
        if name in fd_skip:
            print(f"  [----] {name:8s} FD skipped (max-pool shared weights); torch-checked below")
            continue
        get = getattr(lib, f"nh_get_{name}")
        setw = getattr(lib, f"nh_set_{name}")
        gradf = getattr(lib, f"nh_grad_{name}")
        n = getattr(lib, f"nh_numel_{name}")()
        w0 = np.empty(n, dtype=np.float32); get(w0.ctypes.data_as(VP))
        ga = np.empty(n, dtype=np.float32); gradf(ga.ctypes.data_as(VP))

        if name == "embed_w":
            used = np.unique(np.concatenate([glyphs.reshape(-1), inv_vals.reshape(-1)]))
            D = lib.nh_embed_dim()
            cand = np.array([g * D + d for g in used for d in range(D)], dtype=np.int64)
        else:
            cand = np.arange(n, dtype=np.int64)
        cand = cand[np.abs(ga[cand]) > 1e-3]        # need signal for a meaningful ratio
        if len(cand) == 0:
            print(f"  [SKIP] {name:8s} n={n:8d} (no entry with |grad|>1e-3)")
            continue
        rng.shuffle(cand)

        max_rel, checked, skipped = 0.0, 0, 0
        for i in cand:
            if checked >= 10:
                break
            i = int(i)
            wp = w0.copy(); wp[i] += eps; setw(wp.ctypes.data_as(VP)); Lp = forward_loss()
            wm = w0.copy(); wm[i] -= eps; setw(wm.ctypes.data_as(VP)); Lm = forward_loss()
            setw(w0.ctypes.data_as(VP))  # restore
            if abs(Lp + Lm - 2 * L0) > kink_tol:     # ReLU kink crossing → FD invalid
                skipped += 1
                continue
            gnum = (Lp - Lm) / (2 * eps)
            rel = abs(gnum - ga[i]) / max(1.0, abs(gnum), abs(ga[i]))
            max_rel = max(max_rel, rel)
            checked += 1
        # padded tensors (e.g. sal_b: 1 live of 8) can't yield 3 checks; require what exists
        ok = checked >= min(3, len(cand)) and checked >= 1 and max_rel < rel_tol
        all_ok = all_ok and ok
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:8s} n={n:8d} "
              f"checked={checked} kink_skipped={skipped} "
              f"max|analytic|={np.abs(ga).max():.4g} max_rel_err={max_rel:.2e}")

    # Exact reference: float64 torch autograd replica of the whole encoder.
    # Finite differences can't cleanly verify glb1/inv2 — shared max-pool
    # weights flip near-tied argmax winners below the kink detector's
    # threshold.
    all_ok = torch_check(lib, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, g_out, enc_names, hidden) and all_ok
    all_ok = dec_check(lib, obs_d, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, B, hidden) and all_ok
    return all_ok


def torch_check(lib, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, g_out, enc_names, H):
    import torch
    out, _, w, _ = torch_encoder(lib, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, H)
    L = (out * torch.tensor(g_out.astype(np.float64))).sum()
    if LAB and LAB_AUX:  # fold the aux MSE in so aux_w and the concat injection are checked
        ap, at, am = w["_aux"]
        lib.nh_aux_coef.restype = ctypes.c_float
        coef = float(lib.nh_aux_coef())
        L = L + 0.5 * coef / out.shape[0] * (am * (ap - at) ** 2).sum()
    L.backward()

    ok = True
    for name in enc_names:
        n = getattr(lib, f"nh_numel_{name}")()
        ga = np.empty(n, dtype=np.float32)
        getattr(lib, f"nh_grad_{name}")(ga.ctypes.data_as(VP))
        gt = (w[name].grad.numpy().reshape(-1) if w[name].grad is not None
          else np.zeros(w[name].numel()))  # frozen/detached weight: reference grad is zero
        denom = max(1.0, np.abs(gt).max())
        rel = np.abs(ga - gt).max() / denom
        good = rel < 1e-3
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] torch {name:8s} max_rel_err={rel:.2e}")
    return ok


def dec_check(lib, obs_d, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, B, H):
    """float64 torch replica of the 5-head pointer decoder: forward values,
    weight grads, keygrad (grad into the encoder's slot features) and
    grad_input (grad into the decoder's hidden-state input)."""
    import torch
    OD = lib.nh_dec_od()
    HEADS = lib.nh_heads()
    N_ACT = lib.nh_num_actions()
    N_DIRS = OD - N_ACT - HEADS * 55       # 8 pre-champion, 48 with per-verb dir heads
    PAD = lib.nh_dec_pad()

    # CUDA forward (encoder+decoder) and backward (decoder only)
    out_d = dev(B * (OD + 1) * 4)
    lib.nh_dec_forward(out_d, obs_d, B)
    out_cuda = d2h(out_d, B * (OD + 1)).reshape(B, OD + 1)
    rng = np.random.default_rng(11)
    g = rng.standard_normal((B, OD + 1)).astype(np.float32)
    gl_d, _ = h2d(np.ascontiguousarray(g[:, :OD]))
    gv_d, _ = h2d(np.ascontiguousarray(g[:, OD]))
    di_d = dev(B * H * 4)
    lib.nh_dec_backward(gl_d, gv_d, di_d, B)
    di_cuda = d2h(di_d, B * H).reshape(B, H)
    IH = lib.nh_numel_inv1_b(); SK = 16
    kg_cuda = np.empty(B * 55 * IH, dtype=np.float32)
    lib.nh_dec_keygrad(kg_cuda.ctypes.data_as(VP), B)

    # torch replica: hidden state and keys detached so grads are decoder-local,
    # matching what the CUDA decoder backward produces (the encoder chain gets
    # these via grad_input / keygrad separately).
    h_full, invh, _, spkeys = torch_encoder(lib, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, H)
    h_in = h_full.detach().clone().requires_grad_(True)
    s_k = invh.detach().clone().requires_grad_(True)
    sp_k = spkeys.detach().clone().requires_grad_(True)     # (B,8,16)
    lin_w = getw(lib, "dec_lin_w", (PAD, H))
    q_w = getw(lib, "dec_q_w", ((HEADS + 1) * IH, H))
    k_w = getw(lib, "dec_k_w", (IH, IH))
    tau = getw(lib, "dec_tau", (lib.nh_numel_dec_tau(),))  # padded; first HEADS live

    tmp = h_in @ lin_w.T                                    # (B,PAD), rows N_ACT+48+1 used
    qall = (h_in @ q_w.T).reshape(B, HEADS + 1, IH)
    q = qall[:, :HEADS]
    kmat = s_k @ k_w.T                                      # (B,55,16)
    qn = q.norm(dim=2) + 1e-6
    kn = kmat.norm(dim=2) + 1e-6
    cos = torch.einsum('bhk,bik->bhi', q, kmat) / qn[:, :, None]
    slot = torch.exp(tau[:HEADS])[None, :, None] * cos     # (B,HEADS,55) log-tau
    # spell head: dot(q_spell, key_s) / 4 (dot-product pointer, no tau)
    spell = torch.einsum('bk,bsk->bs', qall[:, HEADS, :SK], sp_k) * 0.25  # spell query uses the first SK dims
    N_DIRS_LIN = 48
    dirlin = tmp[:, N_ACT:N_ACT+N_DIRS_LIN]
    if DIRPTR:
        t_k = _G['t16'].detach().clone().requires_grad_(True)     # (B,80,16) token keys, leaf
        dxy = _G['dxy']
        qd_w = getw(lib, "dec_qd_w", (6 * 16, H)); kd_w = getw(lib, "dec_kd_w", (16, 16)); taud = getw(lib, "dec_taud", (8,))
        qd = (h_in @ qd_w.T).reshape(B, 6, 16); kd = t_k @ kd_w.T
        qdn = qd.norm(dim=2) + 1e-6; kdn = kd.norm(dim=2) + 1e-6
        cosd = torch.einsum('bhk,btk->bht', qd, kd) / (qdn[:, :, None] * kdn[:, None, :])   # (B,6,80)
        # octant per token, mirroring nh_tok_octant (E=0..., mapped to N,S,W,E,NW,NE,SW,SE)
        octs = np.full((B, 80), -1)
        mp = [3, 7, 1, 6, 2, 4, 0, 5]
        for b in range(B):
            for tk in range(80):
                cx, cy = dxy[b, tk, 0] * 79, dxy[b, tk, 1] * 21
                if abs(cx) < 2.6 and abs(cy) < 2.6: continue
                a = np.degrees(np.arctan2(cy, cx)); a = a + 360 if a < 0 else a
                octs[b, tk] = mp[int(np.floor((a + 22.5) / 45.0)) % 8]
        extra = torch.zeros(B, 6, 8, dtype=torch.float64)
        for b in range(B):
            for h in range(6):
                for d in range(8):
                    idx = [tk for tk in range(80) if octs[b, tk] == d]
                    if idx:
                        extra[b, h, d] = torch.exp(taud[h]) * cosd[b, h, idx].max()
        dirlin = dirlin + extra.reshape(B, 48)
    out = torch.cat([tmp[:, :N_ACT], slot.reshape(B, HEADS * 55),
                     dirlin, spell,
                     tmp[:, N_ACT+N_DIRS_LIN:N_ACT+N_DIRS_LIN+1]], dim=1)
    (out * torch.tensor(g.astype(np.float64))).sum().backward()

    ok = True
    rel = np.abs(out_cuda - out.detach().numpy()).max() / max(1.0, np.abs(out.detach().numpy()).max())
    good = rel < 1e-4
    ok = ok and good
    print(f"  [{'OK ' if good else 'FAIL'}] torch dec_out   max_rel_err={rel:.2e}")

    dec_pairs = [("dec_lin_w", lin_w), ("dec_q_w", q_w), ("dec_k_w", k_w), ("dec_tau", tau)]
    if DIRPTR:
        dec_pairs += [("dec_qd_w", qd_w), ("dec_kd_w", kd_w), ("dec_taud", taud)]
    for name, ref in dec_pairs:
        n = getattr(lib, f"nh_numel_{name}")()
        ga = np.empty(n, dtype=np.float32)
        getattr(lib, f"nh_grad_{name}")(ga.ctypes.data_as(VP))
        gt = ref.grad.numpy().reshape(-1)
        denom = max(1.0, np.abs(gt).max())
        rel = np.abs(ga - gt).max() / denom
        good = rel < 1e-3
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] torch {name:9s} max_rel_err={rel:.2e}")

    extra_pairs = [("keygrad", kg_cuda, s_k), ("grad_input", di_cuda, h_in)]
    if DIRPTR:
        tg_cuda = np.empty(B * 80 * 16, dtype=np.float32); lib.nh_dec_tokgrad(tg_cuda.ctypes.data_as(VP), B)
        extra_pairs.append(("tokgrad", tg_cuda, t_k))
    for name, cuda_g, ref in extra_pairs:
        gt = ref.grad.numpy().reshape(-1)
        denom = max(1.0, np.abs(gt).max())
        rel = np.abs(cuda_g.reshape(-1) - gt).max() / denom
        good = rel < 1e-3
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] torch dec {name:9s} max_rel_err={rel:.2e}")
    return ok


if __name__ == "__main__":
    if "--no-build" not in sys.argv:
        build()
    ok = run(load())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
