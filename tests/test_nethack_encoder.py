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
IDEMB = bool(os.environ.get("NH_TEST_IDEMB"))
IDE_NAMES = ["ide_role_w", "ide_race_w", "ide_gend_w", "ide_algn_w"]
if IDEMB:
    WEIGHT_NAMES += IDE_NAMES


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
        "-Xcompiler=-DPLATFORM_DESKTOP", "-Xcompiler=-fPIC",
        "-Xcompiler=-fopenmp", "-O2",
        "-L" + os.path.join(root, "vendor", "fast-nle", "build"), "-lnethack",
        "-Xlinker", "-rpath", "-Xlinker",
        os.path.join(root, "vendor", "fast-nle", "build"),
        "-lcublas", "-lcusolver", "-lcurand", "-lnvidia-ml", "-lcudart",
    ]
    try:
        import nvidia.nccl
        nccl = nvidia.nccl.__path__[0]
        cmd += ["-I" + os.path.join(nccl, "include"),
                "-L" + os.path.join(nccl, "lib"), "-lnccl"]
    except ImportError:
        cmd += ["-lnccl"]
    cmd.append(f"-DNH_ID_EMBED={int(IDEMB)}")  # default is 1; arm A needs explicit 0
    print("building:", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    ], axis=1).astype(np.int64).astype(np.uint32)
    for k in range(4):
        obs[:, bl_off + k::4][:, 27:104] = ((ex >> (8 * k)) & 0xFF).astype(np.float32)
    # inventory entities: 55 slot glyphs int16 LE, tail padded (5976)
    inv_off = bl_off + 104 * 4
    inv = rng.integers(0, max_glyph_used, size=(B, 55)).astype(np.int32)
    inv[:, ::2] = rng.integers(1906, 2359, size=(B, 28))  # object glyphs: armcat coverage
    n_items = rng.integers(3, 12, size=B)
    for b in range(B):
        inv[b, n_items[b]:] = 5976
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
    msg_len = obs_size - msg_off
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
    E_res  = w["embed_w"] = getw(lib, "embed_w", (5977, 32))
    K_w    = w["ekind_w"] = getw(lib, "ekind_w", (14, 32))
    S_w    = w["esub_w"]  = getw(lib, "esub_w", (getattr(lib, "nh_numel_esub_w")() // 32, 32))
    kind_map, sub_map = glyph_map()
    E = E_res + K_w[torch.tensor(kind_map)] + S_w[torch.tensor(sub_map)]   # E_eff
    loc_w  = w["loc_w"]   = getw(lib, "loc_w", (256, CROP * CROP * 32))
    loc_b  = w["loc_b"]   = getw(lib, "loc_b", (256,))
    g1_w   = w["glb1_w"]  = getw(lib, "glb1_w", (16, PW * PH * 32))
    g1_b   = w["glb1_b"]  = getw(lib, "glb1_b", (16,))
    g1_xy  = w["glb1_xy"] = getw(lib, "glb1_xy", (16, 2))
    g2_w   = w["glb2_w"]  = getw(lib, "glb2_w", (128, 16))
    g2_b   = w["glb2_b"]  = getw(lib, "glb2_b", (128,))
    inv1_w = w["inv1_w"]  = getw(lib, "inv1_w", (16, 32))
    inv1_b = w["inv1_b"]  = getw(lib, "inv1_b", (16,))
    inv1s_w = w["inv1s_w"] = getw(lib, "inv1s_w", (16, 24))
    invt_w = w["invt_w"] = getw(lib, "invt_w", (16, 32))
    inv2_w = w["inv2_w"]  = getw(lib, "inv2_w", (128, 16))
    inv2_b = w["inv2_b"]  = getw(lib, "inv2_b", (128,))
    bl_w   = w["bl_w"]    = getw(lib, "bl_w", (64, lib.nh_bl_feat()))
    bl_b   = w["bl_b"]    = getw(lib, "bl_b", (64,))
    proj_w = w["proj_w"]  = getw(lib, "proj_w", (H, lib.nh_concat()))
    proj_b = w["proj_b"]  = getw(lib, "proj_b", (H,))
    msg_w  = w["msg_w"]   = getw(lib, "msg_w", (lib.nh_numel_msg_w() // 32, 32))
    spk_w  = w["spk_w"]   = getw(lib, "spk_w", (16, 36))
    spk2_w = w["spk2_w"]  = getw(lib, "spk2_w", (16, 16))
    spk2_b = w["spk2_b"]  = getw(lib, "spk2_b", (16,))

    # local: crop glyph ids with pad off-map
    hx, hy = bl_vals[:, 0], bl_vals[:, 1]
    crop_idx = np.full((B, CROP * CROP), PAD, dtype=np.int64)
    for b in range(B):
        for p in range(CROP * CROP):
            r, c = hy[b] - 4 + p // CROP, hx[b] - 4 + p % CROP
            if 0 <= r < ROWS and 0 <= c < COLS:
                crop_idx[b, p] = glyphs[b, r * COLS + c]
    x_local = E[torch.tensor(crop_idx)].reshape(B, -1)
    loc = torch.relu(x_local @ loc_w.T + loc_b)
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
    t16 = torch.relu(xp @ g1_w.T + torch.tensor(dxy) @ g1_xy.T + g1_b)
    t128 = t16 @ g2_w.T
    glb = torch.relu(t128.max(dim=1).values + g2_b)
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
    xi = E[torch.tensor(inv_vals.astype(np.int64))]
    # discovered-type channel: pad (5976) slots contribute hard zero
    xt = E[torch.tensor(itr_vals.astype(np.int64))]
    kt = torch.tensor((itr_vals != 5976).astype(np.float64))[:, :, None]
    invh = torch.relu(xi @ inv1_w.T + kt * (xt @ invt_w.T)
                      + torch.tensor(sf) @ inv1s_w.T + inv1_b)  # (B,55,16)
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
            ids.append(((key * 2654435761) & 0xFFFFFFFF) >> (32 - 12))
        cnt = len(ids)
        s = msg_w[torch.tensor(ids, dtype=torch.long)].sum(dim=0) if cnt else torch.zeros(32, dtype=torch.float64)
        rows.append(s / np.sqrt(cnt + 1))
    msg_sum = torch.stack(rows, dim=0)   # (B, 32); grad flows to msg_w
    # spell-key path: per slot, key = spk_w . [e_eff(book glyph) | known,
    # lev/7, fail/100, know/20000]; sum-pool feeds the trunk
    spk_w = w["spk_w"]
    spkeys = []
    for s in range(8):
        c = 23 + 4 * s
        sid = torch.tensor(ex_vals[:, c].astype(np.int64))
        sg = torch.clamp(sid + 1906, max=5975)
        emb = torch.where((sid > 0)[:, None], E[sg], torch.zeros_like(E[sg]))
        sc = torch.stack([
            (sid > 0).double(),
            torch.clamp(torch.tensor(ex_vals[:, c + 1]) * 0.142857, max=1.0),
            torch.clamp(torch.tensor(ex_vals[:, c + 2]) * 0.01, max=1.0),
            torch.clamp(torch.tensor(ex_vals[:, c + 3]) * 0.00005, max=1.0),
        ], dim=1)
        spkeys.append(torch.relu(torch.cat([emb, sc], dim=1) @ spk_w.T))  # (B,16)
    sk = torch.stack(spkeys, dim=1)                            # (B,8,16)
    spool = torch.relu((sk @ spk2_w.T).max(dim=1).values + spk2_b)
    parts = [loc, glb, invp, blh, fb, msg_sum, spool]
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
    concat = torch.cat(parts, dim=1)
    out = torch.relu(concat @ proj_w.T + proj_b)
    return out, invh, w, torch.stack(spkeys, dim=1)


def run(lib):
    B, hidden = 4, 24
    lib.nh_init(B, hidden)
    obs_size = lib.nh_obs_size()
    grid = lib.nh_grid()
    vocab = lib.nh_glyph_vocab()
    print(f"obs_size={obs_size} grid={grid} vocab={vocab} "
          f"bl_feat={lib.nh_bl_feat()} concat={lib.nh_concat()} dec_od={lib.nh_dec_od()}")

    # glb1_xy zero-inits; randomize it so a broken dx,dy forward term is visible
    wxy = np.random.default_rng(5).standard_normal(lib.nh_numel_glb1_xy()).astype(np.float32)
    lib.nh_set_glb1_xy(wxy.ctypes.data_as(VP))
    if IDEMB:  # same idiom: zero-init tables would hide forward bugs
        for i, nm in enumerate(IDE_NAMES):
            wv = np.random.default_rng(6 + i).standard_normal(
                getattr(lib, f"nh_numel_{nm}")()).astype(np.float32)
            getattr(lib, f"nh_set_{nm}")(wv.ctypes.data_as(VP))

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
        return float((out * g_out).sum())

    # Analytic grads: forward then backward with dL/dout = g_out.
    L0 = forward_loss()
    grad_d, _ = h2d(g_out)  # backward mutates grad in place
    lib.nh_backward(grad_d, B)

    enc_names = ["ekind_w", "esub_w", "proj_w", "proj_b", "bl_w", "bl_b", "loc_w", "loc_b",
                 "glb1_w", "glb1_xy", "glb1_b", "glb2_w", "glb2_b",
                 "inv1_w", "inv1_b", "inv1s_w", "invt_w", "inv2_w", "inv2_b", "embed_w", "msg_w"]
    if IDEMB:
        enc_names += IDE_NAMES

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
    fd_skip = {"glb1_w", "glb1_xy", "glb1_b", "inv2_w", "inv1_w", "inv1_b", "invt_w"}
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
        ok = checked >= 3 and max_rel < rel_tol
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
    (out * torch.tensor(g_out.astype(np.float64))).sum().backward()

    ok = True
    for name in enc_names:
        n = getattr(lib, f"nh_numel_{name}")()
        ga = np.empty(n, dtype=np.float32)
        getattr(lib, f"nh_grad_{name}")(ga.ctypes.data_as(VP))
        gt = w[name].grad.numpy().reshape(-1)
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
    kg_cuda = np.empty(B * 55 * 16, dtype=np.float32)
    lib.nh_dec_keygrad(kg_cuda.ctypes.data_as(VP), B)

    # torch replica: hidden state and keys detached so grads are decoder-local,
    # matching what the CUDA decoder backward produces (the encoder chain gets
    # these via grad_input / keygrad separately).
    h_full, invh, _, spkeys = torch_encoder(lib, glyphs, bl_vals, ex_vals, inv_vals, st_vals, itr_vals, msg, H)
    h_in = h_full.detach().clone().requires_grad_(True)
    s_k = invh.detach().clone().requires_grad_(True)
    sp_k = spkeys.detach().clone().requires_grad_(True)     # (B,8,16)
    lin_w = getw(lib, "dec_lin_w", (PAD, H))
    q_w = getw(lib, "dec_q_w", ((HEADS + 1) * 16, H))
    k_w = getw(lib, "dec_k_w", (16, 16))
    tau = getw(lib, "dec_tau", (lib.nh_numel_dec_tau(),))  # padded; first HEADS live

    tmp = h_in @ lin_w.T                                    # (B,PAD), rows N_ACT+48+1 used
    qall = (h_in @ q_w.T).reshape(B, HEADS + 1, 16)
    q = qall[:, :HEADS]
    kmat = s_k @ k_w.T                                      # (B,55,16)
    qn = q.norm(dim=2) + 1e-6
    kn = kmat.norm(dim=2) + 1e-6
    cos = torch.einsum('bhk,bik->bhi', q, kmat) / (qn[:, :, None] * kn[:, None, :])
    slot = torch.exp(tau[:HEADS])[None, :, None] * cos     # (B,HEADS,55) log-tau
    # spell head: dot(q_spell, key_s) / 4 (dot-product pointer, no tau)
    spell = torch.einsum('bk,bsk->bs', qall[:, HEADS], sp_k) * 0.25
    N_DIRS_LIN = 48
    out = torch.cat([tmp[:, :N_ACT], slot.reshape(B, HEADS * 55),
                     tmp[:, N_ACT:N_ACT+N_DIRS_LIN], spell,
                     tmp[:, N_ACT+N_DIRS_LIN:N_ACT+N_DIRS_LIN+1]], dim=1)
    (out * torch.tensor(g.astype(np.float64))).sum().backward()

    ok = True
    rel = np.abs(out_cuda - out.detach().numpy()).max() / max(1.0, np.abs(out.detach().numpy()).max())
    good = rel < 1e-4
    ok = ok and good
    print(f"  [{'OK ' if good else 'FAIL'}] torch dec_out   max_rel_err={rel:.2e}")

    for name, ref in [("dec_lin_w", lin_w), ("dec_q_w", q_w),
                      ("dec_k_w", k_w), ("dec_tau", tau)]:
        n = getattr(lib, f"nh_numel_{name}")()
        ga = np.empty(n, dtype=np.float32)
        getattr(lib, f"nh_grad_{name}")(ga.ctypes.data_as(VP))
        gt = ref.grad.numpy().reshape(-1)
        denom = max(1.0, np.abs(gt).max())
        rel = np.abs(ga - gt).max() / denom
        good = rel < 1e-3
        ok = ok and good
        print(f"  [{'OK ' if good else 'FAIL'}] torch {name:9s} max_rel_err={rel:.2e}")

    for name, cuda_g, ref in [("keygrad", kg_cuda, s_k), ("grad_input", di_cuda, h_in)]:
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
