"""Manifest-driven visual export for encounter NPCs.

Reads monster manifest visual sections + gameval constants + OSRS cache to
produce .models and .anims binaries plus a C header for encounter NPC rendering.
Replaces hardcoded per-encounter export scripts with a single manifest-driven tool.

Usage:
    uv run python tools/export_encounter_npcs.py \
        --group zulrah \
        --modern-cache ../../../.refs/osrs-cache-modern \
        --manifest tools/monsters_manifest.json \
        --output-dir data
"""

import argparse
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# add scripts dir for existing infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from modern_cache_reader import (
    ModernCacheReader,
    read_i32,
    read_string,
    read_u8,
    read_u16,
    read_u24,
    read_u32,
)
from modern_cache_reader import parse_sequence as parse_modern_sequence
from export_models import (
    ModelData,
    _merge_models,
    decode_model,
    load_model_modern,
    write_models_binary,
)
from export_animations import (
    FrameDef,
    SequenceDef,
    _parse_normal_frame,
    load_modern_framebases,
    write_animations_binary,
)

# gameval parser from tools dir
sys.path.insert(0, str(Path(__file__).parent))
from gameval_parser import load_gameval, resolve_names

# modern cache layout
MODERN_NPC_CONFIG_GROUP = 9
MODERN_SPOTANIM_CONFIG_GROUP = 13
MODERN_SEQ_CONFIG_GROUP = 12
MODERN_FRAME_INDEX = 0
MODERN_FRAMEBASE_INDEX = 1

@dataclass
class NpcDef:
    """NPC definition from modern OSRS cache."""

    npc_id: int = 0
    name: str = ""
    model_ids: list[int] = field(default_factory=list)
    chathead_model_ids: list[int] = field(default_factory=list)
    size: int = 1
    idle_anim: int = -1
    walk_anim: int = -1
    run_anim: int = -1
    turn_180_anim: int = -1
    turn_cw_anim: int = -1
    turn_ccw_anim: int = -1
    attack_anim: int = -1
    death_anim: int = -1
    combat_level: int = 0
    width_scale: int = 128
    height_scale: int = 128
    recolor_src: list[int] = field(default_factory=list)
    recolor_dst: list[int] = field(default_factory=list)
    retexture_src: list[int] = field(default_factory=list)
    retexture_dst: list[int] = field(default_factory=list)


@dataclass
class SpotAnimDef:
    """SpotAnim (GFX) definition from modern OSRS cache."""

    id: int = 0
    model_id: int = -1
    seq_id: int = -1
    recolor_src: list[int] = field(default_factory=list)
    recolor_dst: list[int] = field(default_factory=list)
    width_scale: int = 128
    height_scale: int = 128
    rotation: int = 0
    ambient: int = 0
    contrast: int = 0

def parse_modern_npc_def(npc_id: int, data: bytes) -> NpcDef:
    """Parse modern OSRS NPC definition from opcode stream."""
    d = NpcDef(npc_id=npc_id)
    buf = io.BytesIO(data)

    while True:
        opcode_byte = buf.read(1)
        if not opcode_byte:
            break
        opcode = opcode_byte[0]

        if opcode == 0:
            break
        elif opcode == 1:
            count = read_u8(buf)
            d.model_ids = [read_u16(buf) for _ in range(count)]
        elif opcode == 2:
            d.name = read_string(buf)
        elif opcode == 3:
            read_string(buf)
        elif opcode == 5:
            count = read_u8(buf)
            for _ in range(count):
                read_u16(buf)
        elif opcode == 12:
            d.size = read_u8(buf)
        elif opcode == 13:
            d.idle_anim = read_u16(buf)
        elif opcode == 14:
            d.walk_anim = read_u16(buf)
        elif opcode == 15:
            d.turn_180_anim = read_u16(buf)
        elif opcode == 16:
            d.turn_cw_anim = read_u16(buf)
        elif opcode == 17:
            d.walk_anim = read_u16(buf)
            d.turn_180_anim = read_u16(buf)
            d.turn_cw_anim = read_u16(buf)
            d.turn_ccw_anim = read_u16(buf)
        elif opcode == 18:
            read_u16(buf)
        elif 30 <= opcode <= 34:
            read_string(buf)
        elif opcode == 40:
            count = read_u8(buf)
            for _ in range(count):
                d.recolor_src.append(read_u16(buf))
                d.recolor_dst.append(read_u16(buf))
        elif opcode == 41:
            count = read_u8(buf)
            for _ in range(count):
                d.retexture_src.append(read_u16(buf))
                d.retexture_dst.append(read_u16(buf))
        elif opcode == 60:
            count = read_u8(buf)
            d.chathead_model_ids = [read_u16(buf) for _ in range(count)]
        elif 74 <= opcode <= 79:
            read_u16(buf)
        elif opcode == 93:
            pass
        elif opcode == 95:
            d.combat_level = read_u16(buf)
        elif opcode == 97:
            d.width_scale = read_u16(buf)
        elif opcode == 98:
            d.height_scale = read_u16(buf)
        elif opcode == 99:
            pass
        elif opcode == 100:
            read_u8(buf)
        elif opcode == 101:
            read_u8(buf)
        elif opcode == 102:
            bitfield = read_u8(buf)
            bit_count = 0
            tmp = bitfield
            while tmp != 0:
                bit_count += 1
                tmp >>= 1
            for i in range(bit_count):
                if bitfield & (1 << i):
                    pos = buf.tell()
                    peek = buf.read(1)
                    if peek and peek[0] < 128:
                        buf.seek(pos)
                        read_u16(buf)
                    else:
                        buf.seek(pos)
                        read_i32(buf)
                    pos2 = buf.tell()
                    peek2 = buf.read(1)
                    if peek2 and peek2[0] < 128:
                        buf.seek(pos2)
                        read_u16(buf)
                    else:
                        buf.seek(pos2)
                        read_i32(buf)
        elif opcode == 103:
            read_u16(buf)
        elif opcode == 106:
            read_u16(buf)
            read_u16(buf)
            length = read_u8(buf)
            for _ in range(length + 1):
                read_u16(buf)
        elif opcode == 107:
            pass
        elif opcode == 108:
            pass
        elif opcode == 109:
            pass
        elif opcode == 111:
            pass
        elif opcode == 114:
            read_u16(buf)
        elif opcode == 115:
            read_u16(buf)
            read_u16(buf)
            read_u16(buf)
            read_u16(buf)
        elif opcode == 116:
            read_u16(buf)
        elif opcode == 117:
            read_u16(buf)
            read_u16(buf)
            read_u16(buf)
        elif opcode == 118:
            read_u16(buf)
            read_u16(buf)
            read_u16(buf)
            length = read_u8(buf)
            for _ in range(length + 1):
                read_u16(buf)
        elif opcode == 122:
            pass
        elif opcode == 123:
            pass
        elif opcode == 124:
            read_u16(buf)
        elif opcode == 125:
            read_u8(buf)
        elif opcode == 126:
            read_u16(buf)
        elif opcode == 128:
            read_u8(buf)
        elif opcode == 129:
            pass
        elif opcode == 130:
            pass
        elif opcode == 145:
            pass
        elif opcode == 146:
            read_u16(buf)
        elif opcode == 147:
            pass
        elif opcode == 249:
            count_val = read_u8(buf)
            for _ in range(count_val):
                is_string = read_u8(buf)
                read_u24(buf)
                if is_string:
                    read_string(buf)
                else:
                    read_u32(buf)
        else:
            print(f"  warning: unknown npc opcode {opcode} at npc {npc_id}, pos {buf.tell()}", file=sys.stderr)
            break

    return d


def parse_modern_spotanim(spotanim_id: int, data: bytes) -> SpotAnimDef:
    """Parse modern SpotAnim/GFX definition from opcode stream."""
    d = SpotAnimDef(id=spotanim_id)
    buf = io.BytesIO(data)

    while True:
        opcode_byte = buf.read(1)
        if not opcode_byte:
            break
        opcode = opcode_byte[0]

        if opcode == 0:
            break
        elif opcode == 1:
            d.model_id = read_u16(buf)
        elif opcode == 2:
            d.seq_id = read_u16(buf)
        elif opcode == 4:
            d.width_scale = read_u16(buf)
        elif opcode == 5:
            d.height_scale = read_u16(buf)
        elif opcode == 6:
            d.rotation = read_u16(buf)
        elif opcode == 7:
            d.ambient = read_u8(buf)
        elif opcode == 8:
            d.contrast = read_u8(buf)
        elif opcode == 40:
            count = read_u8(buf)
            for _ in range(count):
                d.recolor_src.append(read_u16(buf))
                d.recolor_dst.append(read_u16(buf))
        elif opcode == 41:
            count = read_u8(buf)
            for _ in range(count):
                read_u16(buf)
                read_u16(buf)
        else:
            print(f"  warning: unknown spotanim opcode {opcode} at gfx {spotanim_id}", file=sys.stderr)
            break

    return d

def apply_recolors(md: ModelData, src: list[int], dst: list[int]) -> None:
    """Apply recolor pairs to model face colors in-place."""
    for i, color in enumerate(md.face_colors):
        for s, d in zip(src, dst):
            if color == s:
                md.face_colors[i] = d
                break


def apply_scale(md: ModelData, width_scale: int, height_scale: int) -> None:
    """Apply NPC width/height scale to vertex positions in-place."""
    if width_scale == 128 and height_scale == 128:
        return
    ws = width_scale / 128.0
    hs = height_scale / 128.0
    for i in range(md.vertex_count):
        md.vertices_x[i] = int(md.vertices_x[i] * ws)
        md.vertices_y[i] = int(md.vertices_y[i] * hs)
        md.vertices_z[i] = int(md.vertices_z[i] * ws)

def main() -> None:
    """Manifest-driven export of encounter NPC models + animations."""
    parser = argparse.ArgumentParser(
        description="export encounter NPC models + animations from modern cache via manifest",
    )
    parser.add_argument(
        "--group", required=True,
        help="encounter group name to export (matches visual.group in manifest)",
    )
    parser.add_argument(
        "--modern-cache", type=Path, required=True,
        help="path to modern OpenRS2 flat-file cache",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("tools/monsters_manifest.json"),
        help="path to monsters manifest JSON",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="output directory for generated files",
    )
    args = parser.parse_args()
    group = args.group
    group_upper = group.upper()
    print("loading gameval constants...")
    anim_ids, npc_ids, spotanim_ids = load_gameval()
    print(f"  {len(anim_ids)} anims, {len(npc_ids)} npcs, {len(spotanim_ids)} spotanims")
    print(f"\nloading manifest {args.manifest}, filtering group={group!r}...")
    with open(args.manifest) as f:
        manifest = json.load(f)

    entries = [e for e in manifest if e.get("visual", {}).get("group") == group]
    if not entries:
        print(f"error: no manifest entries with visual.group={group!r}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(entries)} NPCs in group {group!r}")
    reader = ModernCacheReader(args.modern_cache)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nreading NPC definitions from modern cache (index 2, group 9)...")
    npc_files = reader.read_group(2, MODERN_NPC_CONFIG_GROUP)

    npc_defs: dict[int, NpcDef] = {}
    npc_attack_anims: dict[int, list[int]] = {}
    npc_extra_anims: dict[int, list[int]] = {}
    npc_comments: dict[int, str] = {}
    all_anim_ids: set[int] = set()
    all_spotanim_names: set[str] = set()
    anim_name_to_id: dict[str, int] = {}

    for entry in entries:
        npc_id = entry["npc_id"]
        vis = entry["visual"]
        comment = entry.get("comment", entry.get("index", ""))

        if npc_id not in npc_files:
            print(f"  NPC {npc_id} ({comment}): NOT FOUND in cache", file=sys.stderr)
            sys.exit(1)

        npc = parse_modern_npc_def(npc_id, npc_files[npc_id])
        npc_defs[npc_id] = npc
        npc_comments[npc_id] = comment

        # resolve attack anim names -> IDs
        attack_names = vis.get("attack_anims", [])
        if attack_names:
            attack_ids = resolve_names(attack_names, anim_ids, context=f"NPC {npc_id} attack_anims")
        else:
            attack_ids = []
        npc_attack_anims[npc_id] = attack_ids
        for name, aid in zip(attack_names, attack_ids):
            anim_name_to_id[name] = aid

        # resolve extra anim names -> IDs
        extra_names = vis.get("extra_anims", [])
        if extra_names:
            extra_ids = resolve_names(extra_names, anim_ids, context=f"NPC {npc_id} extra_anims")
        else:
            extra_ids = []
        npc_extra_anims[npc_id] = extra_ids
        for name, eid in zip(extra_names, extra_ids):
            anim_name_to_id[name] = eid

        # collect all anim IDs: idle + walk from cache, attack + extra from gameval
        for anim_id in [npc.idle_anim, npc.walk_anim]:
            if anim_id >= 0:
                all_anim_ids.add(anim_id)
        all_anim_ids.update(attack_ids)
        all_anim_ids.update(extra_ids)

        # collect spotanim names
        all_spotanim_names.update(vis.get("spotanims", []))

        print(f"  NPC {npc_id} ({comment}): models={npc.model_ids}, "
              f"idle={npc.idle_anim}, walk={npc.walk_anim}, "
              f"attacks={attack_ids}, extras={extra_ids}")
    print("\nexporting NPC models...")
    all_models: list[ModelData] = []

    for npc_id, npc in sorted(npc_defs.items()):
        sub_models: list[ModelData] = []
        for mid in npc.model_ids:
            raw = load_model_modern(reader, mid)
            if raw is None:
                print(f"  warning: model {mid} not found for NPC {npc_id}")
                continue
            md = decode_model(mid, raw)
            if md is None:
                print(f"  warning: failed to decode model {mid} for NPC {npc_id}")
                continue
            sub_models.append(md)

        if not sub_models:
            print(f"  NPC {npc_id}: no models decoded")
            continue

        if len(sub_models) == 1:
            merged = sub_models[0]
        else:
            merged = _merge_models(sub_models)

        if npc.recolor_src:
            apply_recolors(merged, npc.recolor_src, npc.recolor_dst)
        apply_scale(merged, npc.width_scale, npc.height_scale)

        merged.model_id = 0xC0000 + npc_id
        all_models.append(merged)
        print(f"  NPC {npc_id} ({npc.name}): {merged.vertex_count} verts, {merged.face_count} faces")
    spotanim_defs: dict[int, SpotAnimDef] = {}
    spotanim_name_for_id: dict[int, str] = {}

    if all_spotanim_names:
        print(f"\nresolving {len(all_spotanim_names)} unique spotanim names...")
        spotanim_files = reader.read_group(2, MODERN_SPOTANIM_CONFIG_GROUP)

        for name in sorted(all_spotanim_names):
            gfx_ids = resolve_names([name], spotanim_ids, context="spotanims")
            gfx_id = gfx_ids[0]
            spotanim_name_for_id[gfx_id] = name

            if gfx_id not in spotanim_files:
                print(f"  GFX {gfx_id} ({name}): NOT FOUND in cache", file=sys.stderr)
                sys.exit(1)

            sa = parse_modern_spotanim(gfx_id, spotanim_files[gfx_id])
            spotanim_defs[gfx_id] = sa
            print(f"  GFX {gfx_id} ({name}): model={sa.model_id}, seq={sa.seq_id}")

            if sa.seq_id >= 0:
                all_anim_ids.add(sa.seq_id)

        # export GFX models
        print("\nexporting GFX models...")
        exported_gfx_models: set[int] = set()
        for gfx_id, sa in sorted(spotanim_defs.items()):
            if sa.model_id < 0:
                continue
            raw = load_model_modern(reader, sa.model_id)
            if raw is None:
                print(f"  warning: GFX {gfx_id} model {sa.model_id} not found")
                continue
            md = decode_model(sa.model_id, raw)
            if md is None:
                print(f"  warning: failed to decode GFX {gfx_id} model {sa.model_id}")
                continue
            if sa.recolor_src:
                apply_recolors(md, sa.recolor_src, sa.recolor_dst)
                md.model_id = 0xD0000 | gfx_id
                print(f"  GFX {gfx_id} model {sa.model_id} -> 0x{md.model_id:X} (recolored): {md.vertex_count} verts")
            else:
                if sa.model_id in exported_gfx_models:
                    continue
                print(f"  GFX {gfx_id} model {sa.model_id}: {md.vertex_count} verts")
            exported_gfx_models.add(md.model_id)
            all_models.append(md)
    models_path = output_dir / f"{group}.models"
    write_models_binary(models_path, all_models)
    file_size = models_path.stat().st_size
    print(f"\nwrote {len(all_models)} models ({file_size:,} bytes) to {models_path}")
    print("\nexporting animations...")
    seq_files = reader.read_group(2, MODERN_SEQ_CONFIG_GROUP)

    sequences: dict[int, SequenceDef] = {}
    for seq_id in sorted(all_anim_ids):
        if seq_id not in seq_files:
            print(f"  warning: sequence {seq_id} not found in cache")
            continue
        modern_seq = parse_modern_sequence(seq_id, seq_files[seq_id])
        seq = SequenceDef(
            seq_id=modern_seq.seq_id,
            frame_count=modern_seq.frame_count,
            frame_delays=modern_seq.frame_delays,
            primary_frame_ids=modern_seq.primary_frame_ids,
            frame_step=modern_seq.frame_step,
            interleave_order=modern_seq.interleave_order,
            priority=modern_seq.forced_priority,
            loop_count=modern_seq.max_loops,
            walk_flag=modern_seq.priority,
            run_flag=modern_seq.precedence_animating,
        )
        sequences[seq_id] = seq
        print(f"  seq {seq_id}: {seq.frame_count} frames")

    # collect needed frame groups
    needed_groups: set[int] = set()
    for seq_id in all_anim_ids & set(sequences.keys()):
        seq = sequences[seq_id]
        for fid in seq.primary_frame_ids:
            if fid != -1:
                needed_groups.add(fid >> 16)

    print(f"  loading {len(needed_groups)} frame archives...")

    # first pass: discover framebase IDs from frame data headers
    needed_base_ids: set[int] = set()
    raw_frame_data: dict[int, dict[int, bytes]] = {}
    for group_id in sorted(needed_groups):
        try:
            files = reader.read_group(MODERN_FRAME_INDEX, group_id)
        except (KeyError, FileNotFoundError):
            print(f"  warning: frame archive {group_id} not found")
            continue
        raw_frame_data[group_id] = files
        for file_data in files.values():
            if len(file_data) >= 2:
                fb_id = (file_data[0] << 8) | file_data[1]
                needed_base_ids.add(fb_id)

    print(f"  loading {len(needed_base_ids)} framebases...")
    framebases = load_modern_framebases(reader, needed_base_ids)
    print(f"  loaded {len(framebases)} framebases")

    # second pass: parse frames
    all_frames: dict[int, dict[int, FrameDef]] = {}
    for group_id, files in raw_frame_data.items():
        frames: dict[int, FrameDef] = {}
        for file_id, file_data in files.items():
            if len(file_data) < 3:
                continue
            frame = _parse_normal_frame(group_id, file_id, file_data, framebases)
            if frame is not None:
                frames[file_id] = frame
        if frames:
            all_frames[group_id] = frames

    total_frames = sum(len(v) for v in all_frames.values())
    print(f"  {len(all_frames)} frame archives, {total_frames} total frames")

    # write animations binary
    anims_path = output_dir / f"{group}.anims"
    available_seqs = all_anim_ids & set(sequences.keys())
    write_animations_binary(anims_path, framebases, all_frames, sequences, available_seqs)
    anims_size = anims_path.stat().st_size
    print(f"wrote {len(available_seqs)} sequences ({anims_size:,} bytes) to {anims_path}")
    prefix = group_upper[:3] + "_GEN"

    header_path = output_dir / f"npc_models_{group}.h"
    guard = f"NPC_MODELS_{group_upper}_H"

    print(f"\nwriting C header {header_path}...")

    with open(header_path, "w") as f:
        f.write("/* generated by tools/export_encounter_npcs.py -- do not edit */\n")
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n\n")
        f.write('#include <stdint.h>\n')
        f.write('#include "npc_models.h"  /* for NpcModelMapping typedef */\n\n')

        # NPC model mapping array
        f.write(f"static const NpcModelMapping NPC_MODEL_MAP_{group_upper}_GEN[] = {{\n")
        for npc_id, npc in sorted(npc_defs.items()):
            synth_model = 0xC0000 + npc_id
            idle = npc.idle_anim if npc.idle_anim >= 0 else 0xFFFF
            attacks = npc_attack_anims.get(npc_id, [])
            attack = attacks[0] if attacks else 0xFFFF
            walk = npc.walk_anim if npc.walk_anim >= 0 else 0xFFFF
            comment = npc_comments.get(npc_id, npc.name)
            f.write(f"    {{{npc_id}, 0x{synth_model:X}, {idle}, {attack}, {walk}}},  /* {comment} */\n")
        f.write("};\n\n")

        # animation ID defines
        if anim_name_to_id:
            f.write(f"/* {group} animation IDs */\n")
            for name, aid in sorted(anim_name_to_id.items(), key=lambda x: x[1]):
                f.write(f"#define {prefix}_ANIM_{name}  {aid}\n")
            f.write("\n")

        # spotanim GFX model + animation defines
        if spotanim_defs:
            f.write(f"/* {group} spotanim GFX model + animation IDs */\n")
            for gfx_id, sa in sorted(spotanim_defs.items()):
                gv_name = spotanim_name_for_id.get(gfx_id, f"GFX_{gfx_id}")
                if sa.recolor_src:
                    emit_model_id = 0xD0000 | gfx_id
                else:
                    emit_model_id = sa.model_id
                f.write(f"#define {prefix}_GFX_{gfx_id}_MODEL  {emit_model_id}  /* {gv_name} */\n")
                if sa.seq_id >= 0:
                    f.write(f"#define {prefix}_GFX_{gfx_id}_ANIM   {sa.seq_id}\n")
            f.write("\n")

        f.write(f"#endif /* {guard} */\n")

    print(f"wrote {header_path}")
    print("\ndone.")


if __name__ == "__main__":
    main()
