"""Cook viewer bindings and a distant-car LOD from the existing assets (stdlib only)."""
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parent


def glb(path):
    raw = path.read_bytes()
    size = struct.unpack_from('<I', raw, 12)[0]
    doc = json.loads(raw[20:20 + size])
    return doc, raw[28 + size:]


def accessor(doc, raw, index):
    a = doc['accessors'][index]
    view = doc['bufferViews'][a['bufferView']]
    component = {5123: 'H', 5125: 'I', 5126: 'f'}[a['componentType']]
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[a['type']]
    fmt = '<' + component * width
    stride = view.get('byteStride', struct.calcsize(fmt))
    offset = view.get('byteOffset', 0) + a.get('byteOffset', 0)
    return [struct.unpack_from(fmt, raw, offset + i * stride) for i in range(a['count'])]


def cook_track():
    doc, _ = glb(ROOT / 'assets/track/source/silverstone.glb')
    with (ROOT / 'map_visual.bin').open('rb') as f:
        assert f.read(8) == b'PFVIS004'
        _, textures, count = struct.unpack('<III', f.read(12))
        for _ in range(textures):
            f.seek(struct.unpack('<I', f.read(4))[0], 1)
        records = [struct.unpack('<4I10f', f.read(56)) for _ in range(count)]
    assert count == len(doc['materials'])
    names = {m['name']: i for i, m in enumerate(doc['materials'])}
    with (ROOT / 'render_materials.bin').open('wb') as f:
        f.write(b'PFRMAT01' + struct.pack('<I', count))
        for i, material in enumerate(doc['materials']):
            name = material['name']
            base, detail = records[i][0], records[i][3]
            ground = name in ('tyreswall.001', 'box1.001')
            tiled = name in ('box1_multimap.001', 'tile1.001', 'tile1B.001')
            if tiled:
                base = detail
            if name == 'tyreswall.001':
                detail = records[names['grass.001']][3]
            elif name == 'box1.001':
                detail = records[names['asphalt.001']][0]
            normal, strength = records[i][2], records[i][8]
            if ground:
                source = names['grass.001' if name == 'tyreswall.001' else 'asphalt.001']
                normal, strength = records[source][2], records[source][8]
            flags = int(ground) | (int(tiled) << 1) | (int(material.get('doubleSided', False)) << 2)
            f.write(struct.pack('<IIIIf', base, detail, normal, flags, strength))


def cook_car():
    doc, raw = glb(ROOT / 'car.glb')
    with (ROOT / 'car_render.bin').open('wb') as f:
        f.write(b'PFCRND01' + struct.pack('<I', len(doc['meshes'])))
        for mesh in doc['meshes']:
            m = doc['materials'][mesh['primitives'][0]['material']]
            strength = m.get('extensions', {}).get('KHR_materials_emissive_strength', {}).get('emissiveStrength', 1)
            emissive = [v * strength for v in m.get('emissiveFactor', [0, 0, 0])]
            flags = int(m.get('name') == 'Body_Paint') | (int(m.get('doubleSided', False)) << 1)
            f.write(struct.pack('<I3f', flags, *emissive))
    before = after = 0
    with (ROOT / 'car_lod.bin').open('wb') as f:
        f.write(b'PFCLOD01' + struct.pack('<I', len(doc['meshes'])))
        for mesh in doc['meshes']:
            p = mesh['primitives'][0]
            assert p.get('mode', 4) == 4
            positions = accessor(doc, raw, p['attributes']['POSITION'])
            normals = accessor(doc, raw, p['attributes']['NORMAL'])
            uv = accessor(doc, raw, p['attributes']['TEXCOORD_0']) if 'TEXCOORD_0' in p['attributes'] else [(0, 0)] * len(positions)
            indices = [v[0] for v in accessor(doc, raw, p['indices'])] if 'indices' in p else list(range(len(positions)))
            # Keep UV and normal seams while collapsing geometry below distant pixel size.
            cells, remap, representatives = {}, [], []
            for i, (position, normal, texcoord) in enumerate(zip(positions, normals, uv)):
                key = tuple(round(v / 0.08) for v in position) + tuple(round(v * 16) for v in texcoord) + tuple(round(v * 2) for v in normal)
                if key not in cells:
                    cells[key] = len(representatives)
                    representatives.append(i)
                remap.append(cells[key])
            kept = []
            for i in range(0, len(indices), 3):
                triangle = [remap[v] for v in indices[i:i + 3]]
                if len(set(triangle)) == 3:
                    kept.extend(representatives[v] for v in triangle)
            before += len(indices) // 3
            after += len(kept) // 3
            f.write(struct.pack('<I', len(kept)))
            for values in (positions, uv, normals):
                width = len(values[0])
                for index in kept:
                    f.write(struct.pack('<' + 'f' * width, *values[index]))
    print(f'Car LOD: {before:,} -> {after:,} triangles; render bindings cooked')


if __name__ == '__main__':
    cook_track()
    if "--track" not in sys.argv:
        cook_car()
