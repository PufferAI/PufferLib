"""Compile MJCF models with MuJoCo into the binaries loaded by physics.h.

    python ocean/mujoco/mjcf2bin.py                  # every resources/mujoco/*.xml
    python ocean/mujoco/mjcf2bin.py path/to/model.xml [out.bin]

Each <model>.bin is a header (magic, version, nq, nv, nbody, njnt, ngeom, nsite,
nu) followed by the compiled mjModel arrays in the order read by mj_loadModel,
as int32/float32. Compiling with MuJoCo keeps inertias, invweights and bounding
radii bit-identical to the real thing. Requires pip install mujoco.
"""
import glob
import os
import struct
import sys

import mujoco
import numpy as np


def compile_model(xml, out):
    m = mujoco.MjModel.from_xml_path(xml)
    assert m.neq == 0, "equality constraints unsupported"
    assert all(m.actuator_trntype == 0) and all(m.actuator_dyntype == 0), "motors only"
    assert all(m.jnt_type[m.actuator_trnid[:, 0]] >= 2), "actuators on hinge/slide joints only"
    assert m.opt.cone == 0, "pyramidal cones only"
    assert m.opt.integrator <= 1, "Euler/RK4 integrators only"
    assert all(np.isin(m.geom_type, [0, 2, 3])), "plane/sphere/capsule geoms only"
    assert all(m.jnt_type[m.jnt_limited.astype(bool) | (m.jnt_stiffness != 0)] >= 2), \
        "limits and springs on hinge/slide joints only"
    # fixed tendons without limits, springs or dampers (humanoid) exert no force
    assert not m.tendon_limited.any() and not m.tendon_stiffness.any() \
        and not m.tendon_damping.any(), "tendon limits/springs/dampers unsupported"
    i32, f32 = np.int32, np.float32
    fields = [
        (m.opt.timestep, f32), (m.opt.gravity, f32), (m.opt.integrator, i32),
        (m.opt.impratio, f32),
        (m.body_parentid, i32), (m.body_rootid, i32), (m.body_weldid, i32),
        (m.body_jntadr, i32), (m.body_jntnum, i32), (m.body_dofadr, i32), (m.body_dofnum, i32),
        (m.body_pos, f32), (m.body_quat, f32), (m.body_ipos, f32), (m.body_iquat, f32),
        (m.body_mass, f32), (m.body_subtreemass, f32), (m.body_inertia, f32),
        (m.body_invweight0, f32),
        (m.jnt_type, i32), (m.jnt_qposadr, i32), (m.jnt_dofadr, i32), (m.jnt_limited, i32),
        (m.jnt_axis, f32), (m.jnt_pos, f32), (m.jnt_range, f32), (m.jnt_margin, f32),
        (m.jnt_stiffness, f32), (m.jnt_solref, f32), (m.jnt_solimp, f32),
        (m.dof_bodyid, i32), (m.dof_jntid, i32), (m.dof_parentid, i32), (m.dof_armature, f32),
        (m.dof_damping, f32), (m.dof_invweight0, f32),
        (m.geom_type, i32), (m.geom_bodyid, i32), (m.geom_contype, i32),
        (m.geom_conaffinity, i32), (m.geom_condim, i32), (m.geom_priority, i32),
        (m.geom_solmix, f32), (m.geom_size, f32), (m.geom_pos, f32), (m.geom_quat, f32),
        (m.geom_friction, f32), (m.geom_solref, f32), (m.geom_solimp, f32),
        (m.geom_margin, f32), (m.geom_gap, f32), (m.geom_rbound, f32),
        (m.site_bodyid, i32), (m.site_pos, f32), (m.site_quat, f32),
        (m.actuator_trnid[:, 0], i32), (m.actuator_gear[:, 0], f32),
        (m.actuator_ctrllimited, i32), (m.actuator_ctrlrange, f32),
        (m.qpos0, f32), (m.qpos_spring, f32),
    ]
    with open(out, "wb") as f:
        f.write(struct.pack("9i", 0x4E424A4D, 1, m.nq, m.nv, m.nbody, m.njnt, m.ngeom, m.nsite,
            m.nu))
        for value, dtype in fields:
            f.write(np.ascontiguousarray(value, dtype=dtype).tobytes())
    print("wrote %s: nq %d nv %d nbody %d njnt %d ngeom %d nsite %d nu %d" % (out, m.nq, m.nv,
        m.nbody, m.njnt, m.ngeom, m.nsite, m.nu))


def main():
    if len(sys.argv) > 1:
        xml = sys.argv[1]
        compile_model(xml, sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(xml)[0] + ".bin")
        return
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for xml in sorted(glob.glob(os.path.join(root, "resources", "mujoco", "*.xml"))):
        compile_model(xml, os.path.splitext(xml)[0] + ".bin")


if __name__ == "__main__":
    main()
