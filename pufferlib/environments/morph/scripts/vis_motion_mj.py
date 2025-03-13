import time
import argparse
from types import SimpleNamespace

import isaacgym  # noqa

import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot

import mujoco
import mujoco.viewer

from pufferlib.environments.morph.poselib_skeleton import SkeletonTree
from pufferlib.environments.morph.motion_lib import MotionLibSMPL, FixHeightMode

SMPL_XML = "resources/morph/smpl_humanoid.xml"


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


def key_call_back(keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    else:
        print("not mapped", chr(keycode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--motion-file", type=str, default="amass_train_129_upright.pkl", help="Path to motion file"
    )
    args = parser.parse_known_args()[0]

    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1 / 30, False
    motion_lib_cfg = SimpleNamespace(
        motion_file=args.motion_file,
        device=torch.device("cpu"),
        fix_height=FixHeightMode.full_fix,
        min_length=-1,
        max_length=-1,
        im_eval=False,
        smpl_type="smpl",
        step_dt=dt,
        num_thread=1,
        is_deterministic=True,
    )

    sk_tree = SkeletonTree.from_mjcf(SMPL_XML)
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    motion_lib.load_motions(
        skeleton_trees=[sk_tree] * num_motions,
        gender_betas=[torch.zeros(17)] * num_motions,
        limb_weights=[np.zeros(10)] * num_motions,
        random_sample=False,
        start_idx=curr_start,
    )

    mj_model = mujoco.MjModel.from_xml_path(SMPL_XML)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(len(sk_tree._node_indices)):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.01, np.array([1, 0, 0, 1]))

        while viewer.is_running():
            step_start = time.time()
            motion_len = motion_lib.get_motion_length(motion_id).item()
            motion_time = time_step % motion_len
            motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]), torch.tensor([motion_time]))

            (
                root_pos,
                root_rot,
                dof_pos,
                rb_pos,
            ) = (
                motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["dof_pos"],
                motion_res["rg_pos"],
            )

            mj_data.qpos[:3] = root_pos[0].cpu().numpy()
            mj_data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]
            mj_data.qpos[7:] = sRot.from_rotvec(dof_pos[0].cpu().numpy().reshape(-1, 3)).as_euler("XYZ").flatten()

            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            for i in range(rb_pos.shape[1]):
                viewer.user_scn.geoms[i].pos = rb_pos[0, i]

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
