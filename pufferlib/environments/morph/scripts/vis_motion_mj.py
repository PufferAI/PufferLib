import time
import argparse

import isaacgym  # noqa

import torch
import numpy as np

import joblib
import mujoco
import mujoco.viewer

from pufferlib.environments.morph.poselib_skeleton import SkeletonTree, SkeletonState
from pufferlib.environments.morph.torch_utils import quat_to_exp_map

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--motion-file", type=str, default="amass_train_129_upright.pkl", help="Path to motion file"
    )
    parser.add_argument("-i", "--motion-idx", type=int, default=0, help="Index of the motion to play")
    args = parser.parse_known_args()[0]

    motion_data = joblib.load(args.motion_file)
    keys = list(motion_data.keys())

    motion = motion_data[keys[args.motion_idx]]
    dt = 1.0 / motion["fps"]

    sk_tree = SkeletonTree.from_mjcf(SMPL_XML)
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, torch.from_numpy(motion["pose_quat_global"]), motion["root_trans_offset"], is_local=False
    )

    mj_model = mujoco.MjModel.from_xml_path(SMPL_XML)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    frame_idx = 0
    num_frames = motion["root_trans_offset"].shape[0]

    z_offset = sk_state.global_translation[0, :, 2].min()
    global_translation = sk_state.global_translation.clone()
    global_translation[:, :, 2] -= z_offset

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for _ in range(len(sk_tree._node_indices)):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.01, np.array([1, 0, 0, 1]))

        while viewer.is_running():
            root_pos = global_translation[frame_idx]
            root_rot = sk_state.global_rotation[frame_idx][0]  # joint 0 is root
            local_rot = sk_state.local_rotation[frame_idx][1:]
            dof_pos = quat_to_exp_map(local_rot).flatten()

            mj_data.qpos[:3] = root_pos[0]
            mj_data.qpos[3:7] = root_rot[[3, 0, 1, 2]]  # xyzw -> wxyz
            mj_data.qpos[7:] = dof_pos

            mujoco.mj_forward(mj_model, mj_data)

            # Display the red dots for keypoints
            for i in range(root_pos.shape[0]):
                viewer.user_scn.geoms[i].pos = root_pos[i]

            viewer.sync()

            frame_idx = (frame_idx + 1) % num_frames
            time.sleep(dt)
