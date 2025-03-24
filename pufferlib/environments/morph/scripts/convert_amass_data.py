import glob
import argparse
import os.path as osp

import isaacgym  # noqa

import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot

import joblib
from tqdm import tqdm

from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES

from pufferlib.environments.morph.poselib_skeleton import SkeletonTree, SkeletonState

# SMPL model files need to be in this directory
BODY_MODEL_DIR = "resources/morph"

# NOTE: hardcoded 66
SELECT_DOF = 22 * 3  # 22 SMPL joints, without fingers x 3. Replace fingers with dummy hands
SMPL_JOINT_NUM = len(SMPL_BONE_ORDER_NAMES)

TARGET_FRAME_RATE = 30
UPRIGHT_START = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="/workspace/dataset/AMASS/DFaust_67")
    # Download the occulusion file (amass_occlusion_v3) from https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc
    # See https://github.com/ZhengyiLuo/PHC/blob/master/download_data.sh
    parser.add_argument("--occulusion_file", type=str, default=None, help="Motions to exlude (by the PHC authors)")
    parser.add_argument("--failed-keys", type=str, default=None, help="Failed motion key file from the training")
    parser.add_argument(
        "--name_offset", type=int, default=-3, help="The offset of the dataset name in the full file path"
    )
    args = parser.parse_args()

    process_split = "train"
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": UPRIGHT_START,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": "smpl",
    }

    smpl_local_robot = SMPL_Robot(robot_cfg, data_dir=BODY_MODEL_DIR)
    if not osp.isdir(args.path):
        print("Please specify AMASS data path")

    if args.failed_keys is not None:
        failed_keys = set(joblib.load(args.failed_keys)["failed_keys"])
    else:
        failed_keys = None

    all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)
    amass_occlusion = joblib.load(args.occulusion_file) if args.occulusion_file else {}
    amass_full_motion_dict = {}
    amass_splits = {
        "valid": ["HumanEva", "MPI_HDM05", "SFU", "MPI_mosh"],
        "test": ["Transitions_mocap", "SSM_synced"],
        "train": [
            "CMU",
            "MPI_Limits",
            "TotalCapture",
            "KIT",
            "EKUT",
            "TCD_handMocap",
            "BMLhandball",
            "DanceDB",
            "ACCAD",
            "BMLmovi",
            "BioMotionLab_NTroje",
            "Eyes_Japan_Dataset",
            "DFaust_67",
        ],
    }
    process_set = amass_splits[process_split]
    length_acc = []

    # The same SMPL neutral model is used for all motions
    beta = np.zeros((16))
    gender_number, gender = [0], "neutral"
    skeleton_tree = SkeletonTree.from_mjcf(BODY_MODEL_DIR + "/smpl_humanoid.xml")

    for data_path in tqdm(all_pkls):
        bound = 0

        splits = data_path.split("/")[args.name_offset :]
        key_name_dump = "0-" + "_".join(splits).replace(".npz", "")

        if splits[0] not in process_set:
            continue

        if key_name_dump in amass_occlusion:
            issue = amass_occlusion[key_name_dump]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
                bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calculated assuming 30 FPS...
                if bound < 10:
                    print("bound too small", key_name_dump, bound)
                    continue
            else:
                print("issue irrecoverable", key_name_dump, issue)
                continue

        if failed_keys is not None:
            # Only process the failed keys
            if key_name_dump not in failed_keys:
                continue

        entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

        if "mocap_framerate" not in entry_data:
            continue
        framerate = entry_data["mocap_framerate"]

        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2

        skip = int(framerate / TARGET_FRAME_RATE)
        root_trans = entry_data["trans"][::skip, :]

        pose_aa = np.concatenate(
            [entry_data["poses"][::skip, :SELECT_DOF], np.zeros((root_trans.shape[0], 6))], axis=-1
        )

        num_frames = pose_aa.shape[0]
        if bound == 0:
            bound = num_frames

        root_trans = root_trans[:bound]
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        pose_aa = pose_aa[:bound]
        num_frames = pose_aa.shape[0]
        if num_frames < 10:
            continue

        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(num_frames, SMPL_JOINT_NUM, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(num_frames, SMPL_JOINT_NUM, 4)

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True,
        )

        if UPRIGHT_START:
            pose_quat_global = (
                (
                    sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
                    * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                )
                .as_quat()
                .reshape(num_frames, -1, 4)
            )

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False
            )
            pose_quat = new_sk_state.local_rotation.numpy()

        pose_quat_global = new_sk_state.global_rotation.numpy()

        # Commenting out unused variables
        new_motion_out = {}
        new_motion_out["root_trans_offset"] = root_trans_offset
        new_motion_out["pose_aa"] = pose_aa
        new_motion_out["pose_quat_global"] = pose_quat_global
        # new_motion_out["pose_quat"] = pose_quat
        # new_motion_out["trans_orig"] = root_trans
        new_motion_out["beta"] = np.zeros((16))
        new_motion_out["gender"] = "neutral"
        new_motion_out["fps"] = TARGET_FRAME_RATE

        amass_full_motion_dict[key_name_dump] = new_motion_out
        # print(f"Processed {key_name_dump}")

    num_motions = len(amass_full_motion_dict)
    print("Processed", num_motions, "motions, saving to", args.path)

    if UPRIGHT_START:
        joblib.dump(amass_full_motion_dict, f"{args.path}/amass_train_{num_motions}_upright.pkl", compress=True)
    else:
        joblib.dump(amass_full_motion_dict, f"{args.path}/amass_train_{num_motions}.pkl", compress=True)
