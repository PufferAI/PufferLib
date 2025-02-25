# BSD 3-Clause Clear License
#
# Copyright (c) 2023 Carnegie Mellon University
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY’S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gc
import glob
import time
import random
import os.path as osp
from enum import Enum

import joblib
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation as sRot

import torch
import torch.multiprocessing as mp

from smpl_sim.smpllib.smpl_parser import SMPL_Parser

from pufferlib.environments.morph.poselib_skeleton import SkeletonMotion, SkeletonState
from pufferlib.environments.morph import torch_utils

USE_CACHE = False
# print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)


class MotionlibMode(Enum):
    file = 1
    directory = 2


class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


# def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
#     # Assume each joint is 3dof
#     diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
#     diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
#     dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

#     return dof_vel[1:, :].flatten()


# def compute_motion_dof_vels(motion):
#     num_frames = motion.tensor.shape[0]
#     dt = 1.0 / motion.fps
#     dof_vels = []

#     for f in range(num_frames - 1):
#         local_rot0 = motion.local_rotation[f]
#         local_rot1 = motion.local_rotation[f + 1]
#         frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
#         dof_vels.append(frame_dof_vel)

#     dof_vels.append(dof_vels[-1])
#     dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

#     return dof_vels


# ~15% faster than compute_motion_dof_vels
@torch.jit.script
def compute_motion_dof_vels_jit(local_rotation, fps):
    # type: (Tensor, int) -> Tensor
    num_frames = local_rotation.shape[0]
    dt = 1.0 / fps
    dof_vels = []
    for f in range(num_frames - 1):
        local_rot0 = local_rotation[f]
        local_rot1 = local_rotation[f + 1]

        # frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        frame_dof_vel = dof_vel[1:, :].flatten()

        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels


class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLibBase:
    def __init__(self, motion_lib_cfg):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1 / getattr(self.m_cfg, "step_dt", 1 / 30)  # CHECK ME: hardcoded
        print("SIM FPS (from MotionLibBase):", self._sim_fps)
        self._device = self.m_cfg.device

        self.mesh_parsers = None

        self.load_data(self.m_cfg.motion_file, min_length=self.m_cfg.min_length, im_eval=self.m_cfg.im_eval)
        self.setup_constants(fix_height=self.m_cfg.fix_height, num_thread=self.m_cfg.num_thread)

    def load_data(self, motion_file, min_length=-1, im_eval=False):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
            assert len(self._motion_data_load) > 0

        data_list = self._motion_data_load

        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {
                    k: v for k, v in list(self._motion_data_load.items()) if len(v["pose_quat_global"]) >= min_length
                }
            elif im_eval:
                data_list = {
                    item[0]: item[1]
                    for item in sorted(
                        self._motion_data_load.items(),
                        key=lambda entry: len(entry[1]["pose_quat_global"]),
                        reverse=True,
                    )
                }
                # data_list = self._motion_data
            else:
                data_list = self._motion_data_load

            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)

        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(
                self._motion_data_load[0]
            )  # set self._motion_data_load to a sample of the data

    def setup_constants(self, fix_height=FixHeightMode.full_fix, num_thread=1):
        self.fix_height = fix_height
        self.num_thread = max(num_thread, 1)

        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = (
            torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
        )  # For use in sampling batches
        self._sampling_batch_prob = None  # For use in sampling within batches

    @staticmethod
    def load_motion_with_skeleton(
        ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid
    ):
        raise NotImplementedError

    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        raise NotImplementedError

    def load_motions(
        self,
        skeleton_trees,
        gender_betas,
        limb_weights,
        random_sample=True,
        start_idx=0,
        max_len=-1,
        sample_idxes=None,
    ):
        # load motion load the same number of motions as there are skeletons (humanoids)
        if "gts" in self.__dict__:
            del (
                self.gts,
                self.grs,
                self.lrs,
                self.grvs,
                self.gravs,
                self.gavs,
                self.gvs,
                self.dvs,
            )
            del (
                self._motion_lengths,
                self._motion_fps,
                self._motion_dt,
                self._motion_num_frames,
                self._motion_bodies,
                self._motion_aa,
            )

        motions = []
        self._motion_lengths = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_bodies = []
        self._motion_aa = []

        torch.cuda.empty_cache()
        gc.collect()

        total_len = 0.0
        self.num_joints = len(skeleton_trees[0].node_names)
        num_motion_to_load = len(skeleton_trees)

        # If sample_idxes is provided, use it
        if sample_idxes is None or len(sample_idxes) != num_motion_to_load:
            if not self.m_cfg.is_deterministic and random_sample:
                sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(
                    self._device
                )
            else:
                sample_idxes = torch.remainder(torch.arange(len(skeleton_trees)) + start_idx, self._num_unique_motions).to(
                    self._device
                )

        # import ipdb; ipdb.set_trace()
        self._curr_motion_ids = sample_idxes
        self.one_hot_motions = torch.nn.functional.one_hot(
            self._curr_motion_ids, num_classes=self._num_unique_motions
        ).to(self._device)  # Testing for obs_v5
        self.curr_motion_keys = self._motion_data_keys[sample_idxes]
        self._sampling_batch_prob = (
            self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()
        )

        start_time = time.time()

        print("\n****************************** Current motion keys ******************************")
        print("Sampling motion:", sample_idxes[:30])
        if len(self.curr_motion_keys) < 100:
            print(self.curr_motion_keys)
        else:
            print(self.curr_motion_keys[:30], ".....")
        print("*********************************************************************************\n")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        torch.set_num_threads(1)
        mp.set_sharing_strategy("file_descriptor")

        manager = mp.Manager()
        queue = manager.Queue()

        num_jobs = self.num_thread

        res_acc = {}  # using dictionary ensures order of the results.
        jobs = motion_data_list
        chunk = np.ceil(len(jobs) / num_jobs).astype(int)
        ids = np.arange(len(jobs))

        jobs = [
            (
                ids[i : i + chunk],
                jobs[i : i + chunk],
                skeleton_trees[i : i + chunk],
                gender_betas[i : i + chunk],
                self.mesh_parsers,
                self.m_cfg,
            )
            for i in range(0, len(jobs), chunk)
        ]
        job_args = [jobs[i] for i in range(len(jobs))]
        for i in range(1, len(jobs)):
            worker_args = (*job_args[i], queue, i)
            worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
            worker.start()
        res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

        for i in range(len(jobs) - 1):
            res = queue.get()
            res_acc.update(res)

        for f in range(len(res_acc)):
            motion_file_data, curr_motion = res_acc[f]
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)

            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                self._motion_aa.append(motion_file_data["pose_aa"].reshape(-1, self.num_joints * 3))
                self._motion_bodies.append(curr_motion.gender_beta)
            else:
                self._motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                self._motion_bodies.append(torch.zeros(17))

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

            del curr_motion

        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(self._motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(self._motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)
        self._motion_limb_weights = torch.tensor(np.array(limb_weights), device=self._device, dtype=torch.float32)
        self._num_motions = len(motions)

        # NOTE: These take ~20s for 4096 envs
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = motion.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        print(f"Time to load motions: {int(time.time() - start_time)}s")

        return motions

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    # def update_sampling_weight(self):
    #     ## sampling weight based on success rate.
    #     # sampling_temp = 0.2
    #     sampling_temp = 0.1
    #     curr_termination_prob = 0.5

    #     curr_succ_rate = 1 - self._termination_history[self._curr_motion_ids] / self._sampling_history[self._curr_motion_ids]
    #     self._success_rate[self._curr_motion_ids] = curr_succ_rate
    #     sample_prob = torch.exp(-self._success_rate / sampling_temp)

    #     self._sampling_prob = sample_prob / sample_prob.sum()
    #     self._termination_history[self._curr_motion_ids] = 0
    #     self._sampling_history[self._curr_motion_ids] = 0

    #     topk_sampled = self._sampling_prob.topk(50)
    #     print("Current most sampled", self._motion_data_keys[topk_sampled.indices.cpu().numpy()])

    def update_hard_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only trained on "failed" sequences. Auto PMCP.
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._sampling_prob[:] = 0
            self._sampling_prob[indexes] = 1 / len(indexes)
            print(
                "############################################################ Auto PMCP ############################################################"
            )
            print(f"Training on only {len(failed_keys)} seqs")
            print(failed_keys)
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = (
                torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
            )  # For use in sampling batches

    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP.
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)

            print(
                "############################################################ Auto PMCP ############################################################"
            )
            print(f"Training mostly on {len(self._sampling_prob.nonzero())} seqs ")
            print(self._motion_data_keys[self._sampling_prob.nonzero()].flatten())
            print(
                "###############################################################################################################################"
            )
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = (
                torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
            )  # For use in sampling batches

    def update_sampling_prob(self, termination_history):
        if len(termination_history) == len(self._termination_history) and termination_history.sum() > 0:
            self._sampling_prob[:] = termination_history / termination_history.sum()
            self._termination_history = termination_history
            return True
        else:
            return False

    # def update_sampling_history(self, env_ids):
    #     self._sampling_history[self._curr_motion_ids[env_ids]] += 1
    #     # print("sampling history: ", self._sampling_history[self._curr_motion_ids])

    # def update_termination_history(self, termination):
    #     self._termination_history[self._curr_motion_ids] += termination
    #     # print("termination history: ", self._termination_history[self._curr_motion_ids])

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob, num_samples=n, replacement=True).to(self._device)

        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        # n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_interval(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
        curr_fps = 1 / 30
        motion_time = ((phase * motion_len) / curr_fps).long() * curr_fps

        return motion_time

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps).ceil().int()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        # n = len(motion_ids)
        # num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [
            local_rot0,
            local_rot1,
            body_vel0,
            body_vel1,
            body_ang_vel0,
            body_ang_vel1,
            rg_pos0,
            rg_pos1,
            dof_vel0,
            dof_vel1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1

        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)

        return {
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "motion_bodies": self._motion_bodies[motion_ids],
            "motion_limb_weights": self._motion_limb_weights[motion_ids],
        }

    def get_root_pos_smpl(self, motion_ids, motion_times):
        # n = len(motion_ids)
        # num_bodies = self._get_num_bodies()

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        # print("non_interval", frame_idx0, frame_idx1)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        vals = [rg_pos0, rg_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        return {"root_pos": rg_pos[..., 0, :].clone()}

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)  # clip blend to be within 0 and 1

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)


class MotionLibSMPL(MotionLibBase):
    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg=motion_lib_cfg)

        data_dir = "resources/morph"
        if osp.exists(data_dir):
            if motion_lib_cfg.smpl_type == "smpl":
                # NOTE: SMPL model files must be present in the data_dir.
                # Download from https://smpl.is.tue.mpg.de/
                smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
                smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
                smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
            else:
                raise NotImplementedError(f"SMPL type {motion_lib_cfg.smpl_type} not implemented")

            self.mesh_parsers = {0: smpl_parser_n, 1: smpl_parser_m, 2: smpl_parser_f}

        else:
            print("SMPL models not found, set mesh_parsers to None")
            self.mesh_parsers = None

    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0

        diff_fix = 0
        with torch.no_grad():
            frame_check = 30
            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[gender.item()]
            height_tolorance = 0.0
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(
                pose_aa[:frame_check], betas[None,], trans[:frame_check]
            )

            offset = (
                joints_curr[:, 0] - trans[:frame_check]
            )  # account for SMPL root offset. since the root trans we pass in has been processed, we have to "add it back".

            if fix_height_mode == FixHeightMode.ankle_fix:
                assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
                pick = (
                    (
                        (
                            (assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int()
                            + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int()
                            + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int()
                            + +(assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()
                        )
                        == 4
                    )
                    .nonzero()
                    .squeeze()
                )
                diff_fix = (
                    (vertices_curr[:, pick] - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values
                    - height_tolorance
                ).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            elif fix_height_mode == FixHeightMode.full_fix:
                diff_fix = (
                    (vertices_curr - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance
                ).min()  # Only acount the first 30 frames, which usually is a calibration phase.

            trans[..., -1] -= diff_fix
            return trans, diff_fix

    @staticmethod
    def load_motion_with_skeleton(
        ids,
        motion_data_list,
        skeleton_trees,
        shape_params,
        mesh_parsers,
        config,
        queue,
        pid,
    ):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length
        fix_height = config.fix_height
        # TODO: make it's own random number generator?
        # np.random.seed(np.random.randint(5000) * pid)
        res = {}
        assert len(ids) == len(motion_data_list)

        if pid == 0 and config.num_thread == 1:
            pbar = tqdm(range(len(motion_data_list)))
        else:
            pbar = range(len(motion_data_list))

        for f in pbar:
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = shape_params[f]

            seq_len = curr_file["root_trans_offset"].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = 0 if config.is_deterministic else random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = curr_file["root_trans_offset"].clone()[start:end]
            # trans = curr_file['root_trans_offset'].copy()[start:end]
            pose_aa = to_torch(curr_file["pose_aa"][start:end])
            pose_quat_global = curr_file["pose_quat_global"][start:end]

            B, J, N = pose_quat_global.shape

            ##### ZL: randomize the heading ######
            if not (config.is_deterministic or config.im_eval):
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
                pose_quat_global = (
                    (random_heading_rot * sRot.from_quat(pose_quat_global.reshape(-1, 4))).as_quat().reshape(B, J, N)
                )
                trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T))
            ##### ZL: randomize the heading ######

            if mesh_parsers is not None:
                trans, trans_fix = MotionLibSMPL.fix_trans_height(
                    pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode=fix_height
                )
            # else:
            #     trans_fix = 0

            pose_quat_global = to_torch(pose_quat_global)
            sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_trees[f], pose_quat_global, trans, is_local=False
            )

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            # curr_dof_vels = compute_motion_dof_vels(curr_motion)
            curr_dof_vels = compute_motion_dof_vels_jit(curr_motion.local_rotation, curr_motion.fps)

            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta
            res[curr_id] = (curr_file, curr_motion)

        if queue is not None:
            queue.put(res)
        else:
            return res
