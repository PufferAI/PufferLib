import os
import sys
import os.path as osp
import datetime
from collections import defaultdict, deque

from isaacgym import gymapi
import gymtorch

import torch
import numpy as np

import joblib
import imageio
import matplotlib
import matplotlib.pyplot as plt

from pufferlib.environments.morph.humanoid_phc import HumanoidPHC, StateInit
from pufferlib.environments.morph.torch_utils import to_torch, exp_map_to_quat


PERTURB_OBJS = [
    ["small", 60],
    # ["large", 60],
]


def agt_color(aidx):
    return matplotlib.colors.to_rgb(plt.rcParams["axes.prop_cycle"].by_key()["color"][aidx % 10])


class HumanoidRenderEnv(HumanoidPHC):
    def __init__(
        self,
        cfg,
        sim_params=None,
        physics_engine=gymapi.SIM_PHYSX,
        device_type="cuda",
        device_id=0,  # Allow multi-gpu setting
        headless=True,
    ):
        ### Flags like
        self.flag_server_mode = False
        self.flag_show_traj = True
        self.flag_add_proj = False

        # Reset the init state to the start of the motion
        cfg["env"]["state_init"] = "Start"

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        # If running with a viewer, set up keyboard shortcuts and camera
        self._create_viewer()

        # NOTE: server_mode refers to using webcam or motion generator to get motions to imitate
        # Skipping this for now
        # if self.flag_server_mode:
        #     # bgsk = threading.Thread(target=self.setup_video_client, daemon=True).start()
        #     bgsk = threading.Thread(target=self.setup_talk_client, daemon=False).start()

        if self.viewer or self.flag_server_mode:
            self._build_marker_state_tensors()
            self._init_camera()
            self.change_char_color()

        self.state_record = defaultdict(list)

    def step(self, actions):
        obs, rewards, dones, infos = super().step(actions)
        self.render()
        return obs, rewards, dones, infos

    def render(self):
        if not self.viewer:
            return

        # check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        if self.viewer or self.flag_server_mode:
            self._update_camera()
            self._update_marker()

        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_video_record" and evt.value > 0:
                self.recording = not self.recording
                self.recording_state_change = True
            elif evt.action == "cancel_video_record" and evt.value > 0:
                self.recording = False
                self.recording_state_change = False
                self._video_queue = deque(maxlen=self.max_video_queue_size)
                self.state_record.clear()
            elif evt.action == "reset" and evt.value > 0:
                self.reset()
            elif evt.action == "print_cam" and evt.value > 0:
                cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
                cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
                print("Print camera", cam_pos)
            elif evt.action == "show_traj" and evt.value > 0:
                self.flag_show_traj = not self.flag_show_traj
                print("show_traj: ", self.flag_show_traj)
            elif evt.action == "show_progress" and evt.value > 0:
                print("Progress ", self.progress_buf)
            elif evt.action == "apply_force" and evt.value > 0:
                forces = torch.zeros(
                    (1, self._rigid_body_state.shape[0], 3),
                    device=self.device,
                    dtype=torch.float,
                )
                torques = torch.zeros(
                    (1, self._rigid_body_state.shape[0], 3),
                    device=self.device,
                    dtype=torch.float,
                )
                # forces[:, 8, :] = -800
                for i in range(self._rigid_body_state.shape[0] // self.num_bodies):
                    forces[:, i * self.num_bodies + 3, :] = -3500
                    forces[:, i * self.num_bodies + 7, :] = -3500
                # torques[:, 1, :] = 500
                self.gym.apply_rigid_body_force_tensors(
                    self.sim,
                    gymtorch.unwrap_tensor(forces),
                    gymtorch.unwrap_tensor(torques),
                    gymapi.ENV_SPACE,
                )
            elif evt.action == "prev_env" and evt.value > 0:
                self.viewing_env_idx = (self.viewing_env_idx - 1) % min(self.num_envs, 10)
                print("\nShowing env: ", self.viewing_env_idx)
            elif evt.action == "next_env" and evt.value > 0:
                self.viewing_env_idx = (self.viewing_env_idx + 1) % min(self.num_envs, 10)
                print("\nShowing env: ", self.viewing_env_idx)
            elif evt.action == "resample_motion" and evt.value > 0:
                self.resample_motions()
            elif evt.action == "change_color" and evt.value > 0:
                self.change_char_color()
                print("Change character color")

        if self.recording_state_change:
            if not self.recording:
                if not self.flag_server_mode:
                    self.writer.close()
                    del self.writer

                self._write_states_to_file(self.curr_states_file_name)
                print(f"============ Video finished writing {self.curr_states_file_name}============")
            else:
                print("============ Writing video ============")
            self.recording_state_change = False

        if self.recording:
            if not self.flag_server_mode:
                self.gym.render_all_camera_sensors(self.sim)
                color_image = self.gym.get_camera_image(
                    self.sim,
                    self.envs[self.viewing_env_idx],
                    self.recorder_camera_handles[self.viewing_env_idx],
                    gymapi.IMAGE_COLOR,
                )
                self.color_image = color_image.reshape(color_image.shape[0], -1, 4)

                if "writer" not in self.__dict__:
                    curr_date_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                    self.curr_video_file_name = self._video_path % curr_date_time
                    self.curr_states_file_name = self._states_path % curr_date_time
                    self.writer = imageio.get_writer(
                        self.curr_video_file_name,
                        fps=int(1 / self.dt),
                        macro_block_size=None,
                    )
                self.writer.append_data(self.color_image)

            self._record_states()

        # step graphics
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

    #####################################################################
    ### __init__()
    #####################################################################

    def _create_viewer(self):
        if self.viewer:
            # subscribe to keyboard shortcuts
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "toggle_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SEMICOLON, "cancel_video_record")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "follow")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "fixed")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "divide_group")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_C, "print_cam")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_M, "disable_collision_reset")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_B, "fixed_path")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_N, "real_path")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "show_traj")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "apply_force")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_LEFT, "prev_env")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_RIGHT, "next_env")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "resample_motion")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "slow_traj")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "trigger_input")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "show_progress")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_O, "change_color")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        ###### Custom Camera Sensors ######
        self.recorder_camera_handles = []
        self.max_num_camera = 10
        self.viewing_env_idx = 0
        for idx, env in enumerate(self.envs):
            self.recorder_camera_handles.append(self.gym.create_camera_sensor(env, gymapi.CameraProperties()))
            if idx > self.max_num_camera:
                break

        self.recorder_camera_handle = self.recorder_camera_handles[0]
        self.recording, self.recording_state_change = False, False
        self.max_video_queue_size = 100000
        self._video_queue = deque(maxlen=self.max_video_queue_size)
        rendering_out = osp.join("output", "renderings")
        states_out = osp.join("output", "states")
        os.makedirs(rendering_out, exist_ok=True)
        os.makedirs(states_out, exist_ok=True)
        self.cfg_name = self.cfg["exp_name"]
        self._video_path = osp.join(rendering_out, f"{self.cfg_name}-%s.mp4")
        self._states_path = osp.join(states_out, f"{self.cfg_name}-%s.pkl")
        # self.gym.draw_env_rigid_contacts(self.viewer, self.envs[1], gymapi.Vec3(0.9, 0.3, 0.3), 1.0, True)

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._init_cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
        self._init_cam_target = gymapi.Vec3(10.0, 15.0, 0.0)

        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, self._init_cam_pos, self._init_cam_target)

    def _create_envs(self):
        if self.viewer or self.flag_server_mode:
            self._marker_handles = [[] for _ in range(self.num_envs)]
            self._load_marker_asset()

        if self.flag_add_proj:
            self._proj_handles = []
            self._load_proj_asset()

        super()._create_envs()

    def _load_marker_asset(self):
        asset_root = str("resources/morph")

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, "traj_marker.urdf", asset_options)
        self._marker_asset_small = self.gym.load_asset(self.sim, asset_root, "traj_marker_small.urdf", asset_options)

    def _load_proj_asset(self):
        asset_root = str("resources/morph")

        small_asset_file = "block_projectile.urdf"
        # small_asset_file = "ball_medium.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 10000000.0
        # small_asset_options.fix_base_link = True
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)

        large_asset_file = "block_projectile_large.urdf"
        large_asset_options = gymapi.AssetOptions()
        large_asset_options.angular_damping = 0.01
        large_asset_options.linear_damping = 0.01
        large_asset_options.max_angular_velocity = 100.0
        large_asset_options.density = 10000000.0
        # large_asset_options.fix_base_link = True
        large_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._large_proj_asset = self.gym.load_asset(self.sim, asset_root, large_asset_file, large_asset_options)

    def _build_single_env(self, env_id, env_ptr, humanoid_asset, dof_prop):
        super()._build_single_env(env_id, env_ptr, humanoid_asset, dof_prop)

        if self.viewer or self.flag_server_mode:
            self._build_marker(env_id, env_ptr)

        if self.flag_add_proj:
            self._build_proj(env_id, env_ptr)

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        for i in range(self.num_bodies):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                1,
                0,
            )

            if i in self._track_bodies_id:
                self.gym.set_rigid_body_color(
                    env_ptr,
                    marker_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(0.8, 0.0, 0.0),
                )
            else:
                self.gym.set_rigid_body_color(
                    env_ptr,
                    marker_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(1.0, 1.0, 1.0),
                )
            self._marker_handles[env_id].append(marker_handle)

    def _build_proj(self, env_id, env_ptr):
        pos = [
            [-0.01, 0.3, 0.4],
            # [ 0.0890016, -0.40830246, 0.25]
        ]
        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = pos[i][0]
            default_pose.p.y = pos[i][1]
            default_pose.p.z = pos[i][2]
            obj_type = obj[0]
            if obj_type == "small":
                proj_asset = self._small_proj_asset
            elif obj_type == "large":
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), env_id, 2)
            self._proj_handles.append(proj_handle)

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[
            ..., 1 : (1 + self.num_bodies), :
        ]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rotation = self._marker_states[..., 3:7]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

    #####################################################################
    ### render()
    #####################################################################

    def change_char_color(self):
        colors = []
        offset = np.random.randint(0, 10)
        for env_id in range(self.num_envs):
            rand_cols = agt_color(env_id + offset)
            colors.append(rand_cols)

        self.sample_char_color(torch.tensor(colors), torch.arange(self.num_envs))

    def sample_char_color(self, cols, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(cols[env_id, 0], cols[env_id, 1], cols[env_id, 2]),
                )

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        env_idx = self.viewing_env_idx

        if env_idx == 0:
            new_cam_pos = self._init_cam_pos
            new_cam_target = self._init_cam_target

        else:
            char_root_pos = np.array(self.env_origins[env_idx])
            new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
            new_cam_pos = gymapi.Vec3(char_root_pos[0], char_root_pos[1] - 4.0, 2.0)

        self.gym.set_camera_location(self.recorder_camera_handle, self.envs[env_idx], new_cam_pos, new_cam_target)

        if self.viewer:
            self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

    def _update_marker(self):
        self._marker_pos[:] = 1000

        if self.flag_show_traj:
            motion_times = (
                (self.progress_buf + 1) * self.dt + self._motion_start_times + self._motion_start_times_offset
            )  # + 1 for target.
            motion_res = self._get_state_from_motionlib_cache(
                self._sampled_motion_ids, motion_times, self._global_offset
            )
            ref_rb_pos = motion_res["rg_pos"]  # yes, the rg_pos is rb_pos

            # self._marker_pos[:] = ref_rb_pos
            self._marker_pos[..., self._track_bodies_id, :] = ref_rb_pos[..., self._track_bodies_id, :]

            # self._marker_rotation[..., self._track_bodies_id, :] = ref_rb_rot[..., self._track_bodies_id, :]

        # ######### Heading debug #######
        # points = self.init_root_points()
        # base_quat = self._rigid_body_rot[0, 0:1]
        # base_quat = remove_base_rot(base_quat)
        # heading_rot = calc_heading_quat(base_quat)
        # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
        # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
        # ######### Heading debug #######

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )
        return

    # NOTE: Used in "Heading debug" above
    def init_root_points(self):
        # For debugging purpose
        y = torch.tensor(np.linspace(-0.5, 0.5, 5), device=self.device, requires_grad=False)
        x = torch.tensor(np.linspace(0, 1, 5), device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_root_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_root_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _record_states(self):
        self.state_record["ref_body_pos_subset"].append(self.ref_body_pos_subset.cpu().clone())
        self.state_record["ref_body_pos_full"].append(self.ref_body_pos.cpu().clone())
        # self.state_record['ref_dof_pos'].append(self.ref_dof_pos.cpu().clone())

    def _write_states_to_file(self, file_name):
        self.state_record["skeleton_trees"] = self.skeleton_trees
        self.state_record["humanoid_betas"] = self.humanoid_shapes
        print(f"Dumping states into {file_name}")

        progress = torch.stack(self.state_record["progress"], dim=1)
        progress_diff = torch.cat([progress, -10 * torch.ones(progress.shape[0], 1).to(progress)], dim=-1)

        diff = torch.abs(progress_diff[:, :-1] - progress_diff[:, 1:])
        split_idx = torch.nonzero(diff > 1)
        split_idx[:, 1] += 1
        data_to_dump = {
            k: torch.stack(v)
            for k, v in self.state_record.items()
            if k not in ["skeleton_trees", "humanoid_betas", "progress"]
        }
        fps = 60
        motion_dict_dump = {}
        num_for_this_humanoid = 0
        curr_humanoid_index = 0

        for idx in range(len(split_idx)):
            split_info = split_idx[idx]
            humanoid_index = split_info[0]

            if humanoid_index != curr_humanoid_index:
                num_for_this_humanoid = 0
                curr_humanoid_index = humanoid_index

            if num_for_this_humanoid == 0:
                start = 0
            else:
                start = split_idx[idx - 1][-1]

            end = split_idx[idx][-1]

            dof_pos_seg = data_to_dump["dof_pos"][start:end, humanoid_index]
            B, H = dof_pos_seg.shape
            root_states_seg = data_to_dump["root_states"][start:end, humanoid_index]

            body_quat = torch.cat(
                [
                    root_states_seg[:, None, 3:7],
                    exp_map_to_quat(dof_pos_seg.reshape(B, -1, 3)),
                ],
                dim=1,
            )
            motion_dump = {
                "skeleton_tree": self.state_record["skeleton_trees"][humanoid_index].to_dict(),
                "body_quat": body_quat,
                "trans": root_states_seg[:, :3],
                "root_states_seg": root_states_seg,
                "dof_pos": dof_pos_seg,
            }

            motion_dump["fps"] = fps
            motion_dump["betas"] = self.humanoid_shapes[humanoid_index].detach().cpu().numpy()
            motion_dump.update(
                {
                    k: v[start:end, humanoid_index]
                    for k, v in data_to_dump.items()
                    if k
                    not in [
                        "dof_pos",
                        "root_states",
                        "skeleton_trees",
                        "humanoid_betas",
                        "progress",
                    ]
                }
            )
            motion_dict_dump[f"{humanoid_index}_{num_for_this_humanoid}"] = motion_dump
            num_for_this_humanoid += 1
        joblib.dump(motion_dict_dump, file_name)
        self.state_record.clear()
