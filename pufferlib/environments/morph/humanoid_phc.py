import os
import sys
from enum import Enum
from types import SimpleNamespace

from isaacgym import gymapi

try:
    import gymtorch
except ImportError:
    from isaacgym import gymtorch

from gym import spaces
import torch
import numpy as np

from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES

from pufferlib.environments.morph.poselib_skeleton import SkeletonTree
from pufferlib.environments.morph.motion_lib import MotionLibSMPL, FixHeightMode
from pufferlib.environments.morph.torch_utils import (
    to_torch,
    torch_rand_float,
    exp_map_to_quat,
    calc_heading_quat,
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_mul,
    quat_conjugate,
    quat_to_tan_norm,
    quat_to_angle_axis,
)


class StateInit(Enum):
    Default = 0
    Start = 1
    Random = 2
    Hybrid = 3


class IsaacGymBase:
    def __init__(
        self,
        sim_params,  # NOTE: This is ignored for now
        physics_engine,
        device_type,
        device_id,  # Allow multi-gpu setting
        headless,
        sim_timestep=1.0 / 60.0,
        control_freq_inv=2,
    ):
        assert physics_engine == gymapi.SIM_PHYSX, "Only PhysX is supported"
        assert device_type in ["cpu", "cuda"], "Device type must be cpu or cuda"

        if device_type == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available"
        self.device = "cuda" + ":" + str(device_id) if device_type == "cuda" else "cpu"
        compute_device = -1 if "cuda" not in self.device else device_id
        graphics_device = -1 if headless else compute_device

        # Sim params: keep these hardcoded here for now
        sim_params = gymapi.SimParams()

        sim_params.dt = sim_timestep
        self.control_freq_inv = control_freq_inv
        self.dt = self.control_freq_inv * sim_params.dt

        sim_params.use_gpu_pipeline = "cuda" in self.device
        sim_params.num_client_threads = 0

        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.default_buffer_size_multiplier = 10.0

        sim_params.physx.use_gpu = "cuda" in self.device
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        sim_params.physx.num_subscenes = 0

        # Set gravity based on up axis and return axis index
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        # Create sim and viewer
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        assert self.sim is not None, "Failed to create sim"
        self.sim_params = sim_params

        self.enable_viewer_sync = True
        self.viewer = None

        if not headless:  # Set up a minimal viewer
            # Subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # Set the camera position (Z axis up)
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def reset(self):
        pass

    def step(self, actions):
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)

        self.gym.fetch_results(self.sim, True)

    def render(self):
        if not self.viewer:
            return

        # Check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # Check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # Step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


class HumanoidPHC:
    def __init__(
        self,
        cfg,
        sim_params=None,
        physics_engine=gymapi.SIM_PHYSX,
        device_type="cuda",
        device_id=0,  # Allow multi-gpu setting
        headless=True,
    ):
        # NOTE: Calling without sim_params should work fine for now
        self.isaac_base = IsaacGymBase(sim_params, physics_engine, device_type, device_id, headless)

        self.device = self.isaac_base.device
        self.gym = self.isaac_base.gym
        self.sim = self.isaac_base.sim
        self.viewer = self.isaac_base.viewer

        self.sim_params = self.isaac_base.sim_params
        self.control_freq_inv = self.isaac_base.control_freq_inv
        self.dt = self.isaac_base.dt

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        ##########################
        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.all_env_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_file = cfg["env"]["motion_file"]  # Must be provided

        ### Robot
        self._config_robot()  #  All robot configs should be here
        # NOTE: PHC does not use force sensors.
        self._create_force_sensors(sensor_joint_names=[])  # No sensor joints

        ### Env
        self._config_env()  # All env configs should be here
        self._create_ground_plane()
        # TODO: Testing putting the robots in the same env
        self._create_envs()
        self.gym.prepare_sim(self.sim)

        self._define_gym_spaces()
        self._setup_gym_tensors()
        self._setup_env_buffers()

        ### Flags
        # NOTE: These are to replace flags.
        self.flag_test = False
        self.flag_im_eval = False
        self.flag_debug = self.device == "cpu"  # CHECK ME

        ### Motion data
        # NOTE: self.flag_im_eval is used in _load_motion
        self._load_motion(self.motion_file)

        # TODO: Remove these. Used by rl-games
        self.has_task = False

    def reset(self, env_ids=None):
        safe_reset = (env_ids is None) or len(env_ids) == self.num_envs
        if env_ids is None:
            env_ids = self.all_env_ids

        self._reset_envs(env_ids)

        # ZL: This way it will simulate one step, then get reset again, squashing any remaining wiredness. Temporary fix
        if safe_reset:
            self.gym.simulate(self.sim)
            self._reset_envs(env_ids)
            torch.cuda.empty_cache()

        return self.obs_buf

    def step(self, actions):
        ### Apply actions, which was self.pre_physics_step(actions)
        if self.reduce_action:
            # NOTE: not using it now. We don't have to create a new tensor every time?
            actions_full = torch.zeros([actions.shape[0], self.num_dof]).to(self.device)
            actions_full[:, self.reduced_action_idx] = actions
            pd_tar = self._action_to_pd_targets(actions_full)

        else:
            pd_tar = self._action_to_pd_targets(actions)
            if self._freeze_hand:
                pd_tar[
                    :,
                    self._dof_names.index("L_Hand") * 3 : (self._dof_names.index("L_Hand") * 3 + 3),
                ] = 0
                pd_tar[
                    :,
                    self._dof_names.index("R_Hand") * 3 : (self._dof_names.index("R_Hand") * 3 + 3),
                ] = 0
            if self._freeze_toe:
                pd_tar[:, self._dof_names.index("L_Toe") * 3 : (self._dof_names.index("L_Toe") * 3 + 3)] = 0
                pd_tar[:, self._dof_names.index("R_Toe") * 3 : (self._dof_names.index("R_Toe") * 3 + 3)] = 0

        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

        ### self._physics_step()
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)

        self.gym.fetch_results(self.sim, True)

        ### Compute observations, rewards, resets, which was self.post_physics_step()
        # This is after stepping, so progress buffer got + 1. Compute reset/reward do not need to forward 1 timestep since they are for "this" frame now.
        self.progress_buf += 1

        self._refresh_sim_tensors()

        self._compute_reward()

        # NOTE: Which envs must be reset is computed here, but the envs get reset outside the env
        self._compute_reset()

        # TODO: Move the code for resetting the env here?

        self._compute_observations()  # observation for the next step.

        self.extras["terminate"] = self._terminate_buf.clone()
        self.extras["reward_raw"] = self.reward_raw.detach()

        if self.use_amp_obs:
            self._update_hist_amp_obs()  # One step for the amp obs
            self._compute_amp_observations()
            self.extras["amp_obs"] = self.amp_obs  ## ZL: hooks for adding amp_obs for training

        if self.flag_im_eval:
            motion_times = (
                (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset
            )  # already has time + 1, so don't need to + 1 to get the target for "this frame"
            motion_res = self._get_state_from_motionlib_cache(
                self._sampled_motion_ids, motion_times, self._global_offset
            )  # pass in the env_ids such that the motion is in synced.
            body_pos = self._rigid_body_pos
            self.extras["mpjpe"] = (body_pos - motion_res["rg_pos"]).norm(dim=-1).mean(dim=-1)
            self.extras["body_pos"] = body_pos.cpu().numpy()
            self.extras["body_pos_gt"] = motion_res["rg_pos"].cpu().numpy()

        # obs, reward, done, info
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self):
        if self.viewer:
            self.isaac_base.render()

    def close(self):
        self.isaac_base.close()

    #####################################################################
    ### __init__()
    #####################################################################

    def _config_robot(self):
        # Currently only supporting SMPL neutral humanoids
        # The PHC code can load/create differently-shaped SMPL humanoids, or unitree ones.
        self.humanoid_type = "smpl"

        ### Load from config
        robot_conf = self.cfg.get("robot", {})

        # For SMPL PHC, the below are different from the default
        self._has_self_collision = robot_conf.get("has_self_collision", False)  # is True
        self._has_upright_start = True
        self._has_dof_subset = True
        self._has_mesh = False

        # The below configs have the default value #####
        # NOTE: These are used in the obs/reward compuation. Revisit later.
        self._has_shape_obs = False  # cfg.robot.get("has_shape_obs", False)
        self._has_shape_obs_disc = False  # cfg.robot.get("has_shape_obs_disc", False)
        self._has_limb_weight_obs = False  # cfg.robot.get("has_weight_obs", False)
        self._has_limb_weight_obs_disc = False  # cfg.robot.get("has_weight_obs_disc", False)

        # NOTE: To customize SMPL, see below links
        # https://github.com/ZhengyiLuo/PHC/blob/master/phc/env/tasks/humanoid.py#L270
        # https://github.com/ZhengyiLuo/PHC/blob/master/phc/env/tasks/humanoid.py#L782

        # reduce_action, _freeze_hand, _freeze_toe are used in self.pre_physics_step()
        self.reduce_action = robot_conf.get("reduce_action", False)
        self._freeze_hand = robot_conf.get("freeze_hand", True)
        self._freeze_toe = robot_conf.get("freeze_toe", True)
        self.reduced_action_idx = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 42, 43, 44, 47, 48, 49, 50, 57, 58, 59, 62, 63, 64, 65]  # fmt: skip

        # See self._build_pd_action_offset_scale()
        self._bias_offset = robot_conf.get("bias_offset", False)
        self._has_smpl_pd_offset = robot_conf.get("has_smpl_pd_offset", False)

        ### Define body, joints, dof
        self._body_names = SMPL_MUJOCO_NAMES

        # Following UHC as hand and toes does not have realiable data.
        self.remove_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe"]

        self.joint_groups = [
            ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
            ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
            ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
            ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
            ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
        ]
        self.limb_weight_group = [
            [self._body_names.index(joint_name) for joint_name in joint_group] for joint_group in self.joint_groups
        ]

        disc_idxes = []
        self._dof_names = self._body_names[1:]
        for idx, name in enumerate(self._dof_names):
            if name not in self.remove_names:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))

        self.dof_subset = (
            torch.from_numpy(np.concatenate(disc_idxes)) if len(disc_idxes) > 0 else torch.tensor([]).long()
        )
        self.left_indexes = [idx for idx, name in enumerate(self._dof_names) if name.startswith("L")]
        self.left_lower_indexes = [
            idx
            for idx, name in enumerate(self._dof_names)
            if name.startswith("L") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
        ]
        self.right_indexes = [idx for idx, name in enumerate(self._dof_names) if name.startswith("R")]
        self.right_lower_indexes = [
            idx
            for idx, name in enumerate(self._dof_names)
            if name.startswith("R") and name[2:] in ["Hip", "Knee", "Ankle", "Toe"]
        ]

        ### Load the Neutral SMPL humanoid asset only
        self.gender_beta = np.zeros(17)  # NOTE: gender (1) + betas (16)

        # And we use the same humanoid shapes for all the agents.
        self.humanoid_shapes = torch.tensor(np.array([self.gender_beta] * self.num_envs)).float().to(self.device)

        # NOTE: The below SMPL assets must be present.
        asset_file_real = "resources/morph/smpl_humanoid.xml"
        assert os.path.exists(asset_file_real)

        sk_tree = SkeletonTree.from_mjcf(asset_file_real)
        self.skeleton_trees = [sk_tree] * self.num_envs

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self.humanoid_asset = self.gym.load_asset(self.sim, ".", asset_file_real, asset_options)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(self.humanoid_asset)

        self._dof_offsets = np.linspace(0, self.num_dof, self.num_bodies).astype(int)

        assert self.num_bodies == len(
            self._body_names
        ), "Number of bodies in asset file does not match number of SMPL bodies"
        assert self.num_dof == len(self._dof_names) * 3, "Number of DOF in asset file does not match number of SMPL DOF"

        # Check if the body ids are consistent between humanoid_asset and self._body_names (SMPL_MUJOCO_NAMES)
        for body_id, body_name in enumerate(self._body_names):
            body_id_asset = self.gym.find_asset_rigid_body_index(self.humanoid_asset, body_name)
            assert (
                body_id == body_id_asset
            ), f"Body id {body_id} does not match index {body_id_asset} for body {body_name}"

    def _build_body_ids_tensor(self, body_names):
        body_ids = [self._body_names.index(name) for name in body_names]
        return to_torch(body_ids, device=self.device, dtype=torch.long)

    def _create_force_sensors(self, sensor_joint_names):
        sensor_pose = gymapi.Transform()
        for jt in sensor_joint_names:
            joint_idx = self.gym.find_asset_rigid_body_index(self.humanoid_asset, jt)
            self.gym.create_asset_force_sensor(self.humanoid_asset, joint_idx, sensor_pose)

    def _config_env(self):
        env_config = self.cfg["env"]

        ### Overall env-related
        self.max_episode_length = env_config.get("episode_length", 300)
        self._enable_early_termination = True
        termination_distance = 0.25  # env_config.get("terminationDistance", 0.5)
        self._termination_distances = to_torch(np.array([termination_distance] * self.num_bodies), device=self.device)
        # NOTE: _termination_distances is changed between train/eval, so keep a backup
        self._termination_distances_backup = self._termination_distances.clone()

        self.env_spacing = env_config.get("env_spacing", 5)
        # NOTE: Related to inter-group collision. If False, there is no inter-env collision. See self._build_env()
        self._divide_group = env_config.get("divide_group", False)

        self.collect_dataset = False  # for offline RL

        ### Obs/action-related
        self._local_root_obs = True
        self._root_height_obs = True
        self.num_states = 0  # cfg["env"].get("numStates", 0)  # Not used for PHC

        self.key_bodies = env_config.get("key_bodies", ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"])
        self._key_body_ids = self._build_body_ids_tensor(self.key_bodies)

        contact_bodies = env_config.get("contact_bodies", ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"])
        self._contact_body_ids = self._build_body_ids_tensor(contact_bodies)

        self._full_track_bodies = self._body_names.copy()
        self._track_bodies = env_config.get("trackBodies", self._full_track_bodies)
        self._track_bodies_id = self._build_body_ids_tensor(self._track_bodies)

        self._reset_bodies = env_config.get("reset_bodies", self._track_bodies)
        self._reset_bodies_id = self._build_body_ids_tensor(self._reset_bodies)
        # NOTE: reset_bodies_id is changed between train/eval, so keep a backup
        self._reset_bodies_id_backup = self._reset_bodies_id

        # Used in https://github.com/kywch/PHC/blob/pixi/phc/learning/im_amp.py#L181. Check how it is used.
        self._eval_bodies = self._body_names.copy()
        for name in self.remove_names:
            self._eval_bodies.remove(name)
        self._eval_track_bodies_id = self._build_body_ids_tensor(self._eval_bodies)

        self.add_obs_noise = False
        self.add_action_noise = False
        self.action_noise_std = 0.05

        ### Control-related
        self.control_mode = "isaac_pd"
        self._kp_scale = env_config.get("kp_scale", 1.0)
        self._kd_scale = env_config.get("kd_scale", self._kp_scale)
        self._res_action = env_config.get("res_action", False)

        ### Motion/AMP-related
        self.seq_motions = False
        self._min_motion_len = 5  # env_config.get("min_length", -1)

        # NOTE: Some AMASS motion is over 7000 frames, and it substantially
        # slows down the evaluation. So we limit the max length to 600.
        self._max_motion_len = 600

        self._state_init = StateInit["Random"]
        self._hybrid_init_prob = 0.5

        self.use_amp_obs = env_config.get("use_amp_obs", False)
        self._num_amp_obs_steps = 10
        self._amp_root_height_obs = True

        # NOTE: Used in amp_agent (rl-games), used to resample motions for training.
        # TODO: Remove shape_resampling_interval (this is for rl-games)
        self.motion_resampling_interval = self.shape_resampling_interval = 500

        # NOTE: Auto PMCP updates the motion sampling prob during training
        # See IMAmpAgent.update_training_data() in the eval function
        self.auto_pmcp = False
        self.auto_pmcp_soft = True

        ### Reward-related
        self.use_power_reward = True
        self.power_coefficient = env_config.get("rew_power_coef", 0.0005)

        # NOTE: body pos reward, body rot reward, body vel reward, body ang vel reward
        self._imitation_reward_dim = 4
        self.reward_specs = env_config.get(
            "reward_specs",
            {
                "k_pos": 100,
                "k_rot": 10,
                "k_vel": 0.1,
                "k_ang_vel": 0.3,
                "w_pos": 0.5,
                "w_rot": 0.25,
                "w_vel": 0.1,
                "w_ang_vel": 0.15,
            },
        )

        # NOTE: if _full_body_reward is false, reward is computed based only on _track_bodies_id
        # See self._compute_reward()
        self._full_body_reward = True

        # ### TODO: Remove these
        # self.getup_schedule = False  # training for getting up after falling -- not used in PHC
        # self.obs_v = 6
        # self.amp_obs_v = 1
        # self.self_obs_v = 1
        # self.zero_out_far = False
        # # self.zero_out_far_train = True
        # self.cycle_motion = False

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
        plane_params.static_friction = 1.0  # self.cfg["env"]["plane"]["staticFriction"]
        plane_params.dynamic_friction = 1.0  # self.cfg["env"]["plane"]["dynamicFriction"]
        plane_params.restitution = 0.0  # self.cfg["env"]["plane"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        self.envs = []
        self.env_origins = []
        self.humanoid_handles = []
        self.humanoid_masses = []
        self.humanoid_limb_and_weights = []
        max_agg_bodies, max_agg_shapes = 160, 160

        lower = gymapi.Vec3(-self.env_spacing, -self.env_spacing, 0.0)
        upper = gymapi.Vec3(self.env_spacing, self.env_spacing, self.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        # Since the same humanoid is used for all the envs ...
        dof_prop = self.gym.get_asset_dof_properties(self.humanoid_asset)
        assert self.control_mode == "isaac_pd"
        dof_prop["driveMode"] = gymapi.DOF_MODE_POS
        dof_prop["stiffness"] *= self._kp_scale
        dof_prop["damping"] *= self._kd_scale
        dof_prop["stiffness"] = 1000
        dof_prop["damping"] = 200

        # NOTE: (from Joseph) You get a small perf boost (~4%) by putting all the actors in the same env
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # NOTE: Different humanoid asset files can be provided to _build_env() for each env
            self._build_single_env(i, env_ptr, self.humanoid_asset, dof_prop)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

            # Save the env origins for the camera work (render_env)
            row = i // num_per_row
            col = i % num_per_row
            self.env_origins.append((col * 2 * self.env_spacing, row * 2 * self.env_spacing, 0.0))

        # NOTE: self.humanoid_limb_and_weights comes from self._build_env()
        self.humanoid_limb_and_weights = torch.stack(self.humanoid_limb_and_weights).to(self.device)

        # These should be all the same because we use the same humanoid for all agents
        print("Humanoid Weights", self.humanoid_masses[:10])

        ### Define dof limits
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        for j in range(self.num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            elif dof_prop["lower"][j] == dof_prop["upper"][j]:
                print("Warning: DOF limits are the same")
                if dof_prop["lower"][j] == 0:
                    self.dof_limits_lower.append(-np.pi)
                    self.dof_limits_upper.append(np.pi)
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.dof_limits = torch.stack([self.dof_limits_lower, self.dof_limits_upper], dim=-1)
        self.torque_limits = to_torch(dof_prop["effort"], device=self.device)

        self._build_pd_action_offset_scale()

    # NOTE: HumanoidRenderEnv overrides this method to add marker actors
    def _build_single_env(self, env_id, env_ptr, humanoid_asset, dof_prop):
        # Collision settings: probably affect speed a lot
        if self._divide_group:
            col_group = self._group_ids[env_id]
        else:
            col_group = env_id  # no inter-environment collision
        col_filter = 0 if self._has_self_collision else 1

        assert self.sim_params.up_axis == gymapi.UP_AXIS_Z
        pos = torch.tensor((0, 0, 0.89)).to(self.device)  # NOTE: char_h (0.89) hard coded
        pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(
            1
        )  # ZL: segfault if we do not randomize the position

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # NOTE: Domain randomization code was here. Search for self.cfg.domain_rand.has_domain_rand in the original repos.

        humanoid_handle = self.gym.create_actor(
            env_ptr, humanoid_asset, start_pose, f"humanoid_{env_id}", col_group, col_filter, 0
        )
        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        mass_ind = [prop.mass for prop in self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_handle)]
        humanoid_mass = np.sum(mass_ind)
        self.humanoid_masses.append(humanoid_mass)

        curr_skeleton_tree = self.skeleton_trees[env_id]
        limb_lengths = torch.norm(curr_skeleton_tree.local_translation, dim=-1)
        limb_lengths = [limb_lengths[group].sum() for group in self.limb_weight_group]
        masses = torch.tensor(mass_ind)
        masses = [masses[group].sum() for group in self.limb_weight_group]
        humanoid_limb_weight = torch.tensor(limb_lengths + masses)
        self.humanoid_limb_and_weights.append(humanoid_limb_weight)  # ZL: attach limb lengths and full body weight.

        self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        if self._has_self_collision:
            if self._has_mesh:
                filter_ints = [0, 1, 224, 512, 384, 1, 1792, 64, 1056, 4096, 6, 6168, 0, 2048, 0, 20, 0, 0, 0, 0, 10, 0, 0, 0]  # fmt: skip
            else:
                filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # fmt: skip

            props = self.gym.get_actor_rigid_shape_properties(env_ptr, humanoid_handle)
            assert len(filter_ints) == len(props)

            for p_idx in range(len(props)):
                props[p_idx].filter = filter_ints[p_idx]
            self.gym.set_actor_rigid_shape_properties(env_ptr, humanoid_handle, props)

        self.humanoid_handles.append(humanoid_handle)

    def _build_pd_action_offset_scale(self):
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        num_joints = len(self._dof_offsets) - 1
        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]
            if not self._bias_offset:
                if dof_size == 3:
                    curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                    curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                    curr_low = np.max(np.abs(curr_low))
                    curr_high = np.max(np.abs(curr_high))
                    curr_scale = max([curr_low, curr_high])
                    curr_scale = 1.2 * curr_scale
                    curr_scale = min([curr_scale, np.pi])

                    lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
                    lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

                    # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                    # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

                elif dof_size == 1:
                    curr_low = lim_low[dof_offset]
                    curr_high = lim_high[dof_offset]
                    curr_mid = 0.5 * (curr_high + curr_low)

                    # extend the action range to be a bit beyond the joint limits so that the motors
                    # don't lose their strength as they approach the joint limits
                    curr_scale = 0.7 * (curr_high - curr_low)
                    curr_low = curr_mid - curr_scale
                    curr_high = curr_mid + curr_scale

                    lim_low[dof_offset] = curr_low
                    lim_high[dof_offset] = curr_high
            else:
                curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
                curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset : (dof_offset + dof_size)] = curr_low
                lim_high[dof_offset : (dof_offset + dof_size)] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        self._L_knee_dof_idx = self._dof_names.index("L_Knee") * 3 + 1
        self._R_knee_dof_idx = self._dof_names.index("R_Knee") * 3 + 1

        # ZL: Modified SMPL to give stronger knee
        self._pd_action_scale[self._L_knee_dof_idx] = 5
        self._pd_action_scale[self._R_knee_dof_idx] = 5

        if self._has_smpl_pd_offset:
            if self._has_upright_start:
                self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = -np.pi / 2
                self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = np.pi / 2
            else:
                self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3] = -np.pi / 6
                self._pd_action_offset[self._dof_names.index("L_Shoulder") * 3 + 2] = -np.pi / 2
                self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3] = -np.pi / 3
                self._pd_action_offset[self._dof_names.index("R_Shoulder") * 3 + 2] = np.pi / 2

    def _define_gym_spaces(self):
        ### Observations
        # Self obs: height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
        self._num_self_obs = 1 + self.num_bodies * (3 + 6 + 3 + 3) - 3

        # Task obs: what goes into this? Check compute obs
        self._task_obs_size = len(self._track_bodies) * self.num_bodies

        self.num_obs = self._num_self_obs + self._task_obs_size  # = 934
        assert self.num_obs == 934

        # AMP obs
        self._dof_obs_size = len(self._dof_names) * 6
        self._num_amp_obs_per_step = (
            13 + self._dof_obs_size + self.num_dof + 3 * len(self.key_bodies)
        )  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]

        if self._has_dof_subset:
            self._num_amp_obs_per_step -= (6 + 3) * int((self.num_dof - len(self.dof_subset)) / 3)

        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step

        ### Actions
        if self.reduce_action:
            self.num_actions = len(self.reduced_action_idx)
        else:
            self.num_actions = self.num_dof

        ### Gym/puffer spaces
        self.single_observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf, dtype=np.float32
        )
        self.amp_observation_space = spaces.Box(
            np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf, dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0, dtype=np.float32
        )

    # TODO: Remove this. Used by rl-games
    def get_running_mean_size(self):
        return (self.num_obs,)

    # TODO: Remove this. Used by rl-games
    def get_task_obs_size(self):
        return self._task_obs_size

    # TODO: Remove this. Used by rl-games
    def get_task_obs_size_detail(self):
        return {
            "target": self._task_obs_size,
            "track_bodies": self._track_bodies,
        }

    def _setup_gym_tensors(self):
        ### get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)  # Keep this as reference
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self._refresh_sim_tensors()

        # NOTE: refresh_force_sensor_tensor, refresh_dof_force_tensor were not here. Any change in learning?
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self._root_states.shape[0] // self.num_envs

        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[
            ..., 0, :
        ]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0
        # NOTE: 13 comes from pos 3 + rot 4 + vel 3 + ang vel 3.
        # root_states[:, 7:13] = 0 means zeroing vel and ang vel.

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., : self.num_dof, 1]

        # NOTE: These are used in self._reset_default(), along with self._initial_humanoid_root_states
        # CHECK ME: Is it ok to use zeros for _initial_dof_pos and _initial_dof_vel?
        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = self._rigid_body_state_reshaped[..., : self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., : self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., : self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., : self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., : self.num_bodies, :]

    def _setup_env_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        # self.self_obs_buf = torch.zeros((self.num_envs, self._num_self_obs), device=self.device, dtype=torch.float)

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # NOTE: store indiviaul reward components. 4 and 5 are hardcoded for now.
        self.reward_raw = torch.zeros(
            (self.num_envs, self._imitation_reward_dim + 1 if self.use_power_reward else self._imitation_reward_dim)
        ).to(self.device)

        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.short)

        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)  # This is dones
        # _terminate_buf records early termination
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        self.extras = {}  # Stores info

        # NOTE: states not used here, but keeping it for now
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)

        # NOTE: related to domain randomization. Not used here.
        # self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # AMP/Motion-related
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self._state_reset_happened = False

        self._global_offset = torch.zeros([self.num_envs, 3]).to(self.device)  # pos offset, so dim is 3
        self._motion_start_times = torch.zeros(self.num_envs).to(self.device)
        self._motion_start_times_offset = torch.zeros(self.num_envs).to(self.device)

        self._motion_sample_start_idx = 0
        self._sampled_motion_ids = torch.arange(self.num_envs).to(self.device)
        self.ref_motion_cache = {}

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        # amp_obs_demo_buf is fed into the discriminator training as the real motion data
        # This replaces the demo replay buffer in the original PHC code
        # amp_batch_size is fixed to the number of envs
        self._amp_obs_demo_buf = torch.zeros_like(self._amp_obs_buf)

        # NOTE: These don't seem to be used, except ref_dof_pos when self._res_action is True
        # self.ref_body_pos = torch.zeros_like(self._rigid_body_pos)
        # self.ref_body_vel = torch.zeros_like(self._rigid_body_vel)
        # self.ref_body_rot = torch.zeros_like(self._rigid_body_rot)
        # self.ref_body_pos_subset = torch.zeros_like(self._rigid_body_pos[:, self._track_bodies_id])
        self.ref_dof_pos = torch.zeros_like(self._dof_pos)

    def _load_motion(self, motion_train_file, motion_test_file=None):
        motion_lib_cfg = SimpleNamespace(
            motion_file=motion_train_file,
            device=self.device,
            fix_height=FixHeightMode.full_fix,
            min_length=self._min_motion_len,
            # NOTE: this max_length determines the training time, so using 300 for now
            # TODO: find a way to evaluate full motion, probably not during training
            max_length=self.max_episode_length,
            im_eval=self.flag_im_eval,
            num_thread=4,
            smpl_type=self.humanoid_type,
            step_dt=self.dt,
            is_deterministic=self.flag_debug,
        )
        self._motion_train_lib = MotionLibSMPL(motion_lib_cfg)
        self._motion_lib = self._motion_train_lib

        # TODO: Use motion_test_file for eval?
        motion_lib_cfg.im_eval = True
        self._motion_eval_lib = MotionLibSMPL(motion_lib_cfg)

        # When loading the motions the first time, use even sampling
        interval = self.num_unique_motions / (self.num_envs + 50)  # 50 is arbitrary
        sample_idxes = np.arange(0, self.num_unique_motions, interval)
        sample_idxes = np.floor(sample_idxes).astype(int)[:self.num_envs]
        sample_idxes = torch.from_numpy(sample_idxes).to(self.device)

        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            # NOTE: During initial loading, use even sampling
            sample_idxes=sample_idxes,
            # random_sample=(not self.flag_test) and (not self.seq_motions),
            # max_len=-1 if self.flag_test else self.max_episode_length,  # NOTE: this is ignored in motion lib
            # start_idx=self._motion_sample_start_idx,
        )

    #####################################################################
    ### reset()
    #####################################################################

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) > 0:
            self._reset_actors(env_ids)  # this funciton call _set_env_state, and should set all state vectors
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()
            self._compute_observations(env_ids)
            self._state_reset_happened = True

        if self.use_amp_obs:
            self._init_amp_obs(env_ids)

    def _reset_actors(self, env_ids):
        if self._state_init == StateInit.Default:
            self._reset_default(env_ids)
        elif self._state_init == StateInit.Start or self._state_init == StateInit.Random:
            self._reset_ref_state_init(env_ids)
        elif self._state_init == StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            raise ValueError(f"Unsupported state initialization strategy: {str(self._state_init)}")

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids

    def _reset_ref_state_init(self, env_ids):
        (
            motion_ids,
            motion_times,
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            rb_pos,
            rb_rot,
            body_vel,
            body_ang_vel,
        ) = self._sample_ref_state(env_ids)

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rigid_body_pos=rb_pos,
            rigid_body_rot=rb_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times

        self._global_offset[env_ids] = 0  # Reset the global offset when resampling.
        self._motion_start_times[env_ids] = motion_times
        self._motion_start_times_offset[env_ids] = 0  # Reset the motion time offsets
        self._sampled_motion_ids[env_ids] = motion_ids

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

    def _reset_env_tensors(self, env_ids):
        env_ids_int32 = self._humanoid_actor_ids[env_ids]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_pos.contiguous()),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # print("#################### refreshing ####################")
        # print("rb", (self._rigid_body_state_reshaped[None, :] - self._rigid_body_state_reshaped[:, None]).abs().sum())
        # print("contact", (self._contact_forces[None, :] - self._contact_forces[:, None]).abs().sum())
        # print('dof_pos', (self._dof_pos[None, :] - self._dof_pos[:, None]).abs().sum())
        # print("dof_vel", (self._dof_vel[None, :] - self._dof_vel[:, None]).abs().sum())
        # print("root_states", (self._humanoid_root_states[None, :] - self._humanoid_root_states[:, None]).abs().sum())
        # print("#################### refreshing ####################")

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        self._contact_forces[env_ids] = 0

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            raise NotImplementedError("Not tested yet")
            # self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids, self._reset_ref_motion_times)

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)

        time_steps = -self.dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)

        amp_obs_demo = self._get_amp_obs(motion_ids, motion_times)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)

        # amp_obs_demo_buf is fed into the discriminator training as the real motion data
        self._amp_obs_demo_buf[env_ids] = self._amp_obs_buf[env_ids]

    def _get_amp_obs(self, motion_ids, motion_times):
        motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times)
        key_pos = motion_res["rg_pos"][:, self._key_body_ids]
        key_vel = motion_res["body_vel"][:, self._key_body_ids]

        return self._compute_amp_observations_from_state(
            motion_res["root_pos"],
            motion_res["root_rot"],
            motion_res["root_vel"],
            motion_res["root_ang_vel"],
            motion_res["dof_pos"],
            motion_res["dof_vel"],
            key_pos,
            key_vel,
            motion_res["motion_bodies"],
            motion_res["motion_limb_weights"],
            self.dof_subset,
        )

    def _sample_time(self, motion_ids):
        # Motion imitation, no more blending and only sample at certain locations
        return self._motion_lib.sample_time_interval(motion_ids)
        # return self._motion_lib.sample_time(motion_ids)

    def _sample_ref_state(self, env_ids):
        num_envs = env_ids.shape[0]

        if self._state_init == StateInit.Random or self._state_init == StateInit.Hybrid:
            motion_times = self._sample_time(self._sampled_motion_ids[env_ids])
        elif self._state_init == StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            raise ValueError("Unsupported state initialization strategy: {:s}".format(str(self._state_init)))

        if self.flag_test:
            motion_times[:] = 0

        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids]
        )

        return (
            self._sampled_motion_ids[env_ids],
            motion_times,
            motion_res["root_pos"],
            motion_res["root_rot"],
            motion_res["dof_pos"],
            motion_res["root_vel"],
            motion_res["root_ang_vel"],
            motion_res["dof_vel"],
            motion_res["rg_pos"],  # rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )

    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ### Cache the motion + offset
        if (
            offset is None
            or "motion_ids" not in self.ref_motion_cache
            or self.ref_motion_cache["offset"] is None
            or len(self.ref_motion_cache["motion_ids"]) != len(motion_ids)
            or len(self.ref_motion_cache["offset"]) != len(offset)
            or (self.ref_motion_cache["motion_ids"] - motion_ids).abs().sum()
            + (self.ref_motion_cache["motion_times"] - motion_times).abs().sum()
            + (self.ref_motion_cache["offset"] - offset).abs().sum()
            > 0
        ):
            self.ref_motion_cache["motion_ids"] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache["motion_times"] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache["offset"] = offset.clone() if offset is not None else None

        else:
            return self.ref_motion_cache

        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)
        self.ref_motion_cache.update(motion_res)
        return self.ref_motion_cache

    def _set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rigid_body_pos=None,
        rigid_body_rot=None,
        rigid_body_vel=None,
        rigid_body_ang_vel=None,
    ):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        if (rigid_body_pos is not None) and (rigid_body_rot is not None):
            self._rigid_body_pos[env_ids] = rigid_body_pos
            self._rigid_body_rot[env_ids] = rigid_body_rot
            self._rigid_body_vel[env_ids] = rigid_body_vel
            self._rigid_body_ang_vel[env_ids] = rigid_body_ang_vel

            self._reset_rb_pos = self._rigid_body_pos[env_ids].clone()
            self._reset_rb_rot = self._rigid_body_rot[env_ids].clone()
            self._reset_rb_vel = self._rigid_body_vel[env_ids].clone()
            self._reset_rb_ang_vel = self._rigid_body_ang_vel[env_ids].clone()

    #####################################################################
    ### compute observations
    #####################################################################

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = self.all_env_ids

        # This is the normalized state of the humanoid
        state = self._compute_humanoid_obs(env_ids)

        # This is the difference of state with the demo, but it is
        # called "state" in the paper.
        imitation = self._compute_task_obs(env_ids)

        # Possible the original paper only uses imitation
        obs = torch.cat([state, imitation], dim=-1)

        # NOTE: Not using it for now.
        # This is the normalized vector with position, rotation, velocity, and
        # angular velocity for the simulated humanoid and the demo data
        # self.state, self.demo = self._compute_state_obs(env_ids)

        if self.add_obs_noise and not self.flag_test:
            obs = obs + torch.randn_like(obs) * 0.1

        self.obs_buf[env_ids] = obs

        return obs

    def _compute_humanoid_obs(self, env_ids=None):
        with torch.no_grad():
            if env_ids is None:
                body_pos = self._rigid_body_pos
                body_rot = self._rigid_body_rot
                body_vel = self._rigid_body_vel
                body_ang_vel = self._rigid_body_ang_vel
                body_shape_params = self.humanoid_shapes[:, :-6]
                limb_weights = self.humanoid_limb_and_weights

            else:
                body_pos = self._rigid_body_pos[env_ids]
                body_rot = self._rigid_body_rot[env_ids]
                body_vel = self._rigid_body_vel[env_ids]
                body_ang_vel = self._rigid_body_ang_vel[env_ids]
                body_shape_params = self.humanoid_shapes[env_ids, :-6]
                limb_weights = self.humanoid_limb_and_weights[env_ids]

            return compute_humanoid_observations_smpl_max(
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                body_shape_params,
                limb_weights,
                self._local_root_obs,  # Constant: True
                self._root_height_obs,  # Constant: True
                self._has_upright_start,  # Constant: True
                self._has_shape_obs,  # Constant: False
                self._has_limb_weight_obs,  # Constant: False
            )

    # NOTE: This produces "simplified" amp obs, which goes into the discriminator
    def _compute_state_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        body_pos = self._rigid_body_pos[env_ids]  # [..., self._track_bodies_id]
        body_rot = self._rigid_body_rot[env_ids]  # [..., self._track_bodies_id]
        body_vel = self._rigid_body_vel[env_ids]  # [..., self._track_bodies_id]
        body_ang_vel = self._rigid_body_ang_vel[env_ids]  # [..., self._track_bodies_id]

        sim_obs = compute_humanoid_observations_smpl_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            None,
            None,
            self._local_root_obs,  # Constant: True
            self._root_height_obs,  # Constant: True
            self._has_upright_start,  # Constant: True
            self._has_shape_obs,  # Constant: False
            self._has_limb_weight_obs,  # Constant: False
        )

        motion_times = (
            (self.progress_buf[env_ids] + 1) * self.dt
            + self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )  # Next frame, so +1

        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids]
        )  # pass in the env_ids such that the motion is in synced.

        demo_pos = motion_res["rg_pos"]  # [..., self._track_bodies_id]
        demo_rot = motion_res["rb_rot"]  # [..., self._track_bodies_id]
        demo_vel = motion_res["body_vel"]  # [..., self._track_bodies_id]
        demo_ang_vel = motion_res["body_ang_vel"]  # [..., self._track_bodies_id]

        demo_obs = compute_humanoid_observations_smpl_max(
            demo_pos,
            demo_rot,
            demo_vel,
            demo_ang_vel,
            None,
            None,
            True,  # Constant: True
            self._root_height_obs,  # Constant: True
            self._has_upright_start,  # Constant: True
            self._has_shape_obs,  # Constant: False
            self._has_limb_weight_obs,  # Constant: False
        )

        return sim_obs, demo_obs

    def _compute_task_obs(self, env_ids=None, save_buffer=True):
        if env_ids is None:
            env_ids = self.all_env_ids
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        motion_times = (
            (self.progress_buf[env_ids] + 1) * self.dt
            + self._motion_start_times[env_ids]
            + self._motion_start_times_offset[env_ids]
        )  # Next frame, so +1

        motion_res = self._get_state_from_motionlib_cache(
            self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids]
        )  # pass in the env_ids such that the motion is in synced.

        (
            ref_dof_pos,
            ref_rb_pos,
            ref_rb_rot,
            ref_body_vel,
            ref_body_ang_vel,
        ) = (
            motion_res["dof_pos"],
            motion_res["rg_pos"],  # ref_rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
        ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
        ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]

        # TODO: revisit constant args
        time_steps = 1  # Necessary?
        obs = compute_imitation_observations_v6(
            root_pos,
            root_rot,
            body_pos_subset,
            body_rot_subset,
            body_vel_subset,
            body_ang_vel_subset,
            ref_rb_pos_subset,
            ref_rb_rot_subset,
            ref_body_vel_subset,
            ref_body_ang_vel_subset,
            time_steps,  # Constant: 1
            self._has_upright_start,  # Constant: True
        )

        if self._res_action and save_buffer:
            # self.ref_body_pos[env_ids] = ref_rb_pos
            # self.ref_body_vel[env_ids] = ref_body_vel
            # self.ref_body_rot[env_ids] = ref_rb_rot
            # self.ref_body_pos_subset[env_ids] = ref_rb_pos_subset
            self.ref_dof_pos[env_ids] = ref_dof_pos

        return obs

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]

        # assert self.humanoid_type == "smpl"
        if self.dof_subset is None:
            # ZL hack
            (
                self._dof_pos[:, 9:12],
                self._dof_pos[:, 21:24],
                self._dof_pos[:, 51:54],
                self._dof_pos[:, 66:69],
            ) = 0, 0, 0, 0
            (
                self._dof_vel[:, 9:12],
                self._dof_vel[:, 21:24],
                self._dof_vel[:, 51:54],
                self._dof_vel[:, 66:69],
            ) = 0, 0, 0, 0

        if env_ids is None:
            # TODO: revisit constant args
            self._curr_amp_obs_buf[:] = self._compute_amp_observations_from_state(
                self._rigid_body_pos[:, 0, :],
                self._rigid_body_rot[:, 0, :],
                self._rigid_body_vel[:, 0, :],
                self._rigid_body_ang_vel[:, 0, :],
                self._dof_pos,
                self._dof_vel,
                key_body_pos,
                key_body_vel,
                self.humanoid_shapes,
                self.humanoid_limb_and_weights,
                self.dof_subset,
            )
        else:
            if len(env_ids) == 0:
                return

            self._curr_amp_obs_buf[env_ids] = self._compute_amp_observations_from_state(
                self._rigid_body_pos[env_ids][:, 0, :],
                self._rigid_body_rot[env_ids][:, 0, :],
                self._rigid_body_vel[env_ids][:, 0, :],
                self._rigid_body_ang_vel[env_ids][:, 0, :],
                self._dof_pos[env_ids],
                self._dof_vel[env_ids],
                key_body_pos[env_ids],
                key_body_vel[env_ids],
                self.humanoid_shapes[env_ids],
                self.humanoid_limb_and_weights[env_ids],
                self.dof_subset,
            )

    def _compute_amp_observations_from_state(
        self,
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        dof_pos,
        dof_vel,
        key_body_pos,
        key_body_vels,
        smpl_params,
        limb_weight_params,
        dof_subset,
    ):
        smpl_params = smpl_params[:, :-6]

        # TODO: revisit constant args
        return build_amp_observations_smpl(
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_body_pos,
            smpl_params,
            limb_weight_params,
            dof_subset,
            self._local_root_obs,  # Constant: True
            self._amp_root_height_obs,  # Constant: True
            self._has_dof_subset,  # Constant: True
            self._has_shape_obs_disc,  # Constant: False
            self._has_limb_weight_obs_disc,  # Constant: False
            self._has_upright_start,  # Constant: True
        )

    #####################################################################
    ### step() -- pre_physics_step(), post_physics_step()
    #####################################################################

    def _action_to_pd_targets(self, action):
        # NOTE: self._res_action is False by default
        if self._res_action:
            pd_tar = self.ref_dof_pos + self._pd_action_scale * action
            pd_lower = self._dof_pos - np.pi / 2
            pd_upper = self._dof_pos + np.pi / 2
            pd_tar = torch.maximum(torch.minimum(pd_tar, pd_upper), pd_lower)
        else:
            pd_tar = self._pd_action_offset + self._pd_action_scale * action

        return pd_tar

    def _compute_reward(self):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        motion_times = (
            self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
        )  # reward is computed after physics step, and progress_buf is already updated for next time step.

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)

        (
            ref_rb_pos,
            ref_rb_rot,
            ref_body_vel,
            ref_body_ang_vel,
        ) = (
            motion_res["rg_pos"],  # ref_rb_pos
            motion_res["rb_rot"],
            motion_res["body_vel"],
            motion_res["body_ang_vel"],
        )

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        # NOTE: self._full_body_reward is True by default
        if self._full_body_reward:
            self.rew_buf[:], self.reward_raw[:, : self._imitation_reward_dim] = compute_imitation_reward(
                root_pos,
                root_rot,
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                ref_rb_pos,
                ref_rb_rot,
                ref_body_vel,
                ref_body_ang_vel,
                self.reward_specs,
            )

        else:
            body_pos_subset = body_pos[..., self._track_bodies_id, :]
            body_rot_subset = body_rot[..., self._track_bodies_id, :]
            body_vel_subset = body_vel[..., self._track_bodies_id, :]
            body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

            ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
            ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
            ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
            ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]
            self.rew_buf[:], self.reward_raw[:, : self._imitation_reward_dim] = compute_imitation_reward(
                root_pos,
                root_rot,
                body_pos_subset,
                body_rot_subset,
                body_vel_subset,
                body_ang_vel_subset,
                ref_rb_pos_subset,
                ref_rb_rot_subset,
                ref_body_vel_subset,
                ref_body_ang_vel_subset,
                self.reward_specs,
            )

        if self.use_power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1)
            # power_reward = -0.00005 * (power ** 2)
            power_reward = -self.power_coefficient * power
            # First 3 frame power reward should not be counted. since they could be dropped.
            power_reward[self.progress_buf <= 3] = 0

            self.rew_buf[:] += power_reward
            self.reward_raw[:, -1] = power_reward

    def _compute_reset(self):
        time = (
            (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset
        )  # Reset is also called after the progress_buf is updated.
        pass_time = time >= self._motion_lib._motion_lengths

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, time, self._global_offset)

        body_pos = self._rigid_body_pos[..., self._reset_bodies_id, :].clone()
        ref_body_pos = motion_res["rg_pos"][..., self._reset_bodies_id, :].clone()

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_im_reset(
            self.reset_buf,
            self.progress_buf,
            self._contact_forces,
            self._contact_body_ids,
            body_pos,
            ref_body_pos,
            pass_time,
            self._enable_early_termination,
            self._termination_distances[..., self._reset_bodies_id],
            self.flag_im_eval,
        )

    # NOTE: Training/eval code changes the termination distances.
    def set_termination_distances(self, termination_distances):
        self._termination_distances[:] = termination_distances

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            # Got RuntimeError: unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location. Please clone() the tensor before performing the operation.
            try:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0 : (self._num_amp_obs_steps - 1)]
            except:
                self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0 : (self._num_amp_obs_steps - 1)].clone()

        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0 : (self._num_amp_obs_steps - 1)]

    #####################################################################
    ### Motion/AMP
    #####################################################################

    @property
    def amp_obs(self):
        return self._amp_obs_buf.view(-1, self.num_amp_obs) if self.use_amp_obs else None

    def fetch_amp_obs_demo(self):
        return self._amp_obs_demo_buf.view(-1, self.num_amp_obs) if self.use_amp_obs else None

    # def fetch_amp_obs_demo(self, num_samples):
    #     # Creates the reference motion amp obs, for discriminator.

    #     if self._amp_obs_demo_buf is None:
    #         # NOTE: This is called during training init. Buffer size depends on amp_batch_size.
    #         self._amp_obs_demo_buf = torch.zeros(
    #             (num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step),
    #             device=self.device,
    #             dtype=torch.float32,
    #         )
    #     else:
    #         # Buffer size (amp_batch_size) must not change during training
    #         assert self._amp_obs_demo_buf.shape[0] == num_samples

    #     motion_ids = self._motion_lib.sample_motions(num_samples)
    #     motion_times0 = self._sample_time(motion_ids)
    #     amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)
    #     self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)
    #     amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.num_amp_obs)

    #     return amp_obs_demo_flat

    # def build_amp_obs_demo(self, motion_ids, motion_times0):
    #     # Compute observation for the motion starting point
    #     dt = self.dt
    #     motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])

    #     motion_times = motion_times0.unsqueeze(-1)
    #     time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
    #     motion_times = motion_times + time_steps

    #     motion_ids = motion_ids.view(-1)
    #     motion_times = motion_times.view(-1)

    #     amp_obs_demo = self._get_amp_obs(motion_ids, motion_times)
    #     # if self._add_amp_input_noise:
    #     #     amp_obs_demo = amp_obs_demo + torch.randn_like(amp_obs_demo) * 0.01

    #     return amp_obs_demo

    def resample_motions(self):
        if self.flag_test:
            self.forward_motion_samples()

        else:
            self._motion_lib.load_motions(
                skeleton_trees=self.skeleton_trees,
                limb_weights=self.humanoid_limb_and_weights.cpu(),
                gender_betas=self.humanoid_shapes.cpu(),
                random_sample=(not self.flag_test) and (not self.seq_motions),
                # max_len=-1 if self.flag_test else self.max_episode_length,  # NOTE: this is ignored in motion lib
            )

            time = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
            root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids, time)
            self._global_offset[:, :2] = self._humanoid_root_states[:, :2] - root_res["root_pos"][:, :2]
            self.reset()

    def begin_seq_motion_samples(self):
        # For evaluation
        self._motion_sample_start_idx = 0
        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            random_sample=False,
            start_idx=self._motion_sample_start_idx,
        )
        self.reset()

    def forward_motion_samples(self):
        self._motion_sample_start_idx += self.num_envs
        self._motion_lib.load_motions(
            skeleton_trees=self.skeleton_trees,
            gender_betas=self.humanoid_shapes.cpu(),
            limb_weights=self.humanoid_limb_and_weights.cpu(),
            random_sample=False,
            start_idx=self._motion_sample_start_idx,
        )
        self.reset()

    @property
    def num_unique_motions(self):
        return self._motion_lib._num_unique_motions

    @property
    def current_motion_ids(self):
        return self._motion_lib._curr_motion_ids

    @property
    def motion_sample_start_idx(self):
        return self._motion_sample_start_idx

    @property
    def motion_data_keys(self):
        return self._motion_lib._motion_data_keys

    def get_motion_steps(self):
        return self._motion_lib.get_motion_num_steps()

    # TODO: Remove this. Used by rl-games
    @property
    def start_idx(self):
        return self._motion_sample_start_idx

    #####################################################################
    ### Toggle train/eval model. Used in the training/eval code
    #####################################################################
    def toggle_eval_mode(self):
        self.flag_test = True
        self.flag_im_eval = True

        # Relax the early termination condition for evaluation
        self.set_termination_distances(0.5)  # NOTE: hardcoded

        self._motion_lib = self._motion_eval_lib
        self.begin_seq_motion_samples()  # using _motion_eval_lib
        if len(self._reset_bodies_id) > 15:
            # Following UHC. Only do it for full body, not for three point/two point trackings.
            self._reset_bodies_id = self._eval_track_bodies_id

        # Return the number of motions
        return self._motion_lib._num_unique_motions

    def untoggle_eval_mode(self, failed_keys):
        self.flag_test = False
        self.flag_im_eval = False

        self.set_termination_distances(self._termination_distances_backup)
        self._motion_lib = self._motion_train_lib
        self._reset_bodies_id = self._reset_bodies_id_backup

        if self.auto_pmcp:
            self._motion_lib.update_hard_sampling_weight(failed_keys)
        elif self.auto_pmcp_soft:
            self._motion_lib.update_soft_sampling_weight(failed_keys)

        # Return the motion lib termination history
        return self._motion_lib._termination_history.clone()


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))  # SMPL
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))


# @torch.jit.script
def compute_humanoid_observations_smpl_max(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    smpl_params,
    limb_weight_params,
    local_root_obs,
    root_height_obs,
    upright,
    has_smpl_params,
    has_limb_weight_params,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = calc_heading_quat_inv(root_rot)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if not (local_root_obs):
        root_rot_obs = quat_to_tan_norm(root_rot)  # If not local root obs, you override it.
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]

    if has_smpl_params:
        obs_list.append(smpl_params)

    if has_limb_weight_params:
        obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs


@torch.jit.script
def compute_imitation_observations_v6(
    root_pos,
    root_rot,
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    ref_body_pos,
    ref_body_rot,
    ref_body_vel,
    ref_body_ang_vel,
    time_steps,
    upright,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = calc_heading_quat_inv(root_rot)
    heading_rot = calc_heading_quat(root_rot)
    heading_inv_rot_expand = (
        heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    )
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    body_rot[:, None].repeat_interleave(time_steps, 1)
    diff_global_body_rot = quat_mul(
        ref_body_rot.view(B, time_steps, J, 4),
        quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)),
    )
    diff_local_body_rot_flat = quat_mul(
        quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)),
        heading_rot_expand.view(-1, 4),
    )  # Need to be change of basis

    ##### linear and angular  Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))

    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))

    ##### body pos + Dof_pos This part will have proper futures.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(
        B, 1, 1, 3
    )  # preserves the body position
    local_ref_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = quat_to_tan_norm(local_ref_body_rot)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * 24 * 3
    obs.append(quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * 24 * 6

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs


@torch.jit.script
def dof_to_obs_smpl(pose):
    # type: (Tensor) -> Tensor
    joint_obs_size = 6
    B, jts = pose.shape
    num_joints = int(jts / 3)

    joint_dof_obs = quat_to_tan_norm(exp_map_to_quat(pose.reshape(-1, 3))).reshape(B, -1)
    assert (num_joints * joint_obs_size) == joint_dof_obs.shape[1]

    return joint_dof_obs


@torch.jit.script
def build_amp_observations_smpl(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    dof_pos,
    dof_vel,
    key_body_pos,
    shape_params,
    limb_weight_params,
    dof_subset,
    local_root_obs,
    root_height_obs,
    has_dof_subset,
    has_shape_obs_disc,
    has_limb_weight_obs,
    upright,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool, bool) -> Tensor
    B, N = root_pos.shape
    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot_inv, root_rot)
    else:
        root_rot_obs = root_rot

    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot_inv, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot_inv, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2]
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2]
    )
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2]
    )

    if has_dof_subset:
        dof_vel = dof_vel[:, dof_subset]
        dof_pos = dof_pos[:, dof_subset]

    dof_obs = dof_to_obs_smpl(dof_pos)
    obs_list = []
    if root_height_obs:
        obs_list.append(root_h)
    obs_list += [
        root_rot_obs,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel,
        flat_local_key_pos,
    ]
    # 1? + 6 + 3 + 3 + 114 + 57 + 12
    if has_shape_obs_disc:
        obs_list.append(shape_params)
    if has_limb_weight_obs:
        obs_list.append(limb_weight_params)
    obs = torch.cat(obs_list, dim=-1)

    return obs


@torch.jit.script
def compute_imitation_reward(
    root_pos,
    root_rot,
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    ref_body_pos,
    ref_body_rot,
    ref_body_vel,
    ref_body_ang_vel,
    rwd_specs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = (
        rwd_specs["k_pos"],
        rwd_specs["k_rot"],
        rwd_specs["k_vel"],
        rwd_specs["k_ang_vel"],
    )
    w_pos, w_rot, w_vel, w_ang_vel = (
        rwd_specs["w_pos"],
        rwd_specs["w_rot"],
        rwd_specs["w_vel"],
        rwd_specs["w_ang_vel"],
    )

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = quat_mul(ref_body_rot, quat_conjugate(body_rot))
    diff_global_body_angle = quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)

    return reward, reward_raw


@torch.jit.script
def compute_humanoid_im_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    ref_body_pos,
    pass_time,
    enable_early_termination,
    termination_distance,
    use_mean,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    if enable_early_termination:
        # NOTE: When evaluating, using mean is relaxed vs. max is strict.
        if use_mean:
            has_fallen = torch.any(
                torch.norm(rigid_body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance[0],
                dim=-1,
            )  # using average, same as UHC"s termination condition
        else:
            has_fallen = torch.any(
                torch.norm(rigid_body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1
            )  # using max

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1

        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

        # if (contact_buf.abs().sum(dim=-1)[0] > 0).sum() > 2:
        #     np.set_printoptions(precision=4, suppress=1)
        #     print(contact_buf.numpy(), contact_buf.abs().sum(dim=-1)[0].nonzero().squeeze())

    reset = torch.where(pass_time, torch.ones_like(reset_buf), terminated)

    return reset, terminated
