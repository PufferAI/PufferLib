 
import os
import numpy as np
import gymnasium
import pufferlib

try:
    from . import binding
except ImportError:
    import binding

class RobotArm(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        render_mode=None,
        report_interval=1,
        buf=None,
        max_steps=5000,
        pick_and_place_mode=False,
        **kwargs
    ):

        self.pick_and_place_mode = pick_and_place_mode
        obs_shape = (19,)
        obs_doc = "19D: robot_state + target_object_info + target_basket_info + target_type_onehot"

        self.single_observation_space = gymnasium.spaces.Box(
            low=-2.0,
            high=2.0,
            shape=obs_shape,
            dtype=np.float32,
        )
        
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(7,),
            dtype=np.float32
        )
        
        self.num_agents = num_envs
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0
        self.max_steps = max_steps
        self._wandb = None
        self._use_wandb = bool(kwargs.pop('wandb', False) or os.environ.get('PUFFER_WANDB'))
        self._reward_cum = 0.0
        self._episodes_cum = 0.0
        self._steps_cum = 0.0
        if self._use_wandb:
            try:
                import wandb
                self._wandb = wandb.init(
                    project=kwargs.pop('wandb_project', 'puffer_robotarm'),
                    name=kwargs.pop('wandb_run_name', None),
                    config={
                        'num_envs': num_envs,
                        'max_steps': max_steps,
                        'pick_and_place_mode': pick_and_place_mode,
                        **kwargs,
                    },
                    reinit=True,
                )
            except Exception as e:
                print(f"[wandb] init failed: {e}")
                self._wandb = None
        
        super().__init__(buf)
        
        self._terminals_bool = np.empty_like(self.terminals, dtype=bool)
        self._truncations = np.zeros_like(self.terminals, dtype=bool)
        c_envs = []
        forward_kwargs = dict(kwargs)
        forward_kwargs.pop('max_steps', None)
        forward_kwargs.pop('pick_and_place_mode', None)

        defaults = {
            'frame_skip': 2,
            'success_distance': 0.06,
            'domain_randomization': 0,
            'obs_noise_std': 0.0,
            'actuation_noise_std': 0.0,
            'action_smoothing_alpha': 0.4,
            'accel_limit': 15.0,
            'damping': 0.05,
            'action_penalty_coef': 0.0,
            'reward_scale': 1.0,
            'headless': 0,
            'render_decimation': 1,
            'render_target_fps': 60,
            'vsync': 1,
            'curriculum_episodes': 200,
            'success_distance_start': 0.12,
            'success_distance_min': 0.03,
            'on_gripper_spawn_start': 0.05,
            'on_gripper_spawn_min': 0.005,
            'on_gripper_spawn_prob': 0.05,
            'start_grasp_prob': 0.20,
            'touch_bonus_max': 0.0,
            'touch_decay_steps': 0,
            'assist_enabled': 0,
            'assist_episodes': 200,
            'early_reach_episodes': 200,
            'early_reach_bonus': 0.5,
        }
        for k, v in defaults.items():
            forward_kwargs.setdefault(k, v)

        def _to_int_boolish(v):
            if isinstance(v, bool):
                return int(v)
            try:
                import numpy as _np
                if isinstance(v, (_np.bool_,)):
                    return int(bool(v))
            except Exception:
                pass
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ('1', 'true', 'yes', 'y', 'on'):
                    return 1
                if s in ('0', 'false', 'no', 'n', 'off', ''):
                    return 0
                try:
                    return int(float(s))
                except Exception:
                    return 0
            return int(bool(v))

        def _to_float(v):
            if isinstance(v, bool):
                return 1.0 if v else 0.0
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ('1', 'true', 'yes', 'y', 'on'):
                    return 1.0
                if s in ('0', 'false', 'no', 'n', 'off', ''):
                    return 0.0
                try:
                    return float(s)
                except Exception:
                    return 0.0
            try:
                return float(v)
            except Exception:
                return 0.0

        forward_kwargs['pick_and_place_mode'] = 1

        int_keys = [
            'frame_skip', 'domain_randomization', 'curriculum_episodes',
            'touch_decay_steps', 'assist_enabled', 'assist_episodes', 'early_reach_episodes',
            'headless', 'render_decimation', 'render_target_fps', 'vsync'
        ]
        float_keys = [
            'success_distance', 'obs_noise_std', 'actuation_noise_std',
            'action_smoothing_alpha', 'accel_limit', 'damping',
            'success_distance_start', 'success_distance_min',
            'on_gripper_spawn_start', 'on_gripper_spawn_min',
            'on_gripper_spawn_prob', 'start_grasp_prob', 'touch_bonus_max', 'early_reach_bonus'
        ]
        for k in int_keys:
            if k in forward_kwargs:
                forward_kwargs[k] = _to_int_boolish(forward_kwargs[k])
        for k in float_keys:
            if k in forward_kwargs:
                forward_kwargs[k] = _to_float(forward_kwargs[k])

        if 'headless' not in forward_kwargs:
            forward_kwargs['headless'] = 1 if self.render_mode is None else 0

        for env_num in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[env_num:(env_num+1)],
                self.actions[env_num:(env_num+1)],
                self.rewards[env_num:(env_num+1)],
                self.terminals[env_num:(env_num+1)],
                self.truncations[env_num:(env_num+1)],
                env_num,
                max_steps=max_steps,
                **forward_kwargs,
            ))
        
        self.c_envs = binding.vectorize(*c_envs)
    
    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed is not None else 0)
        if self.terminals is not None:
            self.terminals.fill(0)
        if self.rewards is not None:
            self.rewards.fill(0.0)
        return self.observations, []
    
    def step(self, actions):
        self.actions[:] = actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.report_interval == 0:
            raw = binding.vec_log(self.c_envs)
            if raw:
                n = float(raw.get('n', 0.0) or 0.0)
                avg_ret = float(raw.get('episode_return', 0.0) or 0.0)
                avg_len = float(raw.get('episode_length', 0.0) or 0.0)
                pick_rate_ep = float(raw.get('pick_success_rate', 0.0) or 0.0)
                place_rate_ep = float(raw.get('place_success_rate', 0.0) or 0.0)
                picks_total = pick_rate_ep * n
                places_total = place_rate_ep * n
                avg_ep = float(self._reward_cum / self._episodes_cum) if self._episodes_cum else 0.0

                total_ret = avg_ret * n
                total_len = avg_len * n
                self._reward_cum += total_ret
                self._episodes_cum += n
                self._steps_cum += total_len

                simple = {
                    'avg_reward_per_ep':avg_ep,
                    'avg_ep_return': avg_ret,
                    'avg_ep_length': avg_len,
                    'picks_total': picks_total,
                    'places_total': places_total,
                    'episodes_cum': self._episodes_cum,
                    'steps_cum': self._steps_cum,
                    'reward_total_cum': self._reward_cum,
                }

                info.append(simple)
                if self._wandb:
                    try:
                        self._wandb.log(simple)
                    except Exception as e:
                        print(f"[wandb] log failed: {e}")
        
        np.not_equal(self.terminals, 0, out=self._terminals_bool)
        
        return (
            self.observations,
            self.rewards,
            self._terminals_bool,
            self._truncations,
            info,
        )
    
    def render(self, env_id=0):
        binding.vec_render(self.c_envs, env_id)
    
    def close(self):
        binding.vec_close(self.c_envs)
        if self._wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


def make_robotarm(num_envs=16, render_mode=None, max_steps=5000, pick_and_place_mode=False, **kwargs):
    return RobotArm(
        num_envs=num_envs,
        render_mode=render_mode,
        max_steps=max_steps,
        pick_and_place_mode=pick_and_place_mode,
        **kwargs
    )

def make_robotarm_pick_and_place(num_envs=16, render_mode=None, max_steps=5000, **kwargs):
    kwargs.pop('pick_and_place_mode', None)
    
    return RobotArm(
        num_envs=num_envs,
        render_mode=render_mode,
        max_steps=max_steps,
        pick_and_place_mode=True,
        **kwargs
    )

def make_robotarm_train(num_envs=16, max_steps=5000, **kwargs):
    """High-throughput training: headless by default, no rendering overhead."""
    kwargs.setdefault('headless', 1)
    return RobotArm(
        num_envs=num_envs,
        render_mode=None,
        max_steps=max_steps,
        pick_and_place_mode=True,
        **kwargs,
    )

def make_robotarm_eval(num_envs=1, max_steps=5000, **kwargs):
    """Evaluation with rendering enabled and sensible defaults."""
    kwargs.setdefault('headless', 0)
    kwargs.setdefault('render_decimation', 1)
    kwargs.setdefault('render_target_fps', 60)
    kwargs.setdefault('vsync', 1)
    return RobotArm(
        num_envs=num_envs,
        render_mode='human',
        max_steps=max_steps,
        pick_and_place_mode=True,
        **kwargs,
    )