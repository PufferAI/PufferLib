"""
Robot Arm Fixed - Ocean Environment for PufferLib
6-DOF robotic arm environment following PufferLib ocean patterns.
"""
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
        
        if pick_and_place_mode:
            obs_shape = (19,)
            obs_doc = "19D: robot_state + target_object_info + target_basket_info + target_type_onehot"
        else:
            obs_shape = (14,)
            obs_doc = "14D: joint_angles + target_relative_info + distance + progress"
            
        self.single_observation_space = gymnasium.spaces.Box(
            low=-5.0,
            high=5.0,
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
            except Exception as e:  # noqa: BLE001
                print(f"[wandb] init failed: {e}")
                self._wandb = None
        
        super().__init__(buf)
        
        self.actions = self.actions.astype(np.float32)
        c_envs = []
        forward_kwargs = dict(kwargs)
        forward_kwargs.pop('reach_only', None)
        forward_kwargs.pop('max_steps', None)
        forward_kwargs.pop('pick_and_place_mode', None)

        for env_num in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[env_num:(env_num+1)],
                self.actions[env_num:(env_num+1)],
                self.rewards[env_num:(env_num+1)],
                self.terminals[env_num:(env_num+1)],
                self.truncations[env_num:(env_num+1)],
                env_num,
                max_steps=max_steps,
                pick_and_place_mode=self.pick_and_place_mode,
                reach_only=kwargs.get('reach_only', not self.pick_and_place_mode),
                **forward_kwargs,
            ))
        
        self.c_envs = binding.vectorize(*c_envs)
    
    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed if seed is not None else 0)
        return self.observations, []
    
    def step(self, actions):
        self.actions[:] = actions
        
        self.tick += 1
        binding.vec_step(self.c_envs)
        
        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                n = float(log_data.get('n', 0.0) or 0.0)
                avg_ret = float(log_data.get('episode_return', 0.0) or 0.0)
                avg_len = float(log_data.get('episode_length', 0.0) or 0.0)
                total_ret = avg_ret * n
                total_len = avg_len * n
                self._reward_cum += total_ret
                self._episodes_cum += n
                self._steps_cum += total_len

                # Provide stable gauges
                log_data['environment/reward_total_cumulative'] = self._reward_cum
                log_data['environment/episodes_cumulative'] = self._episodes_cum
                log_data['environment/steps_cumulative'] = self._steps_cum
                log_data['environment/avg_reward_per_step'] = (avg_ret / max(1.0, avg_len)) if avg_len else 0.0

                info.append(log_data)
                if self._wandb:
                    try:
                        self._wandb.log(log_data)
                    except Exception as e:  # noqa: BLE001
                        print(f"[wandb] log failed: {e}")
        
        truncations = np.zeros_like(self.terminals, dtype=bool)
        
        return (
            self.observations, 
            self.rewards, 
            self.terminals.astype(bool), 
            truncations,
            info
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