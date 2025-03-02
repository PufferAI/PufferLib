import os
import gc
import ast
import sys
import uuid
import time
import math
import argparse
import configparser

import joblib
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

import isaacgym  # noqa

import torch
import numpy as np

from smpl_sim.smpllib.smpl_eval import compute_metrics_lite

import pufferlib
import pufferlib.cleanrl
import pufferlib.vector

from pufferlib.environments.morph.environment import make as env_creator
import pufferlib.environments.morph.policy as policy_module

import clean_pufferl


class EvalStats:
    def __init__(self, vec_env, failed_save_path=None):
        self.task_env = vec_env.env
        self.num_envs = self.task_env.num_envs
        device = self.task_env.device
        self.failed_save_path = failed_save_path

        # Prep the env for evaluation
        self.num_unique_motions = self.task_env.toggle_eval_mode()

        self.terminate_state = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.terminate_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_steps = 0
        self.success_rate = 0
        self.failed_keys = []
        self.results = None

        self.pbar = tqdm(range(self.num_unique_motions // self.num_envs))
        self.pbar.set_description("")

    def post_step_eval(self):
        motion_num_steps = self.task_env.get_motion_steps()

        # Eval-related info is stored in the extras
        info = self.task_env.extras

        # If terminate after the last frame, then it is not a termination. curr_step is one step behind simulation.
        termination_state = torch.logical_and(self.curr_steps <= motion_num_steps - 1, info["terminate"])
        self.terminate_state = torch.logical_or(termination_state, self.terminate_state, out=self.terminate_state)
        if (~self.terminate_state).sum() > 0:
            # NOTE: This is to handle when there are more envs than the motions
            max_possible_id = self.num_unique_motions - 1
            curr_ids = self.task_env.current_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = motion_num_steps[:bound][~self.terminate_state[:bound]].max()
                else:
                    curr_max = self.curr_steps - 1  # the ones that should be counted have terminated
                    # The remaining envs are not counted. So set all the remaining envs to terminated
                    self.terminate_state[bound:] = True
            else:
                curr_max = motion_num_steps[~self.terminate_state].max()

            if self.curr_steps >= curr_max:
                curr_max = self.curr_steps + 1  # For matching up the current steps and max steps.
        else:
            curr_max = motion_num_steps.max()

        self.mpjpe.append(info["mpjpe"])
        self.gt_pos.append(info["body_pos_gt"])
        self.pred_pos.append(info["body_pos"])
        self.curr_steps += 1

        # All motions fully played out, or all envs are terminated
        if self.curr_steps >= curr_max or self.terminate_state.sum() == self.num_envs:
            self.curr_steps = 0
            self.terminate_memory.append(self.terminate_state.cpu().numpy())
            self.success_rate = 1 - np.concatenate(self.terminate_memory)[: self.num_unique_motions].mean()

            # MPJPE
            all_mpjpe = torch.stack(self.mpjpe)
            # Max should be the same as the number of frames in the motion.
            assert all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == self.num_envs

            all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(motion_num_steps)]
            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(motion_num_steps)]
            all_body_pos_gt = np.stack(self.gt_pos)
            all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(motion_num_steps)]

            self.mpjpe_all.append(all_mpjpe)
            self.pred_pos_all += all_body_pos_pred
            self.gt_pos_all += all_body_pos_gt

            # All motions have been fully evaluated
            if self.task_env.motion_sample_start_idx + self.num_envs >= self.num_unique_motions:
                return self.get_final_stats()

            # Move on to the next motion
            self.task_env.forward_motion_samples()
            self.terminate_state[:] = False

            self.pbar.update(1)
            self.pbar.refresh()
            self.mpjpe, self.gt_pos, self.pred_pos = [], [], []

        update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_steps} | Start: {self.task_env.motion_sample_start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
        self.pbar.set_description(update_str)

        return False

    def get_final_stats(self):
        self.pbar.clear()
        terminate_hist = np.concatenate(self.terminate_memory)
        succ_idxes = np.flatnonzero(~terminate_hist[: self.num_unique_motions]).tolist()

        pred_pos_all_succ = [(self.pred_pos_all[: self.num_unique_motions])[i] for i in succ_idxes]
        gt_pos_all_succ = [(self.gt_pos_all[: self.num_unique_motions])[i] for i in succ_idxes]

        pred_pos_all = self.pred_pos_all[: self.num_unique_motions]
        gt_pos_all = self.gt_pos_all[: self.num_unique_motions]

        self.failed_keys = self.task_env.motion_data_keys[terminate_hist[: self.num_unique_motions]]
        # success_keys = self.task_env.motion_data_keys[~terminate_hist[:self.num_unique_motions]]

        metrics_all = compute_metrics_lite(pred_pos_all, gt_pos_all)
        metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

        metrics_all_print = {m: np.mean(v) for m, v in metrics_all.items()}
        metrics_succ_print = {m: np.mean(v) for m, v in metrics_succ.items()}

        if len(metrics_succ_print) == 0:
            print("No success!!!")
            metrics_succ_print = metrics_all_print

        print("------------------------------------------")
        print(f"Success Rate: {self.success_rate:.10f}")
        print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
        print("Succ: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]))
        print("Failed keys: ", len(self.failed_keys), ",", self.failed_keys)

        self.results = {
            "eval/success_rate": self.success_rate,
            "eval/mpjpe_all": metrics_all_print["mpjpe_g"],
            "eval/mpjpe_succ": metrics_succ_print["mpjpe_g"],
            "eval/accel_dist": metrics_succ_print["accel_dist"],
            "eval/vel_dist": metrics_succ_print["vel_dist"],
            "eval/mpjpel_all": metrics_all_print["mpjpe_l"],
            "eval/mpjpel_succ": metrics_succ_print["mpjpe_l"],
            "eval/mpjpe_pa": metrics_succ_print["mpjpe_pa"],
        }

        return True

    def update_env_and_close(self):
        # NOTE: Assuming that resampling motion will happen right after the eval,
        # so not resetting the env here.
        termination_history = self.task_env.untoggle_eval_mode(self.failed_keys)

        torch.cuda.empty_cache()
        gc.collect()

        if self.failed_save_path:
            joblib.dump(
                {
                    "failed_keys": self.failed_keys,
                    "termination_history": termination_history,
                },
                self.failed_save_path,
            )

        return self.results


def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args["policy"])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args["rnn"])
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args["train"]["device"])


def init_wandb(args, name, resume=True):
    import wandb

    exp_id = args["env_name"] + "-" + str(uuid.uuid4())[:8]
    wandb.init(
        id = exp_id,
        project=args["wandb_project"],
        allow_val_change=True,
        save_code=True,
        resume=resume,
        config=args,
        name=name,
    )
    return wandb, exp_id


def train(args, vec_env, policy, wandb=None, exp_id=None, skip_resample=False):
    if wandb is None and args["track"]:
        wandb, exp_id = init_wandb(args, args["env_name"])

    if exp_id is None:
        exp_id = args["env_name"] + "-" + str(uuid.uuid4())[:8]

    train_config = pufferlib.namespace(**args["train"], env=args["env_name"], exp_id=exp_id)
    data = clean_pufferl.create(train_config, vec_env, policy, wandb=wandb)

    data_dir = os.path.join(train_config.data_dir, exp_id)
    os.makedirs(data_dir, exist_ok=True)

    while data.global_step < train_config.total_timesteps:
        if not skip_resample and data.epoch > 0 and data.epoch % train_config.motion_resample_interval == 0:
            # Evaluate the model every 600 epochs (train_config.checkpoint_interval)
            if data.epoch % train_config.checkpoint_interval == 0:
                eval_stats = EvalStats(vec_env, failed_save_path=os.path.join(data_dir, f"failed_{data.epoch:06d}.pkl"))
                rollout(vec_env, policy, eval_stats)
                eval_results = eval_stats.update_env_and_close()
                if data.wandb:
                    eval_results["0verview/agent_steps"] = data.global_step
                    eval_results["0verview/epoch"] = data.epoch
                    wandb.log(eval_results)

            # Resample motions every 200 epochs (train_config.motion_resample_interval)
            vec_env.env.resample_motions()

        # Collect data
        results, _ = clean_pufferl.evaluate(data)

        # Update obs running mean and std
        # During evaluate() and train(), the obs_norm is NOT updated.
        rms_update_fn = getattr(data.policy.policy, "update_obs_rms", None)
        if rms_update_fn:
            rms_update_fn(data.experience.obs)

        amp_rms_update_fn = getattr(data.policy.policy, "update_amp_obs_rms", None)
        if data.use_amp_obs and amp_rms_update_fn:
            amp_rms_update_fn(data.experience.amp_obs)

        # Update policy
        clean_pufferl.train(data)

        # Apply learning rate exp decay
        if data.config.lr_decay_rate > 0:
            decay = math.exp(-data.config.lr_decay_rate * data.epoch)
            if decay < data.config.lr_decay_floor:
                decay = data.config.lr_decay_floor
            data.optimizer.param_groups[0]["lr"] = data.config.learning_rate * decay

    uptime = data.profile.uptime

    # Final evaluation
    eval_stats = EvalStats(vec_env)
    rollout(vec_env, policy, eval_stats)
    results.update(eval_stats.update_env_and_close())
    if data.wandb:
        results["0verview/agent_steps"] = data.global_step
        results["0verview/epoch"] = data.epoch
        wandb.log(results)

    # NOTE: Not using standard eval
    # steps_evaluated = 0
    # steps_to_eval = int(train_config.eval_timesteps)
    # batch_size = int(train_config.batch_size)
    # while steps_evaluated < steps_to_eval:
    #     stats, _ = clean_pufferl.evaluate(data)
    #     steps_evaluated += batch_size
    # clean_pufferl.mean_and_log(data)

    clean_pufferl.close(data)

    return results, uptime


def rollout(vec_env, policy, eval_stats=None):
    # NOTE (Important): Using deterministic action for evaluation
    policy.policy.set_deterministic_action(True)  # Ugly... but...

    obs, _ = vec_env.reset()
    state = None

    ep_cnt = 0
    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device)
            if hasattr(policy, "lstm"):
                action, _, _, _, state = policy(obs, state)
            else:
                action, _, _, _ = policy(obs)

            action = action.cpu().numpy().reshape(vec_env.action_space.shape)

        obs, _, _, _, info = vec_env.step(action)

        # Get episode-related info here
        if len(info) > 0:
            ep_ret = info[0]["episode_return"]
            ep_len = info[0]["episode_length"]
            print(f"Episode cnt: {vec_env.episode_count - ep_cnt}, Reward: {ep_ret:.3f}, Length: {ep_len:.3f}")
            ep_cnt = vec_env.episode_count

        if eval_stats:
            is_done = eval_stats.post_step_eval()
            if is_done:
                policy.policy.set_deterministic_action(False)
                break


### CARBS Sweeps
def sweep_carbs(args, sweep_count=500, max_suggestion_cost=3600):
    from math import log, ceil, floor

    from carbs import CARBS
    from carbs import CARBSParams
    from carbs import LinearSpace
    from carbs import LogSpace
    from carbs import LogitSpace
    from carbs import ObservationInParam

    # from carbs import ParamDictType
    from carbs import Param

    def closest_power(x):
        possible_results = floor(log(x, 2)), ceil(log(x, 2))
        return int(2 ** min(possible_results, key=lambda z: abs(x - 2**z)))

    def carbs_param(
        group,
        name,
        space,
        wandb_params,
        mmin=None,
        mmax=None,
        search_center=None,
        is_integer=False,
        rounding_factor=1,
        scale=1,
    ):
        wandb_param = wandb_params[group]["parameters"][name]
        if "values" in wandb_param:
            values = wandb_param["values"]
            mmin = min(values)
            mmax = max(values)

        if mmin is None:
            mmin = float(wandb_param["min"])
        if mmax is None:
            mmax = float(wandb_param["max"])

        if space == "log":
            Space = LogSpace
            if search_center is None:
                search_center = 2 ** (np.log2(mmin) + np.log2(mmax) / 2)
        elif space == "linear":
            Space = LinearSpace
            if search_center is None:
                search_center = (mmin + mmax) / 2
        elif space == "logit":
            Space = LogitSpace
            assert mmin == 0
            assert mmax == 1
            assert search_center is not None
        else:
            raise ValueError(f"Invalid CARBS space: {space} (log/linear)")

        return Param(
            name=f"{group}/{name}",
            space=Space(
                min=mmin,
                max=mmax,
                is_integer=is_integer,
                rounding_factor=rounding_factor,
                scale=scale,
            ),
            search_center=search_center,
        )

    if not os.path.exists("checkpoints"):
        os.system("mkdir checkpoints")

    import wandb

    sweep_id = wandb.sweep(
        sweep=args["sweep"],
        project="carbs",
    )
    target_metric = args["sweep"]["metric"]["name"].split("/")[-1]
    sweep_parameters = args["sweep"]["parameters"]

    # Must be hardcoded and match wandb sweep space for now
    param_spaces = []
    if "total_timesteps" in sweep_parameters["train"]["parameters"]:
        time_param = sweep_parameters["train"]["parameters"]["total_timesteps"]
        min_timesteps = time_param["min"]
        param_spaces.append(
            carbs_param(
                "train", "total_timesteps", "log", sweep_parameters, search_center=min_timesteps, is_integer=True
            )
        )

    # batch_param = sweep_parameters['train']['parameters']['batch_size']
    # default_batch = (batch_param['max'] - batch_param['min']) // 2

    # minibatch_param = sweep_parameters['train']['parameters']['minibatch_size']
    # default_minibatch = (minibatch_param['max'] - minibatch_param['min']) // 2

    # env params to sweep
    if "env" in sweep_parameters:
        param_spaces.append(carbs_param("env", "rew_power_coef", "linear", sweep_parameters, search_center=args["ssc_rew"]))

    param_spaces += [
        carbs_param("train", "learning_rate", "log", sweep_parameters, search_center=args["ssc_lr"]),
        carbs_param('train', 'gamma', 'logit', sweep_parameters, search_center=0.97),
        carbs_param('train', 'gae_lambda', 'logit', sweep_parameters, search_center=0.50),
        # carbs_param('train', 'update_epochs', 'linear', sweep_parameters,
        #     search_center=3, is_integer=True),
        carbs_param("train", "clip_coef", "logit", sweep_parameters, search_center=0.03),
        carbs_param("train", "vf_coef", "linear", sweep_parameters, search_center=2.0),
        # carbs_param("train", "vf_clip_coef", "logit", sweep_parameters, search_center=0.2),
        # carbs_param('train', 'max_grad_norm', 'linear', sweep_parameters, search_center=1.0),
        # carbs_param('train', 'ent_coef', 'log', sweep_parameters, search_center=0.0001),
        # carbs_param('train', 'batch_size', 'log', sweep_parameters,
        #     search_center=default_batch, is_integer=True),
        # carbs_param('train', 'minibatch_size', 'log', sweep_parameters,
        #     search_center=default_minibatch, is_integer=True),
        # carbs_param('train', 'bptt_horizon', 'log', sweep_parameters,
        #     search_center=8, is_integer=True),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=5,
        num_random_samples=len(param_spaces),
        max_suggestion_cost=max_suggestion_cost,
        is_saved_on_every_observation=False,
    )
    carbs = CARBS(carbs_params, param_spaces)

    def main():
        # set torch and pytorch seeds to current time
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))

        wandb, exp_id = init_wandb(args, args["env_name"])
        wandb.config.__dict__["_locked"] = {}

        orig_suggestion = carbs.suggest().suggestion
        suggestion = orig_suggestion.copy()
        print("Suggestion:", suggestion)
        train_suggestion = {k.split("/")[1]: v for k, v in suggestion.items() if k.startswith("train/")}
        env_suggestion = {k.split("/")[1]: v for k, v in suggestion.items() if k.startswith("env/")}
        args["train"].update(train_suggestion)
        # args['train']['batch_size'] = closest_power(
        #     train_suggestion['batch_size'])
        # args['train']['minibatch_size'] = closest_power(
        #     train_suggestion['minibatch_size'])
        # args['train']['bptt_horizon'] = closest_power(
        #     train_suggestion['bptt_horizon'])

        args["env"].update(env_suggestion)
        args["track"] = True
        wandb.config.update({"train": args["train"]}, allow_val_change=True)
        wandb.config.update({"env": args["env"]}, allow_val_change=True)

        print(wandb.config.train)
        print(wandb.config.env)
        print(wandb.config.policy)

        try:
            vec_env = pufferlib.vector.make(env_creator, env_kwargs=args["env"])
            policy_cls = getattr(policy_module, args["policy_name"])
            rnn_cls = None
            if "rnn_name" in args:
                rnn_cls = getattr(policy_module, args["rnn_name"])
            policy = make_policy(vec_env.driver_env, policy_cls, rnn_cls, args)

            stats, uptime = train(args, vec_env, policy, wandb, exp_id)  # , skip_resample=True)

        except Exception as e:
            import traceback

            traceback.print_exc()

        else:
            observed_value = 0
            for k, v in stats.items():
                if k.endswith(target_metric):
                    observed_value = v
                    break

            print("Observed value:", observed_value)
            print("Uptime:", uptime)

            carbs.observe(
                ObservationInParam(
                    input=orig_suggestion,
                    output=observed_value,
                    cost=uptime,
                )
            )

    wandb.agent(sweep_id, main, count=sweep_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument("--config", default="config/morph.ini")
    parser.add_argument(
        "--mode", type=str, default="train", choices="train eval sweep".split()
    )  # render-eval, batch-eval?
    parser.add_argument("-m", "--motion-file", type=str, default=None, help="Path to motion file")
    parser.add_argument("-p", "--eval-model-path", type=str, default=None, help="Path to a pretrained checkpoint")
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--ssc-lr", type=float, default=.0002, help="Sweep search center for learning rate")
    parser.add_argument("--ssc-rew", type=float, default=.0005, help="Sweep search center for rew power")

    args = parser.parse_known_args()[0]

    p = configparser.ConfigParser()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p.read(os.path.join(current_dir, args.config))

    for section in p.sections():
        for key in p[section]:
            if section == "base":
                argparse_key = f"--{key}".replace("_", "-")
            else:
                argparse_key = f"--{section}.{key}".replace("_", "-")
            parser.add_argument(argparse_key, default=p[section][key])

    # Late add help so you get a dynamic menu based on the env
    parser.add_argument(
        "-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit"
    )

    parsed = parser.parse_args().__dict__
    args = {"env": {}, "policy": {}, "rnn": {}}
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:
            prev[subkey] = value

    device = args["train"]["device"]

    # Create the environment
    args["env"]["name"] = args["env_name"]
    args["env"]["device_type"] = device
    if args["motion_file"]:
        args["env"]["motion_file"] = args["motion_file"]

    # If sweep, run sweep here and exit
    if args["mode"] == "sweep":
        sweep_carbs(args, sweep_count=500)
        sys.exit(0)

    # Create the env and policy
    vec_env = pufferlib.vector.make(env_creator, env_kwargs=args["env"])
    policy_cls = getattr(policy_module, args["policy_name"])
    rnn_cls = None
    if "rnn_name" in args:
        rnn_cls = getattr(policy_module, args["rnn_name"])
    policy = make_policy(vec_env.driver_env, policy_cls, rnn_cls, args)

    if args["eval_model_path"]:
        checkpoint = torch.load(args["eval_model_path"], map_location=device)
        policy.load_state_dict(checkpoint["state_dict"])

    # Train or evaluate
    if args["mode"] == "train":
        train(args, vec_env, policy)

    elif args["mode"] == "eval":
        eval_stats = None  # EvalStats(vec_env)
        stats = rollout(vec_env, policy, eval_stats)
