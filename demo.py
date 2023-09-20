from pdb import set_trace as T
import argparse
import sys
import time
import os

import pufferlib

# TODO: Fix circular import depending on import order
from clean_pufferl import CleanPuffeRL

import pufferlib.utils
import pufferlib.models

def make_policy(envs, config):
    policy = config.policy_cls(envs.driver_env, **config.policy_kwargs)
    if config.recurrent_cls is not None:
        policy = pufferlib.models.RecurrentWrapper(
            envs, policy, **config.recurrent_kwargs)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)
    policy = policy.to(config.cleanrl_init.device)
    return policy

def sweep_model(config, env_creator):
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "targets_hit"},
        "parameters": {
            "learning_rate": {"max": 0.1, "min": 0.0001},
        },
    }

    import wandb
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="pufferlib")

    def main():
        config.cleanrl_init.learning_rate = wandb.config.learning_rate
        train_model(config, env_creator)

    wandb.agent(sweep_id, main)


def train_model(config, env_creator):
    env_creator_kwargs = config.env_creators[env_creator]
    trainer = CleanPuffeRL(
        agent_creator=make_policy,
        agent_kwargs={'config': config},
        env_creator=env_creator,
        env_creator_kwargs=env_creator_kwargs,
        **config.cleanrl_init,
    )

    #trainer.load_model(path)

    num_updates = (config.cleanrl_init.total_timesteps 
        // config.cleanrl_init.batch_size)

    for update in range(num_updates):
        trainer.evaluate()
        trainer.train(**config.cleanrl_train)

    trainer.close()

def evaluate_model(config, env_creator):
    env_creator_kwargs = config.env_creators[env_creator]
    env = env_creator(**env_creator_kwargs)

    import torch
    device = config.cleanrl_init.device
    agent = torch.load('agent.pt').to(device)

    ob = env.reset()
    for i in range(100):
        ob = torch.tensor(ob).view(1, -1).to(device)
        with torch.no_grad():
            action  = agent.get_action_and_value(ob)[0][0].item()

        ob, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1)

        if done:
            ob = env.reset()
            print('---Reset---')
            env.render()
            time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse environment argument")
    parser.add_argument("--env", type=str, default="nmmo", help="Environment name")
    parser.add_argument("--mode", type=str, default="train", help="train/eval/sweep")
    parser.add_argument("--run-id", type=str, default="", help="Experiment name")
    parser.add_argument('--track', action='store_true', help='Track on WandB')
    args = parser.parse_args()
    assert args.mode in ('train', 'eval', 'sweep')

    import config as config_module
    config = getattr(config_module, args.env)()

    if args.track:
        os.environ["WANDB_SILENT"] = "true"
        import wandb
        wandb.init(
            id=args.run_id or wandb.util.generate_id(),
            project=config.cleanrl_init.wandb_project,
            entity=config.cleanrl_init.wandb_entity,
            group=config.cleanrl_init.wandb_group,
            #config=run_config.__dict__,
            name='clean pufferl',
            monitor_gym=True,
            save_code=True,
            resume="allow",
        )

    for env_creator in config.env_creators:
        if args.mode == 'train':
            train_model(config, env_creator)
        elif args.mode == 'eval':
            evaluate_model(config, env_creator)
        elif args.mode == 'sweep':
            sweep_model(config, env_creator)
