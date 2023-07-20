from pdb import set_trace as T
import torch
import pufferlib

# TODO: Fix circular import depending on import order
from clean_pufferl import CleanPuffeRL
import config as demo_config
import pufferlib.policy_store
from pufferlib.policy_pool import PolicyPool
from pufferlib.policy_ranker import OpenSkillRanker

config = demo_config.NMMO()
#config = demo_config.Atari(framestack=1)
#config = demo_config.MAgent()
#config = demo_config.Griddly()
#config = demo_config.Crafter()
#config = demo_config.NetHack()
from pufferlib.policy_store import MemoryPolicyStore, MemoryPolicyStore

all_bindings = [config.all_bindings[0]]

def train_model(binding):
    policy_store = MemoryPolicyStore()
    ranker = OpenSkillRanker(anchor='learner')

    #config.batch_size=1024

    agent = config.Policy(binding,
            *config.policy_args,
            recurrent_cls=torch.nn.LSTM,
            recurrent_kwargs=config.recurrent_kwargs,
    ).to(config.device)

    policy_store.add_policy('learner', agent)

    policy_store = pufferlib.policy_store.DirectoryPolicyStore('policies')

    trainer = CleanPuffeRL(
            binding,
            agent,
            data_dir='data',
            policy_store=policy_store,
            num_buffers=config.num_buffers,
            num_envs=config.num_envs,
            num_cores=config.num_cores,
            batch_size=config.batch_size,
            vec_backend=config.vec_backend,
            seed=config.seed,
            policy_ranker=ranker,
    )

    #trainer.load_model(path)
    #trainer.init_wandb()

    num_updates = config.total_timesteps // config.batch_size
    for update in range(num_updates):
        print("Evaluating...", update)
        trainer.evaluate()
        print("Training...", update)
        trainer.train(
            bptt_horizon=config.bptt_horizon,
            batch_rows=config.batch_rows,
        )

    trainer.close()

for binding in all_bindings:
    train_model(binding)
