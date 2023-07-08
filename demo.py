import pufferlib

# TODO: Fix circular import depending on import order
from clean_pufferl import CleanPuffeRL
import config as demo_config
from pufferlib.policy_pool import PolicyPool
from pufferlib.policy_ranker import OpenSkillRanker

config = demo_config.NMMO()
#config = demo_config.Atari(framestack=1)
#config = demo_config.MAgent()
#config = demo_config.Griddly()
#config = demo_config.Crafter()
#config = demo_config.NetHack()
from pufferlib.policy_store import MemoryPolicyStore, FilePolicyRecord, MemoryPolicyStore, PolicyRecord, PolicySelector, PolicyStore

all_bindings = [config.all_bindings[0]]

def train_model(binding):
    policy_store = MemoryPolicyStore()

    agent = pufferlib.frameworks.cleanrl.make_policy(
            config.Policy, recurrent_args=config.recurrent_args,
            recurrent_kwargs=config.recurrent_kwargs,
        )(binding, *config.policy_args, **config.policy_kwargs).to(config.device)
    policy_store.add_policy('learner', agent)

    policy_pool = PolicyPool(num_policies=2, learner_weight=0.8)
    ranker = OpenSkillRanker(anchor='learner')

    trainer = CleanPuffeRL(
            binding,
            agent,
            policy_pool,
            num_buffers=config.num_buffers,
            num_envs=config.num_envs,
            num_cores=config.num_cores,
            batch_size=config.batch_size,
            vec_backend=config.vec_backend,
            seed=config.seed,
    )

    #trainer.load_model(path)
    trainer.init_wandb()

    num_updates = config.total_timesteps // config.batch_size
    for update in range(num_updates):
        trainer.evaluate()

        if update % config.pool_rank_interval == 0:
            ranker.update_ranks(policy_pool.scores())

        if update % config.pool_update_policy_interval == 0:
            policy_pool.update(
                policy_store.select_policies(
                    ranker.selector(
                        num_policies=config.num_policies - 1,
                        exclude=['learner'])),
            )

        if update % config.pool_update_policy_interval == 0:
            policy_store.add_policy_copy(f"learner-{update}", "learner")

        trainer.train(agent, data,
            batch_rows=config.batch_rows,
            bptt_horizon=config.bptt_horizon,
        )

        print(ranker.ranks())

    trainer.close()

for binding in all_bindings:
    train_model(binding)
