from pdb import set_trace as T
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
from pufferlib.policy_store import MemoryPolicyStore, MemoryPolicyStore

all_bindings = [config.all_bindings[0]]

def train_model(binding):
    policy_store = MemoryPolicyStore()
    ranker = OpenSkillRanker(anchor='learner')

    #config.batch_size=1024

    agent = pufferlib.frameworks.cleanrl.make_policy(
            config.Policy, recurrent_args=config.recurrent_args,
            recurrent_kwargs=config.recurrent_kwargs,
        )(binding, *config.policy_args, **config.policy_kwargs).to(config.device)

    policy_store.add_policy('learner', agent)

    policy_pool = PolicyPool(
        agent,
        'learner',
        batch_size = binding.max_agents * config.num_envs,
        num_policies = 2,
        learner_weight = 3
    )

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
    #trainer.init_wandb()

    num_updates = config.total_timesteps // config.batch_size
    for update in range(num_updates):
        print("Evaluating...", update)
        trainer.evaluate()

        if update % config.pool_rank_interval == 0:
            print("Updating OpenSkill Ranks")
            ranker.update_ranks({
                name: score for name, score in policy_pool.scores.items()
            })
            print("Ranks", ranker.ratings())

        if update % config.pool_update_policy_interval == 0:
            print("Updating PolicyPool")
            records = policy_store.select_policies(
                ranker.selector(
                    num_policies=1,
                    exclude=['learner']))
            print("Selected policies", records)
            # TODO: Make policy a property
            policy_pool.update_policies({r.name: r.policy() for r in records})

        if update % config.pool_update_policy_interval == 0:
            print(f"Adding new Policy (learner-{update}) to PolicyStore")
            policy_store.add_policy_copy(f"learner-{update}", "learner")

        print("Training...")
        trainer.train(batch_rows=config.batch_rows, bptt_horizon=config.bptt_horizon)


    trainer.close()

for binding in all_bindings:
    train_model(binding)
