from pdb import set_trace as T

import os

import pufferlib.policy_store
import pufferlib.policy_ranker
import pufferlib.policy_pool
import pufferlib.utils


class PolicyGod:
    def __init__(
            self, envs, agent, data_dir, resume_state, device, num_envs, num_agents,
            selfplay_learner_weight, selfplay_num_policies,
            policy_store=None,
            policy_ranker=None,
            policy_pool=None,
            policy_selector=None
            ):

        # Create policy store
        if policy_store is None:
            assert data_dir is not None
            self.policy_store = pufferlib.policy_store.DirectoryPolicyStore(
                os.path.join(data_dir, "policies"))

        # Create policy ranker
        if policy_ranker is None:
            assert data_dir is not None
            self.policy_ranker = pufferlib.utils.PersistentObject(
                os.path.join(data_dir, "openskill.pickle"),
                pufferlib.policy_ranker.OpenSkillRanker,
                "anchor",
            )
            if "learner" not in self.policy_ranker.ratings():
                self.policy_ranker.add_policy("learner")

        # Setup agent
        if "policy_checkpoint_name" in resume_state:
          try:
              agent = self.policy_store.get_policy(
                resume_state["policy_checkpoint_name"]
              ).policy(policy_args=[envs])
          except:
              print('Failed to load policy checkpoint: ',
                resume_state["policy_checkpoint_name"])

        # TODO: this can be cleaned up
        agent.is_recurrent = hasattr(agent, "lstm")
        agent = agent.to(device)

        # Setup policy pool
        if policy_pool is None:
            self.policy_pool = pufferlib.policy_pool.PolicyPool(
                agent,
                "learner",
                num_envs=num_envs,
                num_agents=num_agents,
                learner_weight=selfplay_learner_weight,
                num_policies=selfplay_num_policies,
            )
            self.forwards = self.policy_pool.forwards

        # Setup policy selector
        if policy_selector is None:
            self.policy_selector = pufferlib.policy_ranker.PolicySelector(
                selfplay_num_policies - 1, exclude_names="learner"
            )

        self.agent = agent

    def update(self):
        self.policy_pool.update_policies({
            p.name: p.policy(
                policy_args=[self.buffers[0]],
                device=self.device,
            ) for p in self.policy_store.select_policies(self.policy_selector)
        })

    def update_scores(self, i, key):
         return self.policy_pool.update_scores(i, key)

    def update_ranks(self, global_step):
        if self.policy_pool.scores and self.policy_ranker is not None:
          self.policy_ranker.update_ranks(
              self.policy_pool.scores,
              wandb_policies=[],
              step=global_step,
          )
          self.policy_pool.scores = {}

    def add_policy(self, name, policy):
        self.policy_store.add_policy(name, policy)
        if self.policy_ranker is not None:
            self.policy_ranker.add_policy_copy(
                name, self.policy_pool._learner_name)

    def update_policies(self):
        self.policy_pool.update_policies({
            p.name: p.policy(
                policy_args=[self.buffers[0]],
                device=self.device,
            ) for p in self.policy_store.select_policies(self.policy_selector)
        })

