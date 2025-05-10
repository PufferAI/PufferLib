from pdb import set_trace as T

class Policy(torch.nn.Module):
    '''Wrap a non-recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.is_continuous = hasattr(policy, 'is_continuous') and policy.is_continuous
        self.hidden_size = policy.hidden_size

    def get_value(self, x, state=None):
        _, value = self.policy(x)
        return value

    def get_action_and_value(self, x, action=None):
         logits, value, e3b, intrinsic_reward = self.policy(x, e3b=e3b)
         action, logprob, entropy = sample_logits(logits, action, self.is_continuous)
         return action, logprob, entropy, value, e3b, intrinsic_reward

    def forward(self, x, action=None, e3b=None):
        return self.get_action_and_value(x, action, e3b)


class RecurrentPolicy(torch.nn.Module):
    '''Wrap a recurrent PyTorch model for use with CleanRL'''
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.is_continuous = hasattr(policy.policy, 'is_continuous') and policy.policy.is_continuous
        self.hidden_size = policy.hidden_size

    @property
    def lstm(self):
        if hasattr(self.policy, 'recurrent'):
            return self.policy.recurrent
        elif hasattr(self.policy, 'lstm'):
            return self.policy.lstm
        else:
            raise ValueError('Policy must have a subnetwork named lstm or recurrent')

    def get_value(self, x, state=None):
        _, value, _ = self.policy(x, state)

    def get_action_and_value(self, x, state=None, action=None, e3b=None):
        #logits, value, state, e3b, intrinsic_reward = self.policy(x, state, e3b=e3b)
        logits, value_mean, value_logstd, state = self.policy(x, state, e3b=e3b)
        action, logprob, entropy = sample_logits(logits, action, self.is_continuous)
        return action, logprob, entropy, value_mean, value_logstd, state#, e3b, intrinsic_reward

    def forward(self, x, state=None, action=None, e3b=None):
        return self.get_action_and_value(x, state, action, e3b)
