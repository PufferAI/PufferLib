
from pufferlib.frameworks import make_recurrent_policy, BasePolicy

def make_cleanrl_policy(policy_cls, lstm_layers=0):
    assert issubclass(policy_cls, BasePolicy)
    if lstm_layers > 0:
        policy_cls = make_recurrent_policy(policy_cls, lstm_layers)

    class CleanRLPolicy(policy_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        # TODO: Compute seq_lens from done
        def get_value(self, x, lstm_state=None, done=None):
            logits, _ = self.forward(x, lstm_state, done)
            return self.value_head(logits)

        # TODO: Compute seq_lens from done
        def get_action_and_value(self, x, lstm_state=None, done=None, action=None):
            logits, lstm_state = self.forward_rnn(x, lstm_state, done)
            value = self.value_head(x)

            mulit_categorical = [Categorical(logits=l) for l in flat_logits]

            if action is None:
                action = torch.stack([c.sample() for c in mulit_categorical])
            else:
                action = action.view(-1, action.shape[-1]).T

            logprob = torch.stack([c.log_prob(a) for c, a in zip(mulit_categorical, action)]).T.sum(1)
            entropy = torch.stack([c.entropy() for c in mulit_categorical]).T.sum(1)

            if lstm_layers > 0:
                return action.T, logprob, entropy, value, lstm_state
            return action.T, logprob.sum, entropy, value
   
    return CleanRLPolicy

