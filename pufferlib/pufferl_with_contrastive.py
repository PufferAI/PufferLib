"""
Extended PuffeRL trainer with contrastive loss integration.

This demonstrates how to integrate the contrastive loss into PufferLib's training loop.
The key additions are:
1. Extracting embeddings from the policy
2. Computing contrastive loss alongside standard losses
3. Adding contrastive metrics to logging
"""

import torch
from collections import defaultdict
from .contrastive_loss import compute_contrastive_loss_pufferlib, get_embeddings_from_policy_data


def train_with_contrastive_loss(pufferl_instance):
    """Modified training function that includes contrastive loss.

    This extends the standard PuffeRL training loop to include contrastive learning.
    You would replace the train() method in PuffeRL with this implementation,
    or create a subclass that overrides the train method.
    """
    profile = pufferl_instance.profile
    epoch = pufferl_instance.epoch
    profile('train', epoch)
    losses = defaultdict(float)
    config = pufferl_instance.config
    device = config['device']

    # Standard PPO setup
    b0 = config['prio_beta0']
    a = config['prio_alpha']
    clip_coef = config['clip_coef']
    vf_clip = config['vf_clip_coef']
    anneal_beta = b0 + (1 - b0) * a * pufferl_instance.epoch / pufferl_instance.total_epochs
    pufferl_instance.ratio[:] = 1

    # Contrastive loss configuration (should come from config)
    use_contrastive = config.get('use_contrastive_loss', False)
    contrastive_coef = config.get('contrastive_coef', 1.0)
    contrastive_temperature = config.get('contrastive_temperature', 0.1)
    contrastive_discount = config.get('contrastive_discount', 0.99)
    embedding_dim = config.get('embedding_dim', 256)

    for mb in range(pufferl_instance.total_minibatches):
        profile('train_misc', epoch, nest=True)
        pufferl_instance.amp_context.__enter__()

        shape = pufferl_instance.values.shape
        advantages = torch.zeros(shape, device=device)

        # Import the advantage computation function
        from pufferlib.pufferl import compute_puff_advantage

        advantages = compute_puff_advantage(
            pufferl_instance.values,
            pufferl_instance.rewards,
            pufferl_instance.terminals,
            pufferl_instance.ratio,
            advantages,
            config['gamma'],
            config['gae_lambda'],
            config['vtrace_rho_clip'],
            config['vtrace_c_clip']
        )

        profile('train_copy', epoch)
        adv = advantages.abs().sum(axis=1)
        prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
        idx = torch.multinomial(prio_probs, pufferl_instance.minibatch_segments)
        mb_prio = (pufferl_instance.segments * prio_probs[idx, None])**-anneal_beta
        mb_obs = pufferl_instance.observations[idx]
        mb_actions = pufferl_instance.actions[idx]
        mb_logprobs = pufferl_instance.logprobs[idx]
        mb_rewards = pufferl_instance.rewards[idx]
        mb_terminals = pufferl_instance.terminals[idx]
        mb_truncations = pufferl_instance.truncations[idx]
        mb_ratio = pufferl_instance.ratio[idx]
        mb_values = pufferl_instance.values[idx]
        mb_returns = advantages[idx] + mb_values
        mb_advantages = advantages[idx]

        profile('train_forward', epoch)
        if not config['use_rnn']:
            mb_obs_flat = mb_obs.reshape(-1, *pufferl_instance.vecenv.single_observation_space.shape)
        else:
            mb_obs_flat = mb_obs

        state = dict(
            action=mb_actions,
            lstm_h=None,
            lstm_c=None,
        )

        # Forward pass through policy
        logits, newvalue = pufferl_instance.policy(mb_obs_flat, state)

        # Import the sampling function
        import pufferlib.pytorch
        actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

        profile('train_misc', epoch)
        newlogprob = newlogprob.reshape(mb_logprobs.shape)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()
        pufferl_instance.ratio[idx] = ratio.detach()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

        adv = advantages[idx]
        adv = compute_puff_advantage(
            mb_values, mb_rewards, mb_terminals,
            ratio, adv, config['gamma'], config['gae_lambda'],
            config['vtrace_rho_clip'], config['vtrace_c_clip']
        )
        adv = mb_advantages
        adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

        # Standard PPO losses
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalue = newvalue.view(mb_returns.shape)
        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        entropy_loss = entropy.mean()

        # Standard PPO loss
        standard_loss = pg_loss + config['vf_coef'] * v_loss - config['ent_coef'] * entropy_loss

        # Contrastive loss computation
        total_loss = standard_loss
        contrastive_loss_value = torch.tensor(0.0, device=device)
        contrastive_metrics = {}

        if use_contrastive:
            try:
                # Extract embeddings from policy output
                # In practice, you'd want to modify your policy to directly provide embeddings
                # This is a fallback that uses value predictions as embeddings
                embeddings = get_embeddings_from_policy_data(
                    logits, newvalue, embedding_dim, device
                )

                # Reshape embeddings to match minibatch structure
                if embeddings.dim() == 2:
                    embeddings = embeddings.view(mb_obs.shape[0], mb_obs.shape[1], -1)

                # Compute contrastive loss
                contrastive_loss_value, contrastive_metrics = compute_contrastive_loss_pufferlib(
                    embeddings=embeddings,
                    terminals=mb_terminals,
                    truncations=mb_truncations,
                    temperature=contrastive_temperature,
                    contrastive_coef=contrastive_coef,
                    embedding_dim=embedding_dim,
                    discount=contrastive_discount,
                    device=device,
                )

                # Add contrastive loss to total loss
                total_loss = total_loss + contrastive_loss_value

            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Contrastive loss computation failed: {e}")
                contrastive_loss_value = torch.tensor(0.0, device=device)

        # Use total loss for backward pass
        pufferl_instance.amp_context.__enter__()

        # Update values as in original
        pufferl_instance.values[idx] = newvalue.detach().float()

        # Standard loss logging
        profile('train_misc', epoch)
        losses['policy_loss'] += pg_loss.item() / pufferl_instance.total_minibatches
        losses['value_loss'] += v_loss.item() / pufferl_instance.total_minibatches
        losses['entropy'] += entropy_loss.item() / pufferl_instance.total_minibatches
        losses['old_approx_kl'] += old_approx_kl.item() / pufferl_instance.total_minibatches
        losses['approx_kl'] += approx_kl.item() / pufferl_instance.total_minibatches
        losses['clipfrac'] += clipfrac.item() / pufferl_instance.total_minibatches
        losses['importance'] += ratio.mean().item() / pufferl_instance.total_minibatches

        # Contrastive loss logging
        if use_contrastive:
            losses['contrastive_loss'] += contrastive_loss_value.item() / pufferl_instance.total_minibatches
            for key, value in contrastive_metrics.items():
                metric_name = f'contrastive_{key}'
                if metric_name not in losses:
                    losses[metric_name] = 0.0
                losses[metric_name] += value / pufferl_instance.total_minibatches

        # Learn on accumulated minibatches
        profile('learn', epoch)
        total_loss.backward()
        if (mb + 1) % pufferl_instance.accumulate_minibatches == 0:
            torch.nn.utils.clip_grad_norm_(pufferl_instance.policy.parameters(), config['max_grad_norm'])
            pufferl_instance.optimizer.step()
            pufferl_instance.optimizer.zero_grad()

    # Rest of the training function remains the same as original
    profile('train_misc', epoch)
    if config['anneal_lr']:
        pufferl_instance.scheduler.step()

    y_pred = pufferl_instance.values.flatten()
    y_true = advantages.flatten() + pufferl_instance.values.flatten()
    var_y = y_true.var()
    explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
    losses['explained_variance'] = explained_var.item()

    profile.end()
    logs = None
    pufferl_instance.epoch += 1
    done_training = pufferl_instance.global_step >= config['total_timesteps']

    if done_training or pufferl_instance.global_step == 0 or pufferl_instance.uptime > pufferl_instance.last_log_time + 0.25:
        logs = pufferl_instance.mean_and_log()
        pufferl_instance.losses = losses
        pufferl_instance.print_dashboard()
        pufferl_instance.stats = defaultdict(list)
        pufferl_instance.last_log_time = pufferl_instance.uptime
        pufferl_instance.last_log_step = pufferl_instance.global_step
        profile.clear()

    if pufferl_instance.epoch % config['checkpoint_interval'] == 0 or done_training:
        pufferl_instance.save_checkpoint()
        pufferl_instance.msg = f'Checkpoint saved at update {pufferl_instance.epoch}'

    return logs


class PuffeRLWithContrastive:
    """Example of how to extend PuffeRL with contrastive loss.

    You can use this as a reference for integrating contrastive loss
    into your own PufferLib training setup.
    """

    def __init__(self, config, vecenv, policy, logger=None):
        # Import and initialize base PuffeRL
        from pufferlib.pufferl import PuffeRL
        self.base_trainer = PuffeRL(config, vecenv, policy, logger)

        # Add contrastive loss specific configuration
        self.use_contrastive = config.get('use_contrastive_loss', False)
        if self.use_contrastive:
            print(f"Contrastive loss enabled with coefficient {config.get('contrastive_coef', 1.0)}")

    def train(self):
        """Training method with contrastive loss."""
        return train_with_contrastive_loss(self.base_trainer)

    def evaluate(self):
        """Evaluation remains the same."""
        return self.base_trainer.evaluate()

    def __getattr__(self, name):
        """Delegate other attributes to base trainer."""
        return getattr(self.base_trainer, name)