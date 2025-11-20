from pufferlib import pufferl

def evaluate(env_name, load_model_path):
    args = pufferl.load_config(env_name)
    args['vec']['num_envs'] = 1
    args['env']['num_envs'] = 4096
    args['load_model_path'] = load_model_path
    # Turn off endgame_envs and scaffolding episodes, which do not report results
    args['env']['endgame_env_prob'] = 0
    args['env']['scaffolding_ratio'] = 0
    args['env']['can_go_over_65536'] = True

    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)
    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    # Each evaluate runs for 64 ticks. NOTE: bppt horizon might be short for g2048?
    # Avg episode length from the current model is ~18000, so it takes ~300 epochs for an avg episode.
    # It's hard to get the single best score because stats are already averaged across done envs.
    for i in range(10000):
        stats = trainer.evaluate()

        trainer.epoch += 1
        if i % 20 == 0:
            trainer.print_dashboard()

    trainer.close()

    # Get the estimates
    num_episodes = sum(stats['n'])
    episode_lengths = sum(n * l for n, l in zip(stats['n'], stats['episode_length'])) / num_episodes
    max_tiles = sum(n * m for n, m in zip(stats['n'], stats['score'])) / num_episodes
    merge_scores = sum(n * s for n, s in zip(stats['n'], stats['merge_score'])) / num_episodes
    reached_32768 = sum(n * s for n, s in zip(stats['n'], stats['reached_32768'])) / num_episodes
    reached_65536 = sum(n * s for n, s in zip(stats['n'], stats['reached_65536'])) / num_episodes

    print(f"Num episodes: {int(num_episodes)}")
    print(f"Max tile avg: {max_tiles:.1f}")
    # The stats from vecenv are averaged across envs that were done in the same tick. Cannot get the single max.
    print(f"Episode length -- Avg: {episode_lengths:.1f}, Max: {max(stats['episode_length']):.1f}")
    print(f"Merge score -- Avg: {merge_scores:.1f}, Max: {max(stats['merge_score']):.1f}")
    print(f"Reached 32768 prob: {reached_32768*100:.2f} %")
    print(f"Reached 65536 prob: {reached_65536*100:.2f} %")

    """
    # hidden 256: https://wandb.ai/kywch/pufferlib/runs/nvd0pfuj?nw=nwuserkywch
    Num episodes: 154406
    Max tile avg: 22532.9
    Episode length -- Avg: 16667.2, Max: 26659.1
    Merge score -- Avg: 462797.9, Max: 744224.9
    Reached 32768 prob: 46.08 %
    Reached 65536 prob: 3.53 %

    # hidden 512: https://wandb.ai/kywch/pufferlib/runs/2ch3my60?nw=nwuserkywch
    Num episodes: 119243
    Max tile avg: 30662.2
    Episode length -- Avg: 21539.7, Max: 29680.3
    Merge score -- Avg: 618011.8, Max: 918755.8
    Reached 32768 prob: 68.25 %
    Reached 65536 prob: 13.09 %

    # hidden 512 (replication): https://wandb.ai/kywch/pufferlib/runs/5thsjr61?nw=nwuserkywch
    Num episodes: 115652
    Max tile avg: 31773.2
    Episode length -- Avg: 22196.4, Max: 30316.5
    Merge score -- Avg: 639395.6, Max: 909969.8
    Reached 32768 prob: 71.22 %
    Reached 65536 prob: 14.75 %
    """

def finetune(env_name, load_model_path):
    args = pufferl.load_config(env_name)
    args['load_model_path'] = load_model_path
    # args['env']['use_sparse_reward'] = True
    args['env']['scaffolding_ratio'] = 0.85

    # args['policy']['hidden_size'] = 512
    # args['rnn']['input_size'] = 512
    # args['rnn']['hidden_size'] = 512

    args['train']['total_timesteps'] = 1_000_000_000
    args['train']['learning_rate'] = 0.00005
    args['train']['anneal_lr'] = False

    args['wandb'] = True
    args['tag'] = 'pg2048'

    pufferl.train(env_name, args)

if __name__ == '__main__':
    import os
    import wandb

    # https://wandb.ai/kywch/pufferlib/runs/5thsjr61?nw=nwuserkywch
    wandb_run_id = '5thsjr61'
    wandb.init(id=wandb_run_id, project='pufferlib', entity='kywch')

    artifact = wandb.use_artifact(f'{wandb_run_id}:latest')
    data_dir = artifact.download()
    model_file = max(os.listdir(data_dir))
    model_path = f'{data_dir}/{model_file}'
    wandb.finish()

    evaluate('puffer_g2048', load_model_path=model_path)
    # finetune('puffer_g2048', load_model_path='puffer_g2048_256_base.pt')
