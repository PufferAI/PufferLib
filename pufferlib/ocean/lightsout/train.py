from pufferlib import pufferl


def train_until_target(env_name="puffer_lightsout", load_model_path=None):
    args = pufferl.load_config(env_name)

    args["train"]["device"] = "cuda"
    args["vec"]["backend"] = "PufferEnv"
    args["vec"]["num_envs"] = 1
    args["env"]["num_envs"] = 4096
    args["env"]["grid_size"] = 8

    # High cap; run stops early when target is stable.
    args["train"]["total_timesteps"] = 2_000_000_000
    args["train"]["ent_coef"] = 0.005
    args["train"]["learning_rate"] = 0.015
    args["train"]["update_epochs"] = 2
    args["train"]["minibatch_size"] = 32768

    if load_model_path is not None:
        args["load_model_path"] = load_model_path

    target_score = 0.42
    target_scramble_p = 0.499
    target_min_n = 50.0
    target_streak = 3
    streak = 0

    def stop_on_target(logs):
        nonlocal streak
        p = logs.get("environment/scramble_p")
        score = logs.get("environment/score")
        n = logs.get("environment/n", 0.0)
        if p is None or score is None:
            return False

        hit = p >= target_scramble_p and score >= target_score and n >= target_min_n
        streak = streak + 1 if hit else 0
        if hit:
            print(
                f"target hit: scramble_p={p:.3f} score={score:.3f} n={n:.1f} "
                f"streak={streak}/{target_streak}"
            )

        return streak >= target_streak

    pufferl.train(env_name, args=args, early_stop_fn=stop_on_target)


if __name__ == "__main__":
    train_until_target("puffer_lightsout", load_model_path=None)
    # train_until_target("puffer_lightsout", load_model_path="latest")