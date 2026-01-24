# BossFight Reinforcement Learning project

I'm implementing a RL environment using PufferLib in C + Python.

It's a **minimal** 2D boss fight environment to learn RL concepts with PufferLib.
Focus: **observation design, reward shaping, training experiments and a bit of game dev using Raylib**

The boss has **1 attack** (AOE burst). All hitboxes are circles (collision = circles overlap).

You are in PufferLib's (puffer.ai) source repository which contains "Ocean" - a collection of environments.

The environment code I'm working on is located in `./pufferlib/ocean/boss_fight/`. Environment configuration is in `./pufferlib/config/boss_fight.ini`

After modifying C files, to test you can run:

```
python setup.py build_boss_fight --inplace --force && puffer train puffer_boss_fight --train.device cpu --vec.num-workers 8 --vec.num-envs 1024 --train.total-timesteps 5000000 && puffer eval puffer_boss_fight --load-model-path $(ls -t experiments/puffer_boss_fight_*/model_*.pt | head -1)
```
