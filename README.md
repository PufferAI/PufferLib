You have a CPU-based RL environment but python multiproc is worse than nothing. You can fix this by moving data through pinned CPU memory. That is hard to do, but pufferlib will do it for you. If your env is compatible with PettingZoo or Gymnasium then it is compatible with Pufferlib. You basically just do this:

```py
env = pufferlib.vector.make(
    pufferlib.emulation.GymnasiumPufferEnv(gymnasium.make("CartPole-v1")),
    num_envs=8, num_workers=8, batch_size=1,
)
try:
    env.async_reset() # You can also use the synchronous API with Multiprocessing
    o, r, d, t, i, env_ids, masks = env.recv()
    actions = env.action_space.sample()
    env.send(actions)
    o, r, d, t, i, env_ids, masks = env.recv()
    print('Observations:', o)
finally:
    env.close()
```

[See](examples/gymnasium_env.py) [the](examples/pettingzoo_env.py) [examples](examples/puffer_env.py) [for](examples/pufferl.py) [more](examples/vectorization.py).

It uses large shared-memory buffers so workers write results in place: no pickling, no copies. If you train on GPU, pin the observation buffer and use non_blocking CUDA copies so env-to-GPU transfer overlaps with compute; keep the buffers persistent to avoid allocation churn.

Extras include a fast PPO trainer (V-trace, prioritized minibatches), native C/Cython Ocean environments, drop-in vector backends for CleanRL/SB3, self-play via policy_pool, configs plus sweeps and autotune, and a Docker image (PufferTank). Use what you need and ignore the rest.

![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pufferlib)
![Github Actions](https://github.com/PufferAI/PufferLib/actions/workflows/install.yml/badge.svg)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

PufferLib is the reinforcement learning library I wish existed during my PhD. It started as a compatibility layer to make working with complex environments a breeze. Now, it's a high-performance toolkit for research and industry with optimized parallel simulation, environments that run and train at 1M+ steps/second, and tons of quality of life improvements for practitioners. All our tools are free and open source. We also offer priority service for companies, startups, and labs!

![Trailer](https://github.com/PufferAI/puffer.ai/blob/main/docs/assets/puffer_2.gif?raw=true)

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support -- post here before opening issues. We're always looking for new contributors, too!

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
