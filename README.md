![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

You have an environment, a PyTorch model, and a reinforcement learning framework that are designed to work together but don’t. PufferLib is a wrapper layer that makes RL on complex game environments as simple as RL on Atari. You write a native PyTorch network and a short binding for your environment; PufferLib takes care of the rest.

All of our [Documentation](https://pufferai.github.io "PufferLib Documentation") is hosted by github.io. @jsuarez5341 on [Discord](https://discord.gg/spT4huaGYV) for support -- post here before opening issues. I am also looking for contributors interested in adding bindings for other environments and RL frameworks.

The current `demo.py` file has a powerful `--help` that generates options based on the specified environment and policy. A few examples:

```
python demo.py --env minigrid --mode train --vec multiprocessing
python demo.py --env squared --mode eval --baseline
python demo.py --env nmmo --mode eval
```

Hyperparams are in `config.yaml`.

Star to power up the next release!
## Star History

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
