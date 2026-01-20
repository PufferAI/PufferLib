# BossFight Reinforcement Learning project

I'm implementing a RL environment using PufferLib in C + Python.

Environment spec file is in `./pufferlib/ocean/boss_fight/README.md`.

You are in PufferLib's (puffer.ai) source repository which contains "Ocean" - a collection of environments.

The environment code I'm working on is located in `./pufferlib/ocean/boss_fight/`. Environment configuration is in `./pufferlib/config/boss_fight.ini`

### Setup

1. Fork pufferlib, create new branch

2. Run these:

```
uv venv
uv pip install -e .
```

3. Setup files using templates, update `environment.py`

4. Not sure what this does yet:

```
python setup.py build_boss_fight --inplace --force
```

### Testing

Make sure shit's running:

```
uv pip install -e .
python -c "
from pufferlib.ocean.boss_fight import BossFight
import numpy as np
env = BossFight(num_envs=2)
env.reset()
for _ in range(100):
    env.step(np.random.randint(0, 7, size=2))
print('ok')
env.close()
"
```

Train and check scores:

```
puffer train puffer_boss_fight --train.total-timesteps 50000
```

## Eval

```
puffer eval puffer_boss_fight --load-model-path $(ls -t experiments/puffer_boss_fight_*/model_*.pt | head -1)
```
