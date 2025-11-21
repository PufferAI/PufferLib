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

# Getting started
PufferLib enables you to build, train and evaluate extremely fast reinforcement learning evnironments. This guide provides defaults specific to Ubuntu but you can adjust them to any other OS.

### Installing required software
This project makes heavy use of C to speed up operations in Python so packages for building native exensions are required. 
```bash
sudo apt update && sudo apt install -y git curl software-properties-common build-essential python3-dev
```

Make sure that you have all of your Nvidia drivers configured correctly so that your GPU can be used for accelerating the RL training. One important thing is having NVCC which is a cuda compiler, installed to enable better performance by compiling some pufferlib speciic kernels but this is optional. You can check if nvcc is installed by running.
```bash
nvcc -V
```

If nvcc is missing or you are missing some nvidia drivers you can install them using the command below. Here are [alternative installtion instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for the Cuda Toolkit.
```bash
sudo apt install nvidia-cuda-toolkit
```

UV is the prefferred package manager for this project but you are free to just use pip and suffer. [alternative installtion instructions](https://docs.astral.sh/uv/getting-started/installation/)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The preferred compiler toolchain for PufferLib is the latest stable version of clang. You can also download the executable directly from [LLVM releases page](https://releases.llvm.org/) 
```bash
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
```

On Ubuntu clang is installed as under clang-<version number> but this project expects the executable to be named clang so it's best to introduce and alias in ~/.bashrc or another file that is preloaded in your preferred shell. 
```bash
alias clang="clang-20"
```
and reload the file
```bash
source ~/.bashrc
```

### Running your first Reinforcement Learning environemnts
Now that you have all of the required software

```bash
git clone git@github.com:PufferAI/PufferLib.git && cd PufferLib
```

Install the local packages. This installs the appropriate version of [Raylib](https://www.raylib.com/) which is a minimalistic library for building video games in C and [Box2D](https://box2d.org/) which is physics engine for 2D games.
```bash
uv pip install -e . 
```

Now you can compile the first RL environemnt. The build_ocean.sh script is used for building Ocean RL environements which is a PufferLib native framework. "target" is the name of the environemnt. You can view all of the environment files as well as other Ocean environments at pufferlib/ocean/target. The environment is configured by a .ini config file which specifies the name, RL policy and training configuration, the one for the target env is located at config/ocean/target.ini. "local" is the type of the build. Local builds contain debug symbols, use an address sanitizer and allow you to verify that the environment works as you intend, a production version can be compiled using the "fast" build. You can also build a web version of the env which will generate a Web Assembly page. 


```bash
scripts/build_ocean.sh target local
```

and then you can run the created executable which demonstrates the environment.
```bash
./target
```

If you see pufferfishes chasing the stars it means that everything works correctly. This demo loads a neural net that has already been trained before. This is why the fish chase the stars rather than bounce around aimlessly which they would do if the weights were selected at random.

### Training your first Reinforcement Learning neural net
In order to train the env we need to use the puffer train command and use the env_name from the config, rather than the file name based one that was used for building.
```bash
puffer train puffer_target
```

After you begin training you should see how over time the policy_loss begins decreasing. This means that the fish are getting better at obtaining the reward which comes from eating the stars. Another observation is that the episode_length decreases which means that over time the fish eat all of the stars faster because they get better at runningto ther closest one. One final observation is that the explained_variance is increasing which means that the policy is responsible for a higher fraction of the variance in the environment. In other words the situation in the environment becomes less random and more dependent on the trained policy because the fish get better at following the stars. 

Now you can export your learned weights so that they can be used in the environment demo. This command exports the latest version by default from the specified environemnt from the experiments/ folder. If you wish to export a specific one use "--load_model_path" option.

```bash
puffer export puffer_target
```

This should generate a puffer_target_weights.bin file which contains all of the learned weights for the neural net for this environment. Now you can see how these weights behave in real life. You need to edit the load path for the weights for the demo at pufferlib/ocean/target/target.c at line 21 from "resources/target/target_weights.bin" to "puffer_target_weights.bin" which you trained. After recompiling the evironment you should notice that the fish are somewhat dumber than the ones which come by default for this env. You can now try change the config (e.g. training for longer or changing other training params) and see how that impacts the behavior. 

Just as a side note the pufferlib/ocean/target/target.c is just a demonstration that allows you to see how your weights behave, it's not required for training. All of the code that is required to train the env is located in pufferlib/ocean/target/target.h (the C code) pufferlib/ocean/target/binding.c (some C code exposed to Python) and pufferlib/ocean/target/target.py (Python env that calls functions from binding.c and target.h)
