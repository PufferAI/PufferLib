![figure](https://pufferai.github.io/source/resource/header.png)

[![PyPI version](https://badge.fury.io/py/pufferlib.svg)](https://badge.fury.io/py/pufferlib)
[![](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341)](https://twitter.com/jsuarez5341)

# PufferLib 1.0 

These are the example scripts used in my presentation of PufferLib 1.0. If you find this project at all interesting or useful, please star pufferai/pufferlib. I am working on this full-time, and it helps out a ton.

I am running these demos on a desktop with an i9 14900k, 128GB RAM, and an RTX 4090. This is more or less our initial target hardware - high-end personal computing available to research labs or enthusiatic hobbyists. For small networks, you can downgrade the GPU substantially without hurting perf too much. Or just run for $0.40 an hour on a vast instance.

NOTE: There is a weird torch bug where sometime torch.tensor(obs) (which copies) is slow and sometimes torch.from_numpy(obs) is slow. Depends on version and background processes. I tried to use the fastest one for each implementation, but it probably depends on hardware and the alignment of the stars.

# Emulation

PufferLib started with the recognition that most of the infra problems in RL come from structured observation and action data, or from fallout thereof. Libraries have to add tons of complex data wrangling code at every stage of processing as a result - in the simulation API, in vectorization, in the model, and even in data storage structures used for training. PufferLib provides a much simpler solution: flatten all of the data so we only have to handle the simple case, thereby *emulating* an Atari-like interface (anything that works with Atari will work for your env). 

You can't train on flat data properly, but we can unpack back to the original structure just-in-time before it goes into the forward pass. Vectorization, experience buffers, etc. do not need to know or care about your data. We can handle nested dicts, mixed data types and sizes, and more with this approach.

```demo_emulation.py```

Hit c on each breakpoint to continue. Now, let's see why this is so useful.

```ppo_minihack.py```

It's just CleanRL's PPO using minihack. You will get an error on the tensor storage. Frankly, I'm surprised it even gets that far (it's pretty easy to break Gym's structured data processing). Now, you have to rewrite half of this file to handle the specific storage structure of Minihack. You won't be able to use the same code for any other env.

Or, you can wrap Minihack with PufferLib and run the exact same code. Once you put a Minihack model in, it will just work.

```puffer_ppo_minihack.py```

We didn't have to introduce any crazy heavy abstractions either. It's just a data flatten/unflatten. The one small bit of jank is that pytorch doesn't let you .view with a numpy datatype. I'm hoping they will add support for this in the future, but for now, we have our own implementation. It's messier than it needs to be because we are trying to not break torch.compile. But it works well for now!

Next, we can add in PufferLib's vectorization.

```puffer_vec_ppo_minihack.py```

Much faster! Why? Read on.

# Vectorization

PufferLib's emulation unlocks a ton of potential for downstream optimizations. We no longer have to care about structured data at all. But first, let's do some optimization without even touching pufferlib to get a fair baseline. 


In each of the following demos, you can diff against vanilla_ppo_atari to see what I changed.

```ppo_atari.py```

I just changed their vec to async because by default, it runs on a single core. Not fair to CleanRL. 

```large_ppo_atari.py```

I just added more envs to the vectorization. Academia has this weird obsession over sample efficiency when there are a lot of cases where the simulator is really fast. We're not indulging that particular debate today. In any event, this is now a decently fair baseline. 

Before we move on, let me make something clear. CleanRL is the best thing since sliced bread. It has my vote for highest-impact overall RL contribution. Seriously, Costa is cracked. The takeaway from this should not be that base cleanrl is slow. Stuff like SB3 isn't any faster. The only libraries you are going to find that are way faster by default are things like SampleFactory or Moolib. These projects are awesome but not a fair comparison to PufferLib. Just read the code to see why. I'm trying to make RL as fast and flexible as possible while keeping it dead simple. They are trying to make RL as fast as possible at any cost - including a really fancy stack. 

With that out of the way, let's make this really fast. First, let's check out PufferLib's vectorization performance. We have small and large batch baselines.

```puffer_ppo_atari.py```
```large_puffer_ppo_atari.py```

Much faster! PufferLib's multiprocessing has a bunch of optimizations even in the simplest case:
- Pipes over queues. Python Queues are slow.
- Multiple envs per process. This reduces infighting among processes. It's also how you get vectorization speedups for even really fast environments (100k+ sps/core)
- Shared memory. Gymnasium has this too, but they try to handle structured data at the same time. Ours is one big buffer of contiguous memory
- Zero-copy. We never have to stack data from different processes. It's all written into the same numpy-backed shared memory array
- Busy-waiting workers. We don't use pipes or queues for communicating status. Workers busy-wait on a value in unlocked shared memory. The only place we pipe any data at all is for environment infos.
- Pruning env infos. We have wrappers that aggregate infos over a full episode. If info is empty on any given step, we don't send it.

I see the basic vectorization as one of the most critical components of RL. So I spent a lot of time optimizing it. There's still a way to go, but this implementation already benchmarks much better than Gymnasium/SB3 on nearly every env I tested.

Oh, and did I mention **native multiagent support**? That's right, single-agent and multi-agent look the same to this implementation. Getting distributed multi-agent working at all is typically a huge pain for researchers.

# Envpool

If you haven't already heard of it, envpool is this awesome project that runs move envs than you need per batch. So while your forward pass is running, your CPU cores are computing the next batch of data. And it returns the first envs that finish stepping, so if one env is slow or is resetting, you don't bottleneck on it. This is a huge optimization, 2-3x throughput for many projects.

There are two issues with (official) envpool:
- It only supports specific cpp environments
- It returns envs out of order, so you have to change your experience buffer

```pip install envpool```
```ppo_atari_envpool.py```

This shows the default cleanrl example. It uses it in sync mode, so you don't get much out of it.

PufferLib provides a highly optimized Python implementation of envpool that works with any Pufferlib compatible environment, single-agent and multi-agent. There isn't a good way to solve the second issue though, so this is where we have to move over to the main PufferLib demo.py script. We have a custom cleanrl script over there that we use for all of our testing. It's a bit longer, but it includes: lstm/flat networks, envpool experience buffer, local dashboard, wandb sweeps integration, and much more!
