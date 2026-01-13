# Learning TODO: RL Foundations

Everything you need to understand `bptt_horizon` and RL training in general.

---

## Level 1: Basic ML Concepts

### 1.1 What is a Neural Network?
- Function that takes numbers in, spits numbers out
- Has "weights" (parameters) that get adjusted during training
- `input → [neural network] → output`

### 1.2 What is Training / Learning?
- Adjusting weights so the network gives better outputs
- Done by computing "loss" (how wrong it was) and updating weights to reduce loss

### 1.3 What is Backpropagation?
- Algorithm to figure out HOW to adjust each weight
- Flows backwards through the network: output → hidden layers → input
- "If the output was wrong, which weights were responsible?"

### 1.4 What is a Batch?
- Group of training examples processed together
- Instead of: train on example 1, then example 2, then example 3...
- Do: train on [example 1, 2, 3, 4, 5] at once
- Why? Faster (GPU parallelism) + more stable learning

### 1.5 What is Minibatch?
- When your batch is too big for GPU memory
- Split batch into smaller "minibatches"
- `batch_size = 1024, minibatch_size = 256` → 4 gradient updates per batch

---

## Level 2: RL Basics

### 2.1 What is a Timestep?
- One tick of the game/simulation
- Agent observes state → takes action → gets reward → new state
- `t=0: see game → press button → get +1 point → game changes`

### 2.2 What is an Episode?
- One complete playthrough from start to end
- Boss fight: episode = one full fight (win or lose)
- `[spawn] → step → step → step → ... → [death or victory]`

```
Episode 1: t0 → t1 → t2 → t3 → DEAD (4 steps)
Episode 2: t0 → t1 → t2 → t3 → t4 → t5 → WIN (6 steps)
```

### 2.3 What is an Observation?
- What the agent "sees" at each timestep
- Your boss_fight: 14 numbers (player pos, boss HP, etc.)

### 2.4 What is a Policy?
- The neural network that decides actions
- `observation (14 floats) → [policy network] → action (0-6)`
- Training = making this network choose better actions

### 2.5 What is a Value Function?
- Predicts "how good is this situation?"
- "I have full HP, boss is low" → high value
- "I'm almost dead, boss is full HP" → low value
- Helps the agent learn which states to aim for

---

## Level 3: How RL Training Works

### 3.1 Collect Experience
```
Run 56 environments in parallel:
  Env 1: obs → action → reward → obs → action → reward → ...
  Env 2: obs → action → reward → obs → action → reward → ...
  ...
  Env 56: obs → action → reward → obs → action → reward → ...

After N steps, you have a "batch" of experience
```

### 3.2 Compute Advantages
- "Was this action better or worse than expected?"
- `advantage = actual_reward - predicted_value`
- Positive advantage → reinforce this action
- Negative advantage → discourage this action

### 3.3 Update the Network
- Use collected experience to adjust policy weights
- Make good actions more likely, bad actions less likely

### 3.4 Repeat
```
while not done:
    1. Collect batch of experience (many timesteps)
    2. Compute advantages
    3. Update network with minibatches
    4. Go to 1
```

---

## Level 4: Sequential Data & Memory

### 4.1 Why Sequence Matters
In games, the PAST affects what you should do NOW:

```
Timestep 1: Boss starts wind-up animation
Timestep 2: Boss still winding up
Timestep 3: Boss about to attack!     ← YOU SHOULD DODGE NOW
Timestep 4: Boss attacks

If you only see timestep 3 in isolation, you might not know to dodge.
But if you saw timesteps 1-2-3 together, you'd see the pattern.
```

### 4.2 MLP (Multi-Layer Perceptron) — No Memory
- Standard neural network
- Only sees CURRENT observation
- `obs_t → [MLP] → action`
- No memory of previous timesteps
- Fine if observation contains all needed info

### 4.3 RNN (Recurrent Neural Network) — Has Memory
- Sees current observation + remembers past
- `obs_t + memory → [RNN] → action + updated_memory`
- Can learn patterns over time
- Types: LSTM, GRU (different memory mechanisms)

```
MLP:  sees [___] [___] [_X_]     ← only current frame
RNN:  sees [_X_] [_X_] [_X_]     ← current + memory of past
```

### 4.4 When Do You Need RNN?
- When current observation is INCOMPLETE
- Example: "Boss is standing still" — is he about to attack or recovering?
- If your observation includes `boss_phase` and `time_to_damage`, MLP might be enough
- If observation only has positions, RNN helps learn timing

---

## Level 5: BPTT (Backpropagation Through Time)

### 5.1 The Problem
RNN has memory that flows through time:

```
t1 → t2 → t3 → t4 → t5 → t6 → ... → t1000

To train RNN, backprop must flow backwards through ALL these connections.
1000 timesteps = 1000 layers of backprop = VERY slow, uses tons of memory
```

### 5.2 The Solution: Truncated BPTT
Don't backprop through entire episode. Cut it into chunks:

```
Episode:     [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12]

bptt_horizon = 4:

Chunk 1: [t1 → t2 → t3 → t4]   ← backprop only through these 4
Chunk 2: [t5 → t6 → t7 → t8]   ← backprop only through these 4
Chunk 3: [t9 → t10 → t11 → t12] ← backprop only through these 4
```

### 5.3 What bptt_horizon Controls
```
bptt_horizon = 16 means:
- RNN sees 16 consecutive timesteps during training
- Gradients flow back through 16 steps max
- RNN can learn patterns up to ~16 steps long
```

### 5.4 Trade-offs
```
Small horizon (8):
  ✓ Fast, low memory
  ✗ RNN can't learn long patterns (>8 steps)

Large horizon (128):
  ✓ RNN learns longer patterns
  ✗ Slow, high memory usage
```

---

## Level 6: Putting It Together

### 6.1 The Batch Math
```
num_envs = 56        (parallel environments)
bptt_horizon = 16    (timesteps per chunk)

batch_size = num_envs × bptt_horizon
           = 56 × 16
           = 896 total samples per training batch
```

### 6.2 Why minibatch_size Must Be ≤ batch_size
```
batch_size = 896     (you collected 896 samples)
minibatch_size = 2048 (you want to train on 2048 at a time)

ERROR: Can't take 2048 samples from a pile of 896!

Fix: minibatch_size = 256 or 512 (smaller than 896)
```

### 6.3 For Your Boss Fight (No RNN)
You're using MLP, so `bptt_horizon` just affects batch math:

```ini
[vec]
num_envs = 56

[train]
bptt_horizon = 16        # 56 × 16 = 896 batch
minibatch_size = 256     # Must be ≤ 896
```

Or increase horizon if you want bigger batches:

```ini
bptt_horizon = 64        # 56 × 64 = 3584 batch
minibatch_size = 2048    # Now this works
```

---

## Summary: What You Actually Need to Know

1. **batch_size** = total samples collected before training
2. **minibatch_size** = chunk size for each gradient update (must be ≤ batch_size)
3. **bptt_horizon** = consecutive timesteps kept together
   - For RNN: determines how far back it can learn patterns
   - For MLP: just affects batch_size math
4. **Your boss_fight uses MLP** — bptt_horizon is just a number to make the math work

---

## Learning Resources

### Videos (start here)
- [ ] 3Blue1Brown: "Neural Networks" series (YouTube)
- [ ] Mutual Information: "Reinforcement Learning" series (YouTube)

### Interactive
- [ ] Andrej Karpathy: "Neural Networks: Zero to Hero" (YouTube + code)

### Reading
- [ ] Spinning Up in Deep RL (OpenAI) — https://spinningup.openai.com
- [ ] CleanRL documentation — similar to PufferLib

### Hands-on
- [ ] Train boss_fight, watch the numbers, build intuition
