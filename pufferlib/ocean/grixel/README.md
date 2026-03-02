## Grixel environment

This is a pixel-based version of the "grid" environment, that is, a gridworld with pixel-based inputs (as in Crafter / Craftax). We use the pixel-based inputs, and the large space of visual stimuli they allow, to implement a very simple meta-learning experiment, based on visual memory.

Each world is a maze (with added gaps at random position to make movement easier, since maze-solving is no the primary purpose of the environment). In addition to the agent, there are two types of moving objects (or "mobs"), namely "rewards" and "zombies". When hitting a mob, the agent receives a reward (positive or negative) and is randomly teleported. Currently all mobs move randomly.

There is a also a "neutral" type of object, which can be picked and dropped by the agent (picking is simply by moving onto it, dropping is a dedicated action). Currently this has no effect at all.

The visual input to the agent is a local portion of the pixel map, or size 11 x 11 x block_size x block_size. 11x11 is inherited from the "grid" environment as the visual input diameter over the gridworld, and block_size (default 5) is the number of pixels in the height/width of each block in the grid.

All objects are represented by binary textures of size block_size x block_size. The exact visual appearance of all objects is governed by the "texture_mode" parameter in the "env" section of the configuration:

- texture_mode=0: the reward and the zombie each have a fixed, unchanging appearance across episodes
- texture_mode=1: the reward and the zombie randomly swap their appearance for each episode (there are still only two possible appearances in total)
- texture_mode=2: the reward and the zombie have completely random appearance, that is, each of them is assigned a random binary texture for each episode. 

In modes 1 and 2, the agent must learn anew which of the two mobs is the reward or the zombie, from experience. This is the meta-learning aspect of the experiment.

Crucially, the agent can also perceive previous-step reward as part of its input; this is required for meta-learning.

The encoder is a CNN where the input layer has both kernel size and stride equal to block_size: the first convolution thus separately maps each block of the gridworld into a single vector. 

The experiment works with the standard LSTM from PufferLib's Recurrent model. We also implemented a transformer and a plastic LSTM, with the plastic LSTM performing best by far in this simple visual memory task. These are not included here as they require modifying the rest of the PufferLib code (though you can see these *highly experimental* implementations [there](https://github.com/ThomasMiconi/PufferLib/blob/grixel/pufferlib/models.py)).

Notably, all episodes have the same lengths, equal to the backpropagation-through-time horizon of the PPO training loop. This avoids difficulties with changing environments and ensures each episode starts with a reset hidden state during training.

This code is provided as is. Everything in this code is experimental and none of it has been thoroughly tested. 

To run the training:

`puffer train puffer_grixel --rnn-name Recurrent --env.texture-mode 2`

To start a visual eval:

`puffer eval  puffer_grixel  --rnn-name Recurrent --load-model-path [checkpoint_file]  --env.texture-mode 2`

