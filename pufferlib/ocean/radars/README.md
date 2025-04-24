# Radars

A multi-function, multi-sensor radar environment.

The intention is for this to be a _continuing environment_ for evaluating reinforcement learning algorithms.
For now, it will reset and such to play nice with pufferlib.

## Context

This is loosely based on a scenario where there is an X-Band and S-Band radar with an overlapping field of view.

The S-Band radar has a wider field of view and longer range, but lower probability of detection.
The X-Band radar has a narrower field of view and shorter range, but higher probability of detection.

Targets resemble aircraft following singer trajectories.

The action space is discrete and contains one action for each tracker, plus an additional 'search' action. 

Searching is done by broadcasting a beam in the least recently observed direction.
Tracking is done by broadcasting a beam in the direction of the target.

Penalties are incurred for delaying a searcher or tracker.
Sufficiently delayed trackers are lost, and incur an additional penalty.

## TODO

- [ ] Implement a multi-sensor EST baseline
- [ ] Force search at 30?
- [ ] Massive Observation Space needs to be normalized
- [ ] grab logging stuff from pong
- [x] update clang and use f-sanatize flags
- [ ] table of all constants

