#!/bin/bash
python demo.py --backend clean_pufferl --config pokemon_red --no-render --vectorization multiprocessing --mode train --track --wandb-entity xinpw8 # --exp-name pokegym_test_pufferbox3_BET_2 # --exp-name boey_test_pufferbox5_1

# change: double max_episode_steps doesn't work! pokemon don't level up at all; all stayed in pallet or went to battle rival
# increase event_reward from 0.3 to 0.4. Uncomment healing reward (i.e. healing reward = 0). Did not work. Very slow to Pewter; does not even start badge.
# Revert event reward to 0.3. Comment healing reward = 0, i.e. healing reward is back! Whoops - realized healing and seen_pokemon rewards set to 0.
# # 
#         level_reward = 0.01 * level_reward
#         # caught_pokemon_reward = 0 # helps it beat early trainers
#         seen_pokemon_reward = 0
#         # healing_reward = 0
#         event_reward = event_reward * 0.3  CRAPPY (gray)

# zero out all except: 3/11/24

        # level_reward = 0.01 * level_reward
        # # caught_pokemon_reward = 0 # helps it beat early trainers
        # seen_pokemon_reward = 0
        # healing_reward = 0
        # event_reward = event_reward * 0.3









# What is acceptable to the rl community wrt solving pokemon:
#         extensive reward engineering = bits of scripting
#   no penalty for: general exploration method, maybe some entroy-related thing.
# dynamically adjust lr as well

# NOT OKAY: 2000-line rewards thingie made specifically to solve the environment
# order of criticism of dota paper:
# compute
# network + reward engineering
# scripting and stuff

    