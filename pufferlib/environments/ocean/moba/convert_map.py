import numpy as np
from PIL import Image
game_map = np.array(Image.open('dota_map.png'))[:, :, -1]
game_map = game_map[::2, ::2][1:-1, 1:-1]
game_map[game_map == 0] = 1
game_map[game_map == 255] = 0

# Precompute all ai pathfinding
from c_precompute_pathing import precompute_pathing
ai_paths = np.asarray(precompute_pathing(game_map))
breakpoint()
ai_paths = ai_paths.astype(np.uint8).ravel()
ai_paths.tofile('ai_paths.npy')

# Save game map
game_map = game_map.ravel()
game_map = game_map.astype(np.uint8)
game_map.tofile('game_map.npy')
