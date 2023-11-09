from pdb import set_trace as T
import numpy as np

import cv2
from matplotlib import pyplot as plt

map_offsets = {
    # https://bulbapedia.bulbagarden.net/wiki/List_of_locations_by_index_number_(Generation_I)
    0: np.array([0,0]), # pallet town
    1: np.array([-10, 72]), # viridian
    2: np.array([-10, 180]), # pewter
    12: np.array([0, 36]), # route 1
    13: np.array([0, 144]), # route 2
    14: np.array([30, 172]), # Route 3
    15: np.array([80, 190]), #Route 4
    33: np.array([-50, 64]), # route 22
    37: np.array([-9, 2]), # red house first
    38: np.array([-9, 25-32]), # red house second
    39: np.array([9+12, 2]), # blues house
    40: np.array([25-4, -6]), # oaks lab
    41: np.array([30, 47]), # Pokémon Center (Viridian City)
    42: np.array([30, 55]), # Poké Mart (Viridian City)
    43: np.array([30, 72]), # School (Viridian City)
    44: np.array([30, 64]), # House 1 (Viridian City)
    47: np.array([21,136]), # Gate (Viridian City/Pewter City) (Route 2)
    49: np.array([21,108]), # Gate (Route 2)
    50: np.array([21,108]), # Gate (Route 2/Viridian Forest) (Route 2)
    51: np.array([-35, 137]), # viridian forest
    52: np.array([-10, 189]), # Pewter Museum (floor 1)
    53: np.array([-10, 198]), # Pewter Museum (floor 2)
    54: np.array([-21, 169]), #Pokémon Gym (Pewter City)
    55: np.array([-19, 177]), #House with disobedient Nidoran♂ (Pewter City)
    56: np.array([-30, 163]), #Poké Mart (Pewter City)
    57: np.array([-19, 177]), #House with two Trainers (Pewter City)
    58: np.array([-25, 154]), # Pokémon Center (Pewter City)
    59: np.array([83, 227]), # Mt. Moon (Route 3 entrance)
    60: np.array([123, 227]), # Mt. Moon
    61: np.array([152, 227]), # Mt. Moon
    68: np.array([65, 190]), # Pokémon Center (Route 4)
    #193: None # Badges check gate (Route 22)
}

map_locations = {
    0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
    1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
    2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
    3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
    62: {"name": "Invaded house (Cerulean City)", "coordinates": np.array([290, 227])},
    63: {"name": "trade house (Cerulean City)", "coordinates": np.array([290, 212])},
    64: {"name": "Pokémon Center (Cerulean City)", "coordinates": np.array([290, 197])},
    65: {"name": "Pokémon Gym (Cerulean City)", "coordinates": np.array([290, 182])},
    66: {"name": "Bike Shop (Cerulean City)", "coordinates": np.array([290, 167])},
    67: {"name": "Poké Mart (Cerulean City)", "coordinates": np.array([290, 152])},
    35: {"name": "Route 24", "coordinates": np.array([250, 235])},
    36: {"name": "Route 25", "coordinates": np.array([270, 267])},
    12: {"name": "Route 1", "coordinates": np.array([70, 43])},
    13: {"name": "Route 2", "coordinates": np.array([70, 151])},
    14: {"name": "Route 3", "coordinates": np.array([100, 179])},
    15: {"name": "Route 4", "coordinates": np.array([150, 197])},
    33: {"name": "Route 22", "coordinates": np.array([20, 71])},
    37: {"name": "Red house first", "coordinates": np.array([61, 9])},
    38: {"name": "Red house second", "coordinates": np.array([61, 0])},
    39: {"name": "Blues house", "coordinates": np.array([91, 9])},
    40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
    41: {"name": "Pokémon Center (Viridian City)", "coordinates": np.array([100, 54])},
    42: {"name": "Poké Mart (Viridian City)", "coordinates": np.array([100, 62])},
    43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
    44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
    47: {"name": "Gate (Viridian City/Pewter City) (Route 2)", "coordinates": np.array([91,143])},
    49: {"name": "Gate (Route 2)", "coordinates": np.array([91,115])},
    50: {"name": "Gate (Route 2/Viridian Forest) (Route 2)", "coordinates": np.array([91,115])},
    51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
    52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
    53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
    54: {"name": "Pokémon Gym (Pewter City)", "coordinates": np.array([49, 176])},
    55: {"name": "House with disobedient Nidoran♂ (Pewter City)", "coordinates": np.array([51, 184])},
    56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
    57: {"name": "House with two Trainers (Pewter City)", "coordinates": np.array([51, 184])},
    58: {"name": "Pokémon Center (Pewter City)", "coordinates": np.array([45, 161])},
    59: {"name": "Mt. Moon (Route 3 entrance)", "coordinates": np.array([153, 234])},
    60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
    61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
    68: {"name": "Pokémon Center (Route 3)", "coordinates": np.array([135, 197])},
    193: {"name": "Badges check gate (Route 22)", "coordinates": np.array([0, 87])}, # TODO this coord is guessed, needs to be updated
    230: {"name": "Badge Man House (Cerulean City)", "coordinates": np.array([290, 137])}
}

bg = cv2.imread('full_map.png')#[::16, ::16]

counts = np.load('session_1ed2ecca/counts_2_4b06783a.png.npy')
#counts = cv2.imread('session_0b3458f6/counts_2_f39b732e.png')
counts = np.kron(counts, np.ones((16, 16), dtype=np.uint8)).astype(np.uint8)
x_pad = 16*16
y_pad = 16*13
counts = np.pad(counts, ((0, y_pad+8), (0, x_pad)))
counts = counts[y_pad+8:, x_pad:]
#counts = cv2.resize(counts, (256*16, 256*16), cv2.INTER_NEAREST)
#counts = counts[-4000:, -4000:]
#counts = counts[:4000//16, :4000//16]

mmax = np.max(counts)
counts[counts>0.1] = 50*(counts[counts>0]/mmax) + 50

bg[:, :, 0] += counts
#bg[:, :, 1] -= counts
bg = np.clip(bg, 0, 255).astype(np.uint8)

'''
map_offsets = {
    0: np.array([0,0]), # pallet town
    1: np.array([-10, 72]), # viridian
    2: np.array([-10, 180]), # pewter
    12: np.array([0, 36]), # route 1
    13: np.array([0, 144]), # route 2
    14: np.array([30, 172]), # Route 3
}
 
for val in map_locations.values():
    x, y = val['coordinates']
    x = x - 16
    y = y + 13
    bg[-16*(y+1) - 1:-16*y - 1, 16*x:16*(x+1), 0] = 255
    bg[-16*(y+1) - 1:-16*y - 1, 16*x:16*(x+1), 1] = 0
    bg[-16*(y+1) - 1:-16*y - 1, 16*x:16*(x+1), 2] = 0
'''

#bg = cv2.resize(bg, (1000, 1000))
plt.imsave('test_overlay.jpeg', bg)
pass
