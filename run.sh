#!/bin/bash
# python demo.py --config pokemon_red --vectorization multiprocessing  --mode train --track
# python demo.py --config pokemon_red --vectorization multiprocessing  --mode train
# python demo.py --config pokemon_red --mode train
python demo.py --backend clean_pufferl --config pokemon_red --vectorization multiprocessing  --mode train --track
