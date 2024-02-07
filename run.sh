#!/bin/bash
python demo.py --config pokemon_red --vectorization multiprocessing --mode train --track
# python demo.py --config pokemon_red --vectorization multiprocessing  --mode train
# python -m pdb demo.py --config pokemon_red --vectorization multiprocessing  --mode train
# python demo.py --config pokemon_red --vectorization serial --mode train