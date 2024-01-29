#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <group>"
    exit 1
fi

GROUP=$1
COMMON="python demo.py --mode train --baseline --config"

case $GROUP in
    ocean)
        $COMMON squared
        $COMMON password
        $COMMON stochastic
        $COMMON memory
        $COMMON multiagent
        $COMMON spaces
        $COMMON bandit
        ;;
    procgen)
        $COMMON bigfish
        $COMMON ninja
        $COMMON miner
        ;;
    atari)
        $COMMON breakout --vectorization multiprocessing
        $COMMON pong --vectorization multiprocessing
        $COMMON beam-rider --vectorization multiprocessing
        $COMMON enduro --vectorization multiprocessing
        $COMMON qbert --vectorization multiprocessing
        $COMMON space-invaders --vectorization multiprocessing
        $COMMON seaquest --vectorization multiprocessing
        ;;
    pokemon)
        $COMMON pokemon --vectorization multiprocessing
        ;;
    crafter)
        $COMMON crafter --vectorization multiprocessing
        ;;
    nethack)
        $COMMON nethack --vectorization multiprocessing
        ;;
    nmmo)
        $COMMON nmmo --vectorization multiprocessing
        ;;
    *)
        echo "Invalid group. Please specify 'ocean' or 'procgen_small'."
        exit 1
        ;;
esac
