#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <group>"
    exit 1
fi

GROUP=$1
COMMON="python demo.py --mode train --baseline --env"

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
    grid_continuous)
        $COMMON grid_continuous --env.task foraging --vec multiprocessing
        $COMMON grid_continuous --env.task predator_prey --vec multiprocessing
        $COMMON grid_continuous --env.task group --vec multiprocessing
        $COMMON grid_continuous --env.task puffer --vec multiprocessing
        $COMMON grid_continuous --env.task center --vec multiprocessing
        ;;
    procgen)
        $COMMON bigfish
        $COMMON bossfight
        $COMMON caveflyer
        $COMMON chaser
        $COMMON climber
        $COMMON coinrun
        $COMMON dodgeball
        $COMMON fruitbot
        $COMMON heist
        $COMMON jumper
        $COMMON leaper
        $COMMON maze
        $COMMON miner
        $COMMON ninja
        $COMMON plunder
        $COMMON starpilot
        ;;
    atari)
        $COMMON pong --vec multiprocessing
        $COMMON breakout --vec multiprocessing
        $COMMON beam-rider --vec multiprocessing
        $COMMON enduro --vec multiprocessing
        $COMMON qbert --vec multiprocessing
        $COMMON space-invaders --vec multiprocessing
        $COMMON seaquest --vec multiprocessing
        ;;
    pokemon)
        $COMMON pokemon --vec multiprocessing
        ;;
    crafter)
        $COMMON crafter --vec multiprocessing
        ;;
    nethack)
        $COMMON nethack --vec multiprocessing
        ;;
    nmmo)
        $COMMON nmmo --vec multiprocessing
        ;;
    *)
        echo "Invalid group. Please check this script for valid groups."
        exit 1
        ;;
esac
