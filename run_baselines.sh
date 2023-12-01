COMMON="--train --track --wandb-group baselines"

python demo.py $COMMON --env classic_control
python demo.py $COMMON --env squared
python demo.py $COMMON --env atari --env-kwargs.name BreakoutNoFrameskip-v4
python demo.py $COMMON --env atari --env-kwargs.name PongNoFrameskip-v4
python demo.py $COMMON --env atari --env-kwargs.name BeamRiderNoFrameskip-v4
python demo.py $COMMON --env atari --env-kwargs.name EnduroNoFrameskip-v4
python demo.py $COMMON --env atari --env-kwargs.name QbertNoFrameskip-v4
python demo.py $COMMON --env atari --env-kwargs.name SpaceInvadersNoFrameskip-v4
python demo.py $COMMON --env atari --env-kwargs.name SeaquestNoFrameskip-v4
python demo.py $COMMON --env procgen --env-kwargs.name bigfish
python demo.py $COMMON --env minigrid
python demo.py $COMMON --env nethack
python demo.py $COMMON --env minihack
python demo.py $COMMON --env crafter
python demo.py $COMMON --env pokemon_red
python demo.py $COMMON --env nmmo
