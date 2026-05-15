#### BOXOBAN

A simple game based on Sokoban where the player must push all boxes on to the targets

Boxoban-levels contains the levels for the game as .txt files. There are various difficulties chosen by the environment variable 'difficulty' which can be 'basic', 'easy', 'medium', 'hard', 'unfiltered'.

Basic - only externals walls and one box 

Easy - only externals walls and up to 4 boxes

Easy and basic maps generate .txt files on first use.
Other difficulties download .txt files from a remote repo on first use.
All dificulties then generate a .bin after which .txt can be deleted.

The hard, medium and unfiltered levels are taken from Googles Boxoban dataset and the license info is included in the repo as well as the credit for citing.
These maps are not easy to generate since they need to be solveable but also interesting, however there are a very good number of maps in those folders ~1M. 

Medium and unlfiltered also have validation sets though these aren't used.


## The first time each difficulty is used a .bin is generated

Play manually using the .c compiled with bash scripts/build_ocean boxoban. 

Sprites included and have an creative license.

You can play different difficulties by adding the arg eg. ./boxoban easy HOWEVER the .bin needs to have been built

<img width="315" height="342" alt="image" src="https://github.com/user-attachments/assets/f5ea4eac-ec64-4444-b54a-b06c9ef2d252" />





