#### BOXOBAN

A simple game based on Sokoban where the player must push all boxes on to the targets

Boxoban-levels contains the levels for the game as .txt files. There are various difficulties chosen by the environment variable 'difficulty' which can be 'basic', 'easy', 'medium', 'hard', 'unfiltered'.

Basic - only externals walls and one box 

Easy - only externals walls and up to 4 boxes

These can both be generated using the generate_easy_maps.py script and settings the internals to required options and output str.

The hard, medium and unfiltered levels are taken from Googles Boxoban dataset and the license info is included in the file.
These maps are not easy to generate since they need to be solveable but also interesting, however there are a very good number of maps in those folders ~1M. 

Medium and ulfiltered also have validation sets though these aren't used.

<img width="315" height="342" alt="image" src="https://github.com/user-attachments/assets/f5ea4eac-ec64-4444-b54a-b06c9ef2d252" />





