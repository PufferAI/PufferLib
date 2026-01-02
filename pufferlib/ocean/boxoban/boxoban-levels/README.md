# Boxoban

This repository contains levels for boxoban, a box-pushing puzzle game inspired by Sokoban.

If you use this dataset in your work, please cite the following:

## Bibtex

```
@misc{boxobanlevels,
author = {Arthur Guez and Mehdi Mirza and Karol Gregor and Rishabh Kabra and Sebastien Racaniere and Theophane Weber and David Raposo and Adam Santoro and Laurent Orseau and Tom Eccles and Greg Wayne and David Silver and Timothy Lillicrap and Victor Valdes},
title = {An investigation of Model-free planning: boxoban levels},
howpublished= {https://github.com/deepmind/boxoban-levels/},
year = "2018"
}
```

Questions regarding the dataset can be directed to Theophane Weber (theophane@google.com).

## Description

The aim is to push the boxes on top of the targets. Each level has four boxes and four targets. The player can push a box as long as nothing (another box, a wall) is behind it.

Puzzles are grouped into sets of one thousand puzzles, each set encoded as a text file.  Each puzzle is a 10 by 10 ASCII string which uses the following encoding: '\#' for wall, '@' for the player character, '$' for a box, and '.' for a goal position. 

Within a file, puzzles are separated by a semicolon and a puzzle number. There are three levels of difficulties fo puzzles: 'unfiltered', 'medium', and 'hard'.

The unfiltered levels are separated into train (900000 puzzles), validation (100000 puzzles), and test (1000 puzzles). 

Unfiltered puzzles are generated according to the procedure described in 'Imagination-Augmented Agents for Deep Reinforcement Learning' (Racaniere et. al, proceedings of NeurIPS, 2017). The unfiltered test is the one used by Orseau et al. in 'Single-Agent Policy Tree Search With Guarantees' (proceedings of NeurIPS, 2018). 

The medium and hard datasets generation procedure is described in 'An investigation of model-free planning' (Guez, Mirza, Gregor, Kabra et. al, arxiv, 2018). The medium set is composed of train (450000 puzzles) and validation (50000 puzzles); the hard set has 3332 levels.



## Disclaimers

This is not an official Google product.
