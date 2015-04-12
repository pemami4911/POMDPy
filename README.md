This open-source project contains a framework for implementing discrete or continuous POMDPs in Python. The organization of the code was inspired by [TAPIR](http://robotics.itee.uq.edu.au/~hannakur/dokuwiki/doku.php?id=wiki:tapir) and the [POMCP](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications.html) algorithm.

This is project has been conducted strictly for research purposes. If you would like to contribute to POMDPy or if you have any comments or suggestions, feel free to send me a pull request or send me an email at pemami@ufl.edu.  

## Installation ##
From the root directory of the project, run 
    `python setup.py install`

## Running a sample ##
After tweaking the configuration files found in the config package <optional>, run the RockSample Problem with

    python src/run_pomdp.py

## Dependencies ##

This project uses:
* Python 2.7.9
* numpy 1.9.2
* matplotlib 1.4.3
* pytest 2.7.0
* pygame 1.9.1 

## TODO ##
* Unit Test coverage is currently minimal, so this area is going to be expanded upon soon
* The only current "working" test-problem is RockSample. More test problems are being worked on 
* An extension for GPU-MCTS is being planned. The NumbaPro Python module seems to be a good bet for this
* The PyGame sim is currently not supported, due to recent changes
