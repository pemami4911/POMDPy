![Build](https://travis-ci.org/pemami4911/POMDPy.svg?branch=master)

This open-source project contains a framework for implementing discrete or continuous POMDPs in Python. The organization of the code was inspired by [TAPIR](http://robotics.itee.uq.edu.au/~hannakur/dokuwiki/doku.php?id=wiki:tapir) and the [POMCP](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications.html) algorithm.

[What the heck is a POMDP?](http://www.pomdp.org/tutorial/index.shtml)

Here's David Silver and Joel Veness's paper on POMCP, a ground-breaking POMDP solver. [Monte-Carlo Planning in Large POMDPs](http://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)

This is project has been conducted strictly for research purposes. If you would like to contribute to POMDPy or if you have any comments or suggestions, feel free to send me a pull request or send me an email at pemami@ufl.edu.  

## Installation ##
Download the files as a zip or clone into the repository.
From the root directory of the project, run
 
    python setup.py install

## Running a sample ##
You can optionally tweak the system and RockSample configuration files (`system-config.json` and `rock_problem-config.json`), which are located in `pompdy/config`.
The following maps are available:
* RockSample(7, 8), a 7 x 7 grid with 8 rocks.
* RockSample(11, 11), an 11 x 11 grid with 11 rocks
* RockSample(15, 15), a 15 x 15 grid with 15 rocks

To run the RockSample problem:

    `python run_pomdp.py 1`
    
To run the Tiger sample problem: 

    `python run_pomdp.py 2`
    
See `pompdy/README.md` for more implementation details.

## Running tests ##
Unit tests can be ran with 
    
    py.test test/unit_tests.py
    
## Dependencies ##

This project uses:

* Python 2.7.9
* numpy 1.9.2
* pytest 2.7.0

Optional, for extended functionality:

* matplotlib 1.4.3
* pygame 1.9.1 

## TODO ##
[] Add more unit tests
[] Add additional benchmark problems 
[] Add GPU support (MCTS-GPU)
[] Supply an easy-to-use configuration schema to specify a POMDP and auto-generate the classes
[] Add ways of creating/learning generative models
[x] Clean up output displayed to the user
[] Continuous-time POMDPs? DNN-POMDPs? AIXI?
