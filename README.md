## POMDPy
![Build](https://travis-ci.org/pemami4911/POMDPy.svg?branch=master) ![Python27](https://img.shields.io/badge/python-2.7-blue.svg)  ![Python35](https://img.shields.io/badge/python-3.5-blue.svg)

This open-source project contains a framework for implementing discrete action/state POMDPs in Python.

[What the heck is a POMDP?](http://www.pomdp.org/tutorial/index.shtml)

Here's David Silver and Joel Veness's paper on POMCP, a ground-breaking POMDP solver. [Monte-Carlo Planning in Large POMDPs](http://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)

This project has been conducted strictly for research purposes. If you would like to contribute to POMDPy or if you have any comments or suggestions, feel free to send me a pull request or send me an email at pemami@ufl.edu.  

## Dependencies ##
Download the files as a zip or clone into the repository.

    git clone https://github.com/pemami4911/POMDPy.git

This project uses:

* numpy >= 1.11
* matplotlib >= 1.4.3
* scipy >= 0.15.1
* future >= 0.16
* tensorflow >= 0.12

See the [Tensorflow docs](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#download-and-setup) for information about installing Tensorflow. 

The easiest way to satisfy the dependencies is to use Anaconda. You might have to run `pip install --upgrade future` after installing Anaconda, however. 

## Supported Solvers ##

* [POMCP](https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/pomcp.py)
* [Value Iteration](https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/value_iteration.py)
* [Linear Value Function Approximation](https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/linear_alpha_net.py)

### POMCP

POMCP uses the off-policy Q-Learning algorithm and the UCT action-selection strategy. It is an anytime planner that approximates the action-value
estimates of the current belief via Monte-Carlo simulations before taking a step. This is known as Monte-Carlo Tree Search (MCTS).

To solve a POMDP with POMCP, the following classes should be implemented:

* Discrete Action
* Discrete State
* Discrete Observation
* Discrete/Enumerated ActionPool
* Model - this module is the most important, since it acts as the black-box generator
    for (S', A, O, R) steps. This also encodes the belief update rule for the particle filter.

    You may want to to provide a .txt of .cfg containing a map or other data that encapsulate
    the environment and hence the transition probabilities for the world which the POMDP lives in.

#### Heirarchy of nodes in the belief tree from a parent `BeliefNode` to a child `BeliefNode`:

(parent) BeliefNode -> ActionMapping -> ActionMappingEntry -> ActionNode -> ObservationMap -> ObservationMappingEntry -> (child) BeliefNode

### Value Iteration

Implemented with [Lark's pruning algorithm.](https://arxiv.org/ftp/arxiv/papers/1302/1302.1525.pdf)

## Running an example ##
You can run tests with POMCP on RockSample, and use Value Iteration to solve the Tiger example.

**Note: The RockSample env needs some work to fully match up with the implementation described in Silver et al.**

You can optionally edit the RockSample configuration file `rock_sample_config.json` to change the map size or environment parameters.
This file is located in `pompdy/config`.
The following maps are available:
* RockSample(7, 8), a 7 x 7 grid with 8 rocks.
* RockSample(11, 11), an 11 x 11 grid with 11 rocks
* RockSample(15, 15), a 15 x 15 grid with 15 rocks
* As well as a few others, such as (7, 2), (7, 3), (12, 12), and more. It is fairly easy to make new maps.

To run RockSample with POMCP:

     python pomcp.py --env RockSample --solver POMCP --max_steps 200 --epsilon_start 1.0 --epsilon_decay 0.99 --n_epochs 10 --n_sims 500  --preferred_actions --seed 123
        
To run the Tiger example with the full-width planning value iteration algorithm: 

     python vi.py --env Tiger --solver ValueIteration --planning_horizon 8 --n_epochs 10 --max_steps 10 --seed 123

## Experimental Results for Approximate Value Iteration

