## POMDPy
![Build](https://travis-ci.org/pemami4911/POMDPy.svg?branch=master)

This open-source project contains a framework for implementing discrete action/state POMDPs in Python. This work was inspired by [TAPIR](http://robotics.itee.uq.edu.au/~hannakur/dokuwiki/doku.php?id=wiki:tapir) and the [POMCP](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications.html) algorithm.

[What the heck is a POMDP?](http://www.pomdp.org/tutorial/index.shtml)

Here's David Silver and Joel Veness's paper on POMCP, a ground-breaking POMDP solver. [Monte-Carlo Planning in Large POMDPs](http://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf)

This project has been conducted strictly for research purposes. If you would like to contribute to POMDPy or if you have any comments or suggestions, feel free to send me a pull request or send me an email at pemami@ufl.edu.  

## Installation ##
Download the files as a zip or clone into the repository.
From the root directory of the project, run
 
    python setup.py install

## POMCP and SARSA ##

So far, there are implementations of POMCP (Q-learning and PO-UCT with MCTS) and SARSA. The main difference between the two algorithms is that POMCP uses the off-policy Q-Learning
algorithm and the UCT action-selection strategy. SARSA uses an on-policy variant of TD-Learning. 

Both algorithms encode the action-value function as a belief search tree. POMCP is an anytime planner that approximates the action-value
estimates of the current belief via Monte-Carlo before taking a step. This is known as Monte-Carlo Tree Search (MCTS).
SARSA is episodic, in that the agent repeatedly carries out full episodes 
and uses the generated history of experiences to back-up the action-value estimates up the taken path to the root of the belief tree. 

## Running an example ##
Currently, you can test POMCP and SARSA on the classic Tiger and RockSample POMDPs. 

You can optionally edit the RockSample configuration file `rock_problem_config.json` to change the map size or environment parameters.
This file is located in `pompdy/config`.

Edit the run parameters in `config.json`.
The following maps are available:
* RockSample(7, 8), a 7 x 7 grid with 8 rocks.
* RockSample(11, 11), an 11 x 11 grid with 11 rocks
* RockSample(15, 15), a 15 x 15 grid with 15 rocks
* As well as a few others, such as (7, 2), (7, 3), (12, 12), and more. It is fairly easy to make new maps.

To run the RockSample problem with POMCP:

    python run_pomdp.py --env RockProblem --solver POMCP --use_sims
        
To run the episodic Tiger problem with SARSA: 

    python run_pomdp.py --env TigerProblem --solver SARSA
       
See `pompdy/README.md` for more implementation details.

## Running tests ##
Unit tests can be ran with 
    
    py.test test/unit_tests.py
    
## Dependencies ##

This project uses:

* Python 2.7.9
* numpy 1.9.2
* pytest 2.7.0

Used for plotting results [NOT CURRENTLY FUNCTIONAL]:

* matplotlib 1.4.3

## TODO ##
* [ ] Add more unit tests
* [ ] Add additional benchmark problems 
* [ ] Supply an easy-to-use configuration schema to specify a POMDP and auto-generate the classes
* [ ] Add ways of creating/learning generative models
* [x] Clean up output displayed to the user
* [ ] Continuous-time POMDPs? DNN-POMDPs? AIXI?
