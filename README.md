## POMDPy
![Build](https://travis-ci.org/pemami4911/POMDPy.svg?branch=master)  ![Python27](https://img.shields.io/badge/python-2.7-blue.svg)  ![Python35](https://img.shields.io/badge/python-3.5-blue.svg)

This open-source project contains a framework for implementing discrete action/state POMDPs in Python. This work was inspired by [TAPIR](http://robotics.itee.uq.edu.au/~hannakur/dokuwiki/doku.php?id=wiki:tapir) and the [POMCP](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Applications.html) algorithm.

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
* [SARSA](https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/sarsa.py)
* [Value Iteration](https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/value_iteration.py)

The main difference between POMCP and SARSA is that POMCP uses the off-policy Q-Learning
algorithm and the UCT action-selection strategy. SARSA uses an on-policy variant of TD-Learning. **Both algorithms 
encode the action-value function as a belief search tree.** POMCP is an anytime planner that approximates the action-value
estimates of the current belief via Monte-Carlo simulations before taking a step. This is known as Monte-Carlo Tree Search (MCTS).
SARSA is episodic, in that the agent repeatedly carries out full episodes 
and uses the generated history of experiences to back-up the action-value estimates up the taken path to the root of the belief tree. 

I have also implemented exact Value Iteration with Lark's pruning algorithm. This can only be used on the Tiger Problem. 

## Running an example ##
Currently, you can test POMCP and SARSA on the classic Tiger and RockSample POMDPs. 

You can optionally edit the RockSample configuration file `rock_problem_config.json` to change the map size or environment parameters.
This file is located in `pompdy/config`.
The following maps are available:
* RockSample(7, 8), a 7 x 7 grid with 8 rocks.
* RockSample(11, 11), an 11 x 11 grid with 11 rocks
* RockSample(15, 15), a 15 x 15 grid with 15 rocks
* As well as a few others, such as (7, 2), (7, 3), (12, 12), and more. It is fairly easy to make new maps.

To run the RockSample problem with POMCP:

    ./main.py --env RockProblem --solver POMCP --max_steps 200 --epsilon_start 1.0 --epsilon_decay 0.01 --n_runs 10 --n_sims 500  --preferred_actions --seed 123
        
To run the Tiger problem with SARSA: 

    ./main.py --env TigerProblem --solver SARSA --max_steps 5 --epsilon_start 0.5 --n_runs 100 --seed 123
       
See `pompdy/README.md` for details about implementing new POMDP benchmark problems.
    
## TODO ##
* [ ] Random baseline solver
* [ ] Add more unit tests
* [ ] Add additional benchmark problems 
* [ ] Continuous-action/state space POMDPs
