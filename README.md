Q-Learning and Multi-Armed Bandit Approach to Solving POMDPs
=======================================================

This project will contain the Python code that I am writing to implement my altered version of the Q-Learning 
Stochastic Approximation algorithm and the benchmark problem that I am going to be testing it on. 

Features to implement
=====================

- The agent should establish that a rock is bad if it sees that the chance that it is good is very low
- Kill the agent & end the sequence if it does a bad action
- forget legal actions.. let it learn by doing!
- Stop the simulation if the agent reaches a percentage within the max total reward ?


Notes
=====

Belief Tree structure
---------------------
BeliefNode -> ActionMapping -> ActionMappingEntry -> ActionNode -> ObservationMap -> ObservationMappingEntry -> BeliefNode

If the Sampling action doesn't do the agent good early on, it will be mostly ignored because its Q is getting flattened

Step size is killing the rewards gained by the sampling action. a static step size is an unbiased judge,
whose hammer flattens rewards and penalties alike
