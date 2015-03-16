Q-Learning and Multi-Armed Bandit Approach to Solving POMDPs
=======================================================

This project will contain the Python code that I am writing to implement my altered version of the Q-Learning 
Stochastic Approximation algorithm and the benchmark problem that I am going to be testing it on. 

Notes
=====

Belief Tree structure
---------------------
BeliefNode -> ActionMapping -> ActionMappingEntry -> ActionNode -> ObservationMap -> ObservationMappingEntry -> BeliefNode

If the Sampling action doesn't do the agent good early on, it will be mostly ignored because its Q is getting flattened

An exploring agent needs landmarks to localize --> SLAM
Otherwise it becomes too challenging for it to obtain a good representation of the environment just by
using random action sampling to find its way around

-> TD Q Learning acts randomly with a decreasing rate : epsilon-greedy. 

-> Need to set up a knowledge base module, where the programmer specifies the rules that govern the action available 
to the agent at each time step
