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

Step size is killing the rewards gained by the sampling action. a static step size is an unbiased judge,
whose hammer flattens rewards and penalties alike. 

-> to solve this, force the agent to explore until the average total reward for the policy is the reward for reaching
the goal, and then allow the agent to search for rocks --> DIDN'T WORK. An exploring agent needs landmarks to localize 
itself, otherwise it becomes too challenging for it to obtain a good representation of the environment just by
using random action sampling to find its way around

-> TD Q Learning acts randomly with a decreasing rate of 1/n, which screws up shit
-> Otherwise, so long as the sensor's settings is set to "pretty shitty", it's working pretty well

