Q-Learning and Multi-Armed Bandit Approach to Solving POMDPs
=======================================================

This project will contain the Python code that I am writing to implement my altered version of the Q-Learning 
Stochastic Approximation algorithm and the benchmark problem that I am going to be testing it on. 

Features to implement
=====================


Why does the agent not establish that a rock is bad if it sees that the chance that it is good is very low
- Kill the agent (end the sequence) if it does a bad action
- forget legal actions, let it learn by doing
- stop the simulation if the agent reaches a percentage within the max total reward
- Optimistic initial values --> Maybe initialize the Q value of the preferred action to + 20 or something like that?



Notes
=====

Belief Tree structure
---------------------
BeliefNode -> ActionMapping -> ActionMappingEntry -> ActionNode -> ObservationMap -> ObservationMappingEntry -> BeliefNode

If the Sampling action doesn't do the agent good early on, it will be mostly ignored because its Q is getting flattened

Step size is killing the rewards gained by the sampling action. a static step size is an unbiased judge,
whose hammer flattens rewards and penalties alike. 

-> to solve this, force the agent to explore until the average total reward for the policy is the reward for reaching
the goal, and then allow the agent to search for rocks --> DIDN'T WORK. An exploring agent needs landmark to localize 
itself, otherwise it becomes to challenging to obtain using random action sampling to find its way around

--> TD Q Learning acts randomly with a decreasing rate of 1/n, which screws up shit
--> Otherwise, so long as the sensor's settings is set to "pretty shitty", it's working pretty well

