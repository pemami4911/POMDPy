## Brief Implementation Rundown ##

To implement a POMDP with discrete states, actions, observations, and rewards, the following files should be extended: 

* Discrete Action
* Discrete State
* Discrete Observation
* Historical Data ( optional, used to designate preferred actions )
* Discrete/Enumerated ActionPool
* Model - this module is the most important, since it acts as the black-box generator 
    of (S', A, O, R) steps. 

    You may want to to provide a .txt of .cfg containing a map or other data that encapsulate
    the environment and hence the transition probabilities for the world which the POMDP lives in.
   
Support for POMDPs with continuous state/action/observation spaces is on the way.  

## Solvers ##

* [POMCP](http://papers.nips.cc/paper/4031-monte-carlo-planning-in-large-pomdps.pdf): Monte-Carlo Tree Search with the UCB1 algorithm for efficient exploration of the belief space 
* [SARSA](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node64.html)

## Belief Tree structure ##

Parent BeliefNode -> ActionMapping -> ActionMappingEntry -> ActionNode -> ObservationMap -> ObservationMappingEntry -> Child BeliefNode
