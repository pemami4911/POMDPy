# Brief Implementation Rundown #

To implement a POMDP with discrete states, actions, observations, and rewards, the following files should be extended: 

* Discrete Action
* Discrete State
* Discrete Observation
* Historical Data
* Discrete/Enumerated ActionPool
* Model - this module is the most important, since it acts as the black-box generator 
    of (S', A, O, R) steps. 

    You may want to to provide a .txt of .cfg containing a map or other data that encapsulate
    the environment and hence the transition probabilities for the world which the POMDP lives in.
   
Continuous POMDPs are not yet supported.
 
Point.py, ActionMapping.py, ActionPool.py, ObservationPool.py,
ObservationMapping.py, etc. in the POMDP package can all be extended to support a continuous POMDP.

## Belief Tree structure ##

Parent BeliefNode -> ActionMapping -> ActionMappingEntry -> ActionNode -> ObservationMap -> ObservationMappingEntry -> Child BeliefNode
