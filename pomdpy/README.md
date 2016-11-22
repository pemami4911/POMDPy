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

## Belief Tree structure ##

Parent BeliefNode -> ActionMapping -> ActionMappingEntry -> ActionNode -> ObservationMap -> ObservationMappingEntry -> Child BeliefNode
