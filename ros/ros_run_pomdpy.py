#!/usr/bin/env python

__author__ = 'patrick emami'

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import uint32

from sampleproblems.tigerproblem import *
from POMDP.solvers import *

"""
Simple Publisher Node for POMDPy 
"""

if __name__ == "__main__": 
	rospy.init_node('pomdpy_publisher')

	action_pub = rospy.Publisher('/robot_action', uint32, queue_size=1)
	state_pub = rospy.Publisher('/robot_state', PoseWithCovarianceStamped, queue_size=1)
	observation_pub = rospy.Publisher('/robot_observation', uint32, queue_size=1)

        simulator = TigerModel.TigerModel("Tiger Problem")
        my_solver = Solver.Solver(simulator)

        # ros_run() is a generator that sends over the step result after each MCTS update
        for action, observation, next_state in my_solver.ros_run(): 

    	# convert state to PoseWithCovarianceStamped

    	action_pub.publish(action)
    	observation_pub.publish(observation)
    	#state_pub.publish(next_state)
