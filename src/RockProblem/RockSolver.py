__author__ = 'patrickemami'

import logging

import numpy as np

from Solvers import Solver
import TDStepper
import ActionSelectors


class RockSolver(Solver.Solver):
        
    def __init__(self, model):
        super(RockSolver, self).__init__(model)
        self.model = model
        self.logger = logging.getLogger('Model.Solver.RockSolver')
        self.step_generator = TDStepper.TDStepper(self.model, self)
        self.policy_step_count = 0
        self.sigmoid_steepness = 0
        self.n_episodes = self.model.config["num_episodes"]
        self.current_episode = 0

        # ------------ data collection ------------- #
        self.current_episode = 0

    ''' ------- Policy generation -------'''
    def generate_policy(self):

        # Initialize data structures
        self.initialize_empty()
        ActionSelectors.reset()

        # populate the root node with a set of history entries and then extend and backup each new sequence
        for policy, total_reward in self.generate_episodes(self.n_episodes, self.policy.root):
            yield policy, total_reward, self.model.num_reused_nodes

    ''' -------- Method for carrying out each episode during a simulation ------------'''
    def generate_episodes(self, n_episodes, root_node):

        # Calculate the steepness of the sigmoid curve used to decide the probability of using the NN heuristic
        # Least squares best fit curve for {(10, 1), (100, 0.1), (1000, 0.01)}
        # self.sigmoid_steepness = -1/(self.n_episodes**2)*self.current_episode**2 + 1
        # self.sigmoid_steepness = 1.3 * np.exp(-0.026 * self.current_episode)

        # The agent always starts at the same position - however, there will be
        # different initial rock configurations. The initial belief is that each rock
        # has equal probability of being Good or Bad
        state = self.model.sample_an_init_state()

        # number of times to extend out from the belief node
        for self.current_episode in range(0, n_episodes):
            print "Episode ", (self.current_episode + 1)
            #self.logger.info("Episode %s", str(self.current_episode + 1))

            # create a new sequence
            seq = self.histories.create_sequence()

            # create the first entry
            first_entry = seq.add_entry()
            first_entry.register_state(state)
            first_entry.register_node(root_node)

            # limit to episodes of length n for testing
            status = self.step_generator.extend_and_backup(seq, self.model.config["maximum_depth"])

            total_reward, policy = self.execute_most_recent_policy(seq)
            #self.logger.info("Total accumulated reward %s", str(total_reward))

            yield policy, total_reward

            # reset the root historical data for the next episode
            self.policy.reset_root_data()
            self.model.sampled_rock_yet = False
            self.model.num_times_sampled = 0.0
            self.model.good_samples = 0.0
            self.model.num_reused_nodes = 0
            self.model.num_bad_rocks_sampled = 0
            self.model.num_bad_checks = 0
            self.model.num_good_checks = 0
            self.model.unique_rocks_sampled = {}

    def execute_most_recent_policy(self, seq):
        policy = []
        total_reward = 0
        for entry in seq.entry_sequence:
            policy.append((entry.state, entry.action, entry.reward))
            total_reward += entry.reward
        return total_reward, policy

    # Traverse the belief tree and extract the embedded policy
    def execute(self):

        self.policy_step_count = 0
        total_discounted_reward = 0
        current_node = self.policy.root
        policy = []

        while True:
            q_value = -np.inf
            immediate_reward = None
            best_action = None
            observation = None
            belief = None

            history_entries_list = current_node.particles

            # starting at the root belief node, return the action with the highest Q
            # Look at every history entry associated with the current belief node
            for entry in history_entries_list:
                assert entry.associated_belief_node is current_node

                if entry.action is not None:
                    # grab the action mapping entry associated with the history entry
                    action_mapping_entry = entry.associated_belief_node.action_map.get_entry(entry.action)

                    if q_value < action_mapping_entry.mean_q_value:
                        q_value = action_mapping_entry.mean_q_value
                        best_action = entry.action
                        observation = entry.observation
                        immediate_reward = entry.reward
                        belief = entry.state

            # We're stuck!?
            if best_action is None:
                self.logger.info("Couldn't find an action to take from this belief node!")
                return total_discounted_reward, policy

            # add up the total discounted accumulated so far
            total_discounted_reward += immediate_reward
            self.policy_step_count += 1

            policy.append((belief, best_action, immediate_reward))
            #yield best_action, total_discounted_reward

            # Advance to the next belief node, corresponding to the chosen action
            current_node = current_node.get_child(best_action, observation)

            # Once there are no more belief nodes
            if current_node is None:
                return total_discounted_reward, policy
