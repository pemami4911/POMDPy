__author__ = 'patrickemami'

import Solver
import logging
import TDStepper
import numpy as np

class RockSolver(Solver.Solver):

    def __init__(self, model):
        super(RockSolver, self).__init__(model)
        self.model = model
        self.logger = logging.getLogger('Model.Solver.RockSolver')
        self.step_generator = TDStepper.TDStepper(self.model)
        self.policy_step_count = 0
        self.sigmoid_steepness = 0
        self.n_episodes = self.model.config["num_episodes"]

        # ------------ data collection ------------- #
        self.total_accumulated_rewards = []


    ''' ------- Policy generation -------'''
    def generate_policy(self):

        # populate the root node with a set of history entries and then extend and backup each new sequence
        #reward, policy = self.generate_episodes(self.model.config["num_episodes"], self.policy.root)
        #return reward, policy

        for policy, total_reward in self.generate_episodes(self.n_episodes, self.policy.root):
            yield policy, total_reward, self.model.num_reused_nodes

        #return total_reward, self.policy_step_count, self.successful_samples
        #return policy, self.policy_step_count

    ''' -------- Method for carrying out each episode during a simulation ------------'''
    def generate_episodes(self, n_episodes, root_node):

        # Calculate the steepness of the sigmoid curve used to decide the probability of using the NN heuristic
        # Least squares best fit curve for {(10, 1), (100, 0.1), (1000, 0.01)}
        self.sigmoid_steepness = 1.29155 * np.exp(-0.0255843 * n_episodes)

        # The agent always starts at the same position - however, there will be
        # different initial rock configurations. The initial belief is that each rock
        # has equal probability of being Good or Bad
        state = self.model.sample_an_init_state()

        #print "Initial Belief: ",
        #state.print_state()

        # number of times to extend out from the belief node
        for self.current_episode in range(0, n_episodes):
            # create a new sequence
            seq = self.histories.create_sequence()

            # create the first entry
            first_entry = seq.add_entry()
            first_entry.register_state(state)
            first_entry.register_node(root_node)

            # limit to episodes of length n for testing
            status = self.step_generator.extend_and_backup(seq, self.model.config["maximum_depth"])

            total_reward, policy = self.execute_most_recent_policy(seq)
            self.total_accumulated_rewards.append(total_reward)
            yield policy, total_reward

            # reset the root historical data for the next episode
            self.policy.reset_root_data()
            self.model.sampled_rock_yet = False
            self.model.num_reused_nodes = 0
            #average_reward = (average_reward + reward)/(idx + 1)

            # Display the final status after each episode is generated
            #self.logger.info("Extend and backup status: %s", status)

        #return reward, policy

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
