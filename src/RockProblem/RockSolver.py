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
        #self.done_exploring = False

    ''' ------- Policy generation -------'''
    def generate_policy(self):

        # populate the root node with a set of history entries and then extend and backup each new sequence
        policy = self.generate_episodes(self.model.config["num_episodes"], self.policy.root)

        #return total_reward, self.policy_step_count, self.successful_samples

        return policy, self.policy_step_count

    ''' -------- Method for carrying out each episode during a simulation ------------'''
    def generate_episodes(self, n_episodes, root_node):

        average_reward = 0

        # The agent always starts at the same position - however, there will be
        # different initial rock configurations. The initial belief is that each rock
        # has equal probability of being Good or Bad
        state = self.model.sample_an_init_state()

        print "Initial Belief: ",
        state.print_state()

        # Store the current policy in here
        policy = []

        # number of times to extend out from the belief node
        for idx in range(0, n_episodes):

            # create a new sequence
            seq = self.histories.create_sequence()

            # create the first entry
            first_entry = seq.add_entry()
            first_entry.register_state(state)
            first_entry.register_node(root_node)

            # limit to episodes of length n for testing
            status = self.step_generator.extend_and_backup(seq, self.model.config["maximum_depth"])

            #print "-------------Current best policy -----------"
            reward, policy = self.execute()
            print "Total reward: ", reward
            print "Total step count: ", self.policy_step_count

            # reset the root historical data for the next episode
            self.policy.reset_root_data()
            self.model.sampled_rock_yet = False
            #average_reward = (average_reward + reward)/(idx + 1)

            #if average_reward > 0.9 * self.model.exit_reward:
            #    self.done_exploring = True
            #else:
            #    self.done_exploring = False
            #if reward > 0.5 * self.model.max_val:
            #   print "Scored better than 50% of max value!!!"
            #    self.successful_samples = self.model.number_of_successful_samples
            #    return policy

            # Display the final status after each episode is generated
            #self.logger.info("Extend and backup status: %s", status)

        return policy

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

            #current_node.data.grid_position.print_position()
            #print "Best Q Value: ", q_value
            #print "Belief: ",
            #belief.print_state()

            policy.append((best_action, belief))
            #yield best_action, total_discounted_reward

            # Advance to the next belief node, corresponding to the chosen action
            current_node = current_node.get_child(best_action, observation)

            # Once there are no more belief nodes
            if current_node is None:
                return total_discounted_reward, policy
