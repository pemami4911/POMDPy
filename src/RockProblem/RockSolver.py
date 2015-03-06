__author__ = 'patrickemami'

import Solver
import logging
import SmartStepper
import numpy as np

class RockSolver(Solver.Solver):

    def __init__(self, model):
        super(RockSolver, self).__init__(model)
        self.model = model
        self.logger = logging.getLogger('Model.Solver.RockSolver')
        self.generator = SmartStepper.SmartStepper(self.model, self)
        self.policy_step_count = 0
        self.successful_samples = 0

    '''----- Utility function[s] -------'''
    def traverse_history_sequence(self, seq):
        """
        # Generator for accessing each History entry in a history sequence
        :param seq:
        :return: History Entry
        """
        for entry in seq:
            yield entry

    ''' ------- Policy generation -------'''
    def generate_policy(self):

        # populate the root node with a set of history entries and then extend and backup each new sequence
        policy = self.generate_episodes(self.model.config["num_episodes"], self.policy.root)

        #return total_reward, self.policy_step_count, self.successful_samples

        return policy, self.policy_step_count, self.successful_samples

    def generate_episodes(self, n_episodes, root_node):

         # The agent always starts at the same position - however, there will be
        # different initial rock configurations. The initial belief is that each rock
        # has equal probability of being Good or Bad
        state = self.model.sample_an_init_state()

        print "Initial Belief: ",
        state.print_state()
        policy = []

        # number of times to extend out from the belief node
        for idx in range(0, n_episodes):

            # create a new sequence
            seq = self.histories.create_sequence()

            # create the first entry
            first_entry = seq.add_entry()

            # change to toString
            #state.position.print_position()

            first_entry.register_state(state)

            # adds this History Entry to the root belief node
            first_entry.register_node(root_node)

            # limit to episodes of length n for testing
            status = self.generator.extend_and_backup(seq, self.model.config["maximum_depth"])

            #print "-------------Current best policy -----------"
            reward, policy = self.execute()
            print "Total reward: ", reward
            print "Total step count: ", self.policy_step_count

            #if reward > 0.5 * self.model.max_val:
            #   print "Scored better than 50% of max value!!!"
            #    self.successful_samples = self.model.number_of_successful_samples
            #    return policy

            # Display the final status after each episode is generated
            #self.logger.info("Extend and backup status: %s", status)

        self.successful_samples = self.model.number_of_successful_samples
        return policy

    # Back-propagation of the new Q values
    def update_sequence(self, seq):
        if seq.__len__() < 1:
            self.logger.warning("Cannot update sequence of length < 1")
            return

        discount_factor = self.model.config["discount_factor"]
        step_size = self.model.config["step_size"]

        # reverse the sequence, since we must back up starting from the end
        r_seq = list(seq)
        r_seq.reverse()

        # Since the last history entry has no more actions to take, initialize maximal q to 0
        maximal_q = 0

        # handle the first entry in the reversed sequence, which won't have a corresponding observation mapping entry
        first_entry = r_seq[0]
        action_mapping_entry = first_entry.associated_belief_node.action_map.get_entry(first_entry.action)
        current_q = action_mapping_entry.mean_q_value

        # update the action mapping entry with the newly calculated maximal q value
        maximal_q = (discount_factor * (maximal_q - current_q) + first_entry.reward) * step_size
        action_mapping_entry.update_q_value(maximal_q)

        # traverse the rest of the list, starting with the second history entry
        for entry in self.traverse_history_sequence(r_seq[1:]):

            action_mapping_entry = entry.associated_belief_node.action_map.get_entry(entry.action)

            current_q = action_mapping_entry.mean_q_value

            maximal_q = (discount_factor * (maximal_q - current_q) + entry.reward) * step_size

            # update the action's Q value and visit count
            action_mapping_entry.update_q_value(maximal_q)

            # update the observation visit count
            obs_entry = action_mapping_entry.child_node.observation_map.get_entry(entry.observation)
            obs_entry.update_visit_count(1)

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

            policy.append(best_action)
            #yield best_action, total_discounted_reward

            # Advance to the next belief node, corresponding to the chosen action
            current_node = current_node.get_child(best_action, observation)

            # Once there are no more belief nodes
            if current_node is None:
                return total_discounted_reward, policy
