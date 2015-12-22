__author__ = 'patrickemami'

from solver import Solver
from pomdpy.action_selection import ucb_action
from numpy import random as rand


class SARSA(Solver):
    """
    Implementation of the on-policy SARSA learning algorithm
    """

    def __init__(self, agent, on_policy=True):
        super(SARSA, self).__init__(agent, on_policy)

    def simulate(self, state, start_time, sim_num):
        """
        Implementation of the SARSA algorithm.

        Major differences between MCTS and SARSA:
            * This carries out on-policy search versus MCTS's off-policy search
            * The next action to be taken at a step within an episode is selected based on the current policy
                in SARSA (hence the second A in SARSA)

        Does not advance or modify the policy iterator
        """

        # save the state of the policy iterator
        #current_belief = self.policy_iterator.copy()

        # epsilon-greedy action selection of initial action
        if rand.random < (1 / (1 + sim_num)):
            action = rand.random.choice(self.model.get_all_actions()[0])
        else:
            action = self.select_action()

        approximate_q_value = self.traverse(state, action, 0)

        # reset the state
        #self.policy_iterator = current_belief

        return approximate_q_value

    def traverse(self, state, action, depth):
        depth += 1
        # generate S' and R
        step_result, is_legal = self.model.generate_step(state, action)

        if step_result.is_terminal or depth >= self.model.sys_cfg["maximum_depth"]:
            return 0

        # Find the child belief node for the step result
        child_belief_node = self.policy_iterator.child(action, step_result.observation)

        # Generate the child belief node if it didn't already exist, regardless of whether any of the actions
        # have been tried yet
        if child_belief_node is None and not step_result.is_terminal:
            child_belief_node, added = self.policy_iterator.create_or_get_child(action,
                                                                                step_result.observation)
        # Add S' to the new belief node
        # Add a state particle with the new state
        if child_belief_node.state_particles.__len__() < self.model.sys_cfg["max_particle_count"]:
            child_belief_node.state_particles.append(step_result.next_state)

        action_mapping_entry = self.policy_iterator.action_map.get_entry(action.bin_number)
        q_value = action_mapping_entry.mean_q_value
        # Bn = Bn'
        self.policy_iterator = child_belief_node

        # epsilon-greedy action selection of A' given S'
        if rand.random < (1 / (1 + depth)):
            next_action = rand.random.choice(self.model.get_all_actions()[0])
        else:
            next_action = self.select_action()

        next_q_value = self.traverse(step_result.next_state, next_action, depth)

        # on-policy SARSA update rule
        q_value = q_value + (step_result.reward + (self.model.sys_cfg["discount"] * next_q_value)
                             - q_value) * self.step_size

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        return q_value

    @staticmethod
    def reset(agent, model):
        return SARSA(agent, True)

    def select_action(self):
        """
        Return an action given the current belief, as marked by the belief tree iterator, using an epsilon-greedy policy.

        If necessary, first carry out a rollout_search to expand the episode
        :return:
        """
        if self.disable_tree:
            self.rollout_search(self.policy_iterator)
        return ucb_action(None, self.policy_iterator, greedy=True)

    def prune(self, belief_node):
        pass


