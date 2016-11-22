from belief_tree_solver import BeliefTreeSolver
from pomdpy.action_selection import e_greedy
from pomdpy.util import console
import time

module = "SARSA"


class SARSA(BeliefTreeSolver):
    """
    Implementation of the on-policy SARSA learning algorithm
    """

    def __init__(self, agent):
        super(SARSA, self).__init__(agent)

    @staticmethod
    def reset(agent):
        return SARSA(agent)

    def select_eps_greedy_action(self, eps, start_time):
        """
        Return an action given the current belief, as marked by the belief tree iterator, using an epsilon-greedy policy.

        If necessary, first carry out a rollout_search to expand the episode
        :param eps:
        :param start_time:
        :return:
        """
        if self.disable_tree:
            self.rollout_search(self.belief_tree_index)

        return e_greedy(self.belief_tree_index, eps)

    def simulate(self, belief_node, eps, start_time):
        """
        Implementation of the SARSA algorithm.

        Major differences between MCTS and SARSA:
            * This carries out on-policy search versus MCTS's off-policy search (MCTS uses Q-Learning)
            * The next action to be taken at a step within an episode is selected based on the current policy
                in SARSA (hence the second A in SARSA)

        Does not advance or modify the belief tree index
        
        :param belief_node:
        :param eps:
        :param start_time
        """

        # save the state of the current belief
        # only passing a reference to the action map
        current_belief = belief_node.copy()

        # epsilon-greedy action selection of initial action
        action = e_greedy(current_belief, eps)

        self.traverse(current_belief, action, eps, 0, start_time)

    def traverse(self, belief_node, action, eps, depth, start_time):
        depth += 1
        # generate S' and R
        state = belief_node.sample_particle()
        step_result, is_legal = self.model.generate_step(state, action)

        action_mapping_entry = self.belief_tree_index.action_map.get_entry(action.bin_number)

        if step_result.is_terminal or depth >= self.model.max_depth or not is_legal:
            action_mapping_entry.update_visit_count(1)
            action_mapping_entry.update_q_value(step_result.reward)
            return step_result.reward

        # Find the child belief node for the step result
        child_belief_node = self.belief_tree_index.child(action, step_result.observation)

        # Generate the child belief node if it didn't already exist, regardless of whether any of the actions
        # have been tried yet
        if child_belief_node is None and not step_result.is_terminal:
            child_belief_node, added = self.belief_tree_index.create_or_get_child(action, step_result.observation)

        # Add S' to the new belief node
        # Add a state particle with the new state
        if child_belief_node.state_particles.__len__() < self.model.max_particle_count:
            child_belief_node.state_particles.append(step_result.next_state)

        q_value = action_mapping_entry.mean_q_value

        # epsilon-greedy action selection of A' given S'
        next_action = e_greedy(child_belief_node, eps)

        next_q_value = self.traverse(child_belief_node, next_action, eps, depth, start_time)

        # on-policy SARSA update rule
        q_value += (step_result.reward + (self.model.discount * next_q_value) - q_value)

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        return q_value

