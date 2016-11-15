from solver import Solver
from pomdpy.action_selection import e_greedy
from pomdpy.util import console
import time

module = "SARSA"


class SARSA(Solver):
    """
    Implementation of the on-policy SARSA learning algorithm
    """

    def __init__(self, agent):
        super(SARSA, self).__init__(agent)

    @staticmethod
    def reset(agent, model):
        return SARSA(agent)

    def select_action(self, eps, start_time):
        """
        Return an action given the current belief, as marked by the belief tree iterator, using an epsilon-greedy policy.

        If necessary, first carry out a rollout_search to expand the episode
        :return:
        """
        if self.disable_tree:
            self.rollout_search(self.belief_tree_index)
        elif self.agent.use_sims:
            # Get a monte carlo approximation of Q(b,a) for all a
            self.monte_carlo_approx(eps, start_time)
        return e_greedy(self.belief_tree_index, eps)

    def simulate(self, belief_state, eps, start_time):
        """
        Implementation of the SARSA algorithm.

        Major differences between MCTS and SARSA:
            * This carries out on-policy search versus MCTS's off-policy search (MCTS uses Q-Learning)
            * The next action to be taken at a step within an episode is selected based on the current policy
                in SARSA (hence the second A in SARSA)

        Does not advance or modify the policy iterator
        
        :param belief_state:
        :param eps:
        """

        # save the state of the current belief
        # only passing a reference to the action map
        current_belief = self.belief_tree_index.copy()

        # epsilon-greedy action selection of initial action
        action = e_greedy(self.belief_tree_index, eps)

        self.traverse(belief_state, action, eps, 0, start_time)

        # reset the index
        self.belief_tree_index = current_belief

    def traverse(self, state, action, eps, depth, start_time):
        # Time expired
        if time.time() - start_time > self.model.sys_cfg["action_selection_time_out"]:
            console(4, module, "action selection timeout")
            return 0

        depth += 1
        # generate S' and R
        step_result, is_legal = self.model.generate_step(state, action)

        action_mapping_entry = self.belief_tree_index.action_map.get_entry(action.bin_number)

        if step_result.is_terminal or depth >= self.model.sys_cfg["maximum_depth"]:
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
        if child_belief_node.state_particles.__len__() < self.model.sys_cfg["max_particle_count"]:
            child_belief_node.state_particles.append(step_result.next_state)

        q_value = action_mapping_entry.mean_q_value
        # Bn = Bn'
        self.belief_tree_index = child_belief_node

        # epsilon-greedy action selection of A' given S'
        next_action = e_greedy(self.belief_tree_index, eps)

        next_q_value = self.traverse(step_result.next_state, next_action, eps, depth, start_time)

        # on-policy SARSA update rule
        # + (1/(1 + action_mapping_entry.visit_count))
        q_value += (step_result.reward + (self.model.sys_cfg["discount"] * next_q_value) - q_value)

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        return q_value

