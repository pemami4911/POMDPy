__author__ = 'patrickemami'

import time
import numpy as np
from pomdpy.util import console
from pomdpy.action_selection import action_selectors
from solver import Solver

module = "monte_carlo_tree_search"


class MCTS(Solver):
    """
    Monte-Carlo Tree Search implementation, from POMCP
    """

    # Dimensions for the fast-UCB table
    UCB_N = 10000
    UCB_n = 100

    def __init__(self, agent, model):
        """
        Initialize an instance of the MCTS solver
        :param agent:
        :param model:
        :return:
        """
        super(MCTS, self).__init__(agent)

        # Pre-calculate UCB values for a speed-up
        self.fast_UCB = [[None for _ in range(MCTS.UCB_n)] for _ in range(MCTS.UCB_N)]

        for N in range(MCTS.UCB_N):
            for n in range(MCTS.UCB_n):
                if n is 0:
                    self.fast_UCB[N][n] = np.inf
                else:
                    self.fast_UCB[N][n] = model.sys_cfg["ucb_coefficient"] * np.sqrt(np.log(N + 1) / n)

    @staticmethod
    def reset(agent, model):
        """
        Generate a new MCTS solver

        Implementation of abstract method
        """
        return MCTS(agent, model)

    def prune(self, belief_node):
        """
        Prune the siblings of the chosen belief node and
        set that node as the new "root"
        :return:
        """
        start_time = time.time()
        self.belief_tree.prune_siblings(belief_node)
        elapsed = time.time() - start_time
        console(2, module, "Time spent pruning = " + str(elapsed) + " seconds")

    def find_fast_ucb(self, total_visit_count, action_map_entry_visit_count, log_n):
        """
        Look up and return the value in the UCB table corresponding to the params
        :param total_visit_count:
        :param action_map_entry_visit_count:
        :param log_n:
        :return:
        """
        assert self.fast_UCB is not None
        if total_visit_count < MCTS.UCB_N and action_map_entry_visit_count < MCTS.UCB_n:
            return self.fast_UCB[int(total_visit_count)][int(action_map_entry_visit_count)]

        if action_map_entry_visit_count == 0:
            return np.inf
        else:
            return self.model.sys_cfg["ucb_coefficient"] * np.sqrt(log_n / action_map_entry_visit_count)

    def select_action(self):
        """
        Starts off the Monte-Carlo Tree Search and returns the selected action. If the belief tree
                data structure is disabled, random rollout is used.

        Implementation of abstract method
        """
        if self.disable_tree:
            self.rollout_search(self.policy_iterator)
        else:
            self.policy_iteration()
        return action_selectors.ucb_action(self, self.policy_iterator, True)

    def simulate(self, state, start_time, sim_num):
        """
        :param state:
        :param start_time:
        :return:
        """
        return self.traverse(state, self.policy_iterator, 0, start_time)

    def traverse(self, state, belief_node, tree_depth, start_time):
        delayed_reward = 0

        # Time expired
        if time.time() - start_time > self.model.sys_cfg["action_selection_time_out"]:
            return 0

        action = action_selectors.ucb_action(self, belief_node, False)

        # Search horizon reached
        if tree_depth >= self.model.sys_cfg["maximum_depth"]:
            console(4, module, "Search horizon reached")
            return 0

        step_result, is_legal = self.model.generate_step(state, action)

        child_belief_node = belief_node.child(action, step_result.observation)
        if child_belief_node is None and not step_result.is_terminal and belief_node.action_map.total_visit_count > 0:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)

        if not step_result.is_terminal:
            tree_depth += 1
            if child_belief_node is not None:
                # Add S' to the new belief node
                # Add a state particle with the new state
                if child_belief_node.state_particles.__len__() < self.model.sys_cfg["max_particle_count"]:
                    child_belief_node.state_particles.append(step_result.next_state)
                delayed_reward = self.traverse(step_result.next_state, child_belief_node, tree_depth, start_time)
            else:
                delayed_reward = self.rollout(state, belief_node.data.generate_legal_actions())
            tree_depth -= 1
        else:
            console(3, module, "Reached terminal state.")

        # delayed_reward is "Q maximal"
        # current_q_value is the Q value of the current belief-action pair
        action_mapping_entry = belief_node.action_map.get_entry(action.bin_number)

        current_q_value = action_mapping_entry.mean_q_value
        visit_count = action_mapping_entry.visit_count

        # off-policy Q learning update rule
        q_value = current_q_value + (step_result.reward + (self.model.sys_cfg["discount"] * delayed_reward)
                                     - current_q_value) * (self.step_size / (1 + visit_count))

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        # Add RAVE ?
        return q_value