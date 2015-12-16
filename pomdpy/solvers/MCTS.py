__author__ = 'patrickemami'

import time
import random
import numpy as np
from pomdpy.util import console
from pomdpy.pomdp import Statistic
from pomdpy.action_selection import action_selectors
from pomdpy.pomdp.belief_tree import BeliefTree
from solver import Solver

module = "MCTS"


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
        super(MCTS, self).__init__(agent, model)
        self.policy = BeliefTree(agent)
        self.peak_tree_depth = 0
        self.step_size = self.model.sys_cfg["step_size"]
        self.tree_depth_stats = Statistic("Tree Depth")
        self.rollout_depth_stats = Statistic("Rollout Depth")
        self.total_reward_stats = Statistic("Total Reward")

        if self.model.sys_cfg["policy_representation"] is "Tree" or "tree":
            self.disable_tree = False
        else:
            self.disable_tree = True

        # Pre-calculate UCB values for a speed-up
        self.fast_UCB = [[None for _ in range(MCTS.UCB_n)] for _ in range(MCTS.UCB_N)]

        for N in range(MCTS.UCB_N):
            for n in range(MCTS.UCB_n):
                if n is 0:
                    self.fast_UCB[N][n] = np.inf
                else:
                    self.fast_UCB[N][n] = model.sys_cfg["ucb_coefficient"] * np.sqrt(np.log(N + 1) / n)

        # Initialize the Belief Tree
        self.policy.reset()
        self.policy.initialize()

        # generate state particles for root node belief state estimation
        # This is for simulation
        for i in range(self.model.sys_cfg["num_start_states"]):
            particle = self.model.sample_an_init_state()
            self.policy.root.state_particles.append(particle)

    @staticmethod
    def reset(agent, model):
        """
        Generate a new MCTS solver

        Implementation of abstract method
        """
        return MCTS(agent, model)

    def clear_stats(self):
        """
        Reset statistics
        """
        self.total_reward_stats.clear()
        self.tree_depth_stats.clear()
        self.rollout_depth_stats.clear()

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
            self.rollout_search()
        else:
            self.uct_search()
        return action_selectors.ucb_action(self, self.policy.root, True)

    def update(self, step_result):
        """
        Given the result of applying the action selected by MCTS, update the belief tree, particle set, etc

        Implementation of abstract method
        :param step_result:
        :return:
        """
        # Update the Simulator with the Step Result
        # This is important in case there are certain actions that change the state of the simulator
        self.model.update(step_result)

        child_belief_node = self.policy.root.get_child(step_result.action, step_result.observation)

        # If the child_belief_node is None because the step result randomly produced a different observation,
        # grab any of the beliefs extending from the belief node's action node
        if child_belief_node is None:
            action_node = self.policy.root.action_map.get_action_node(step_result.action)
            if action_node is None:
                # I grabbed a child belief node that doesn't have an action node. Use rollout from here on out.
                console(2, module, "Reached branch with no leaf nodes, using random rollout to finish the episode")
                self.disable_tree = True
                return

            obs_mapping_entries = action_node.observation_map.child_map.values()

            for entry in obs_mapping_entries:
                if entry.child_node is not None:
                    child_belief_node = entry.child_node
                    console(2, module, "Had to grab nearest belief node...variance added")
                    break

        # If the new root does not yet have the max possible number of particles add some more
        if child_belief_node.state_particles.__len__() < self.model.sys_cfg["max_particle_count"]:

            num_to_add = self.model.sys_cfg["max_particle_count"] - child_belief_node.state_particles.__len__()

            # Generate particles for the new root node
            child_belief_node.state_particles += self.model.generate_particles(self.policy.root, step_result.action,
                                                                               step_result.observation, num_to_add,
                                                                               self.policy.root.state_particles)

            # If that failed, attempt to create a new state particle set
            if child_belief_node.state_particles.__len__() == 0:
                child_belief_node.state_particles += self.model.generate_particles_uninformed(self.policy.root,
                                                                                              step_result.action,
                                                                                              step_result.observation,
                                                                                              self.model.sys_cfg[
                                                                                                  "min_particle_count"])

        # Failed to continue search- ran out of particles
        if child_belief_node is None or child_belief_node.state_particles.__len__() == 0:
            console(1, module, "Couldn't refill particles, must use random rollout to finish episode")
            self.disable_tree = True
            return

        # Prune the siblings of the chosen belief node and
        # set that node as the new "root"
        start_time = time.time()
        self.policy.prune_siblings(child_belief_node)
        elapsed = time.time() - start_time
        console(2, module, "Time spent pruning = " + str(elapsed) + " seconds")
        self.policy.root = child_belief_node

    def rollout_search(self):
        """
        At each node, examine all legal actions and choose the actions with
        the highest evaluation
        :return:
        """
        for i in range(self.model.sys_cfg["num_sims"]):
            state = self.policy.root.sample_particle()
            legal_actions = self.policy.root.data.generate_legal_actions()
            action = legal_actions[i % legal_actions.__len__()]

            # model.generate_step casts the variable action from an int to the proper DiscreteAction subclass type
            step_result, is_legal = self.model.generate_step(state, action)

            if not step_result.is_terminal:
                child_node, added = self.policy.root.create_or_get_child(step_result.action, step_result.observation)
                child_node.state_particles.append(step_result.next_state)
                delayed_reward = self.rollout(step_result.next_state, child_node.data.generate_legal_actions())
            else:
                delayed_reward = 0

            # TODO Might want to subtract out the current mean_q_value
            total_reward = (step_result.reward + self.model.sys_cfg["discount"] * delayed_reward) * self.step_size
            action_mapping_entry = self.policy.root.action_map.get_entry(step_result.action.bin_number)
            assert action_mapping_entry is not None

            action_mapping_entry.update_visit_count(1.0)
            action_mapping_entry.update_q_value(total_reward)

    def rollout(self, start_state, starting_legal_actions):
        """
        Iterative random rollout search
        """
        legal_actions = list(starting_legal_actions)
        state = start_state.copy()
        is_terminal = False
        total_reward = 0.0
        discount = 1.0
        num_steps = 0

        while num_steps < self.model.sys_cfg["maximum_depth"] and not is_terminal:
            legal_action = random.choice(legal_actions)
            step_result, is_legal = self.model.generate_step(state, legal_action)
            is_terminal = step_result.is_terminal
            total_reward += step_result.reward * discount
            discount *= self.model.sys_cfg["discount"]
            # advance to next state
            state = step_result.next_state
            # generate new set of legal actions from the new state
            legal_actions = self.model.get_legal_actions(state)
            num_steps += 1

        self.rollout_depth_stats.add(num_steps)
        return total_reward

    ''' --------------- Multi-Armed Bandit Search -------------- '''
    def uct_search(self):
        """
        Expand the root node via random simulations, using the UCT action selection strategy
        """
        start_time = time.time()

        self.clear_stats()
        # Create a snapshot of the current information state
        initial_root_data = self.policy.root.data.copy()

        for i in range(self.model.sys_cfg["num_sims"]):
            # Reset the Simulator
            self.model.reset_for_sim()

            # Reset the root node to the information state at the beginning of the UCT Search
            # After each simulation
            self.policy.root.data = initial_root_data.copy()

            state = self.policy.root.sample_particle()
            # Tree depth, which increases with each recursive step
            tree_depth = 0
            self.peak_tree_depth = 0

            console(3, module, "Starting simulation at random state = " + state.to_string())

            # initiate
            total_reward = self.simulate_node(state, self.policy.root, tree_depth, start_time)

            self.total_reward_stats.add(total_reward)
            self.tree_depth_stats.add(self.peak_tree_depth)

            console(3, module, "Total reward = " + str(total_reward))

        # Reset the information state back to the state it was in before the simulations occurred for consistency
        self.policy.root.data = initial_root_data

    def simulate_node(self, state, belief_node, tree_depth, start_time):
        """
        Begin the process of traversing down the Belief Tree by selecting an action according to the UCB1 algorithm
        :param state:
        :param belief_node:
        :param tree_depth:
        :param start_time:
        :return:
        """
        # Time expired
        if time.time() - start_time > self.model.sys_cfg["action_selection_time_out"]:
            return 0

        action = action_selectors.ucb_action(self, belief_node, False)

        self.peak_tree_depth = tree_depth

        # Search horizon reached
        if tree_depth >= self.model.sys_cfg["maximum_depth"]:
            console(4, module, "Search horizon reached")
            return 0

        if tree_depth == 1:
            # Add a state particle with the new state
            if belief_node.state_particles.__len__() < self.model.sys_cfg["max_particle_count"]:
                belief_node.state_particles.append(state)

        # Q value
        total_reward = self.step_node(belief_node, state, action, tree_depth, start_time)
        # Add RAVE ?
        return total_reward

    def step_node(self, belief_node, state, action, tree_depth, start_time):
        """
        Generate the next step in the current episode based on the selected action and update the corresponding Q value
        :param belief_node:
        :param state:
        :param action:
        :param tree_depth:
        :param start_time:
        :return:
        """

        # Time expired
        if time.time() - start_time > self.model.sys_cfg["action_selection_time_out"]:
            return 0

        delayed_reward = 0

        step_result, is_legal = self.model.generate_step(state, action)

        console(4, module, "Step Result.Action = " + step_result.action.to_string())
        console(4, module, "Step Result.Observation = " + step_result.observation.to_string())
        console(4, module, "Step Result.Next_State = " + step_result.next_state.to_string())
        console(4, module, "Step Result.Reward = " + str(step_result.reward))

        child_belief_node = belief_node.child(action, step_result.observation)
        if child_belief_node is None and not step_result.is_terminal and belief_node.action_map.total_visit_count > 0:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)

        if not step_result.is_terminal:
            tree_depth += 1
            if child_belief_node is not None:
                delayed_reward = self.simulate_node(step_result.next_state, child_belief_node, tree_depth, start_time)
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

        q_value = current_q_value + (step_result.reward + (self.model.sys_cfg["discount"] * delayed_reward)
                                     - current_q_value) * (self.step_size / (1 + visit_count))

        # q_value = step_result.reward + (self.model.sys_cfg["discount"] * delayed_reward)

        # q_value = (self.step_size / (1 + visit_count)) * (step_result.reward + self.model.sys_cfg["discount"]
        #  * delayed_reward)

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        return q_value
