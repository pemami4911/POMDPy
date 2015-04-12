__author__ = 'patrickemami'

import time

import numpy as np

import BeliefTree as BT
import ActionSelectors
import Statistic
import BeliefNode
from console import *


module = "MCTS"
disable_tree = False

''' ------------ Global vars ----------------- '''


class MCTS(object):
    """
    Monte-Carlo Tree Search POMDP solver, from POMCP
    """
    UCB_N = 10000
    UCB_n = 100

    def __init__(self, solver, model):
        self.solver = solver
        self.model = model
        self.policy = BT.BeliefTree(solver) # Search Tree
        self.peak_tree_depth = 0
        self.tree_depth_stats = Statistic.Statistic("Tree Depth")
        self.rollout_depth_stats = Statistic.Statistic("Rollout Depth")
        self.total_reward_stats = Statistic.Statistic("Total Reward")
        self.step_size = self.model.sys_cfg["step_size"]

        # Solver owns Histories, the collection of History Sequences.
        # There is one sequence per run of the MCTS algorithm
        self.history = self.solver.histories.create_sequence()

        # Pre-calculate UCB values for a speed-up
        self.fast_UCB = [[None for _ in range(MCTS.UCB_n)] for _ in range(MCTS.UCB_N)]

        for N in range(MCTS.UCB_N):
            for n in range(MCTS.UCB_n):
                if n is 0:
                    self.fast_UCB[N][n] = np.inf
                else:
                    self.fast_UCB[N][n] = model.sys_cfg["ucb_coefficient"] * np.sqrt(np.log(N + 1)/n)

        # Initialize the Belief Tree
        self.reset()

    def reset(self):
        # Initialize policy root stuff
        self.policy.reset()
        self.policy.initialize()

        # generate state particles for root node belief state estimation
        # This is for simulation
        for i in range(self.model.sys_cfg["num_start_states"]):
            particle = self.model.sample_an_init_state()
            self.policy.root.state_particles.append(particle)

    def clear_stats(self):
        self.total_reward_stats.clear()
        self.tree_depth_stats.clear()
        self.rollout_depth_stats.clear()

    def find_fast_ucb(self, total_visit_count, action_map_entry_visit_count, log_n):
        assert self.fast_UCB is not None
        if total_visit_count < MCTS.UCB_N and action_map_entry_visit_count < MCTS.UCB_n:
            return self.fast_UCB[int(total_visit_count)][int(action_map_entry_visit_count)]

        if action_map_entry_visit_count == 0:
            return np.inf
        else:
            return self.model.sys_cfg["ucb_coefficient"] * np.sqrt(log_n/action_map_entry_visit_count)

    def select_action(self):
        if disable_tree:
            self.rollout_search()
        else:
            self.UCT_search()
        return ActionSelectors.ucb_action(self, self.policy.root, True)

    def update(self, step_result):

        # Update the Simulator with the Step Result
        # This is important in case there are certain actions that change the state of the simulator
        self.model.update(step_result)

        child_belief_node = self.policy.root.get_child(step_result.action, step_result.observation)

        # If the child_belief_node is None because the step result randomly produced a different observation,
        # grab any of the beliefs extending from the belief node's action node
        if child_belief_node is None:
            obs_mapping_entries = self.policy.root.action_map.get_action_node(step_result.action).\
                observation_map.child_map.values()
            for entry in obs_mapping_entries:
                if entry.child_node is not None:
                    child_belief_node = entry.child_node
                    print "Had to grab nearest belief node...uncertainty introduced"
                    break

        new_root = BeliefNode.BeliefNode(self.solver)
        new_root.data = child_belief_node.data.shallow_copy()
        new_root.action_map = self.solver.action_pool.create_action_mapping(new_root)

        # Extend the history sequence
        new_hist_entry = self.history.add_entry()
        new_hist_entry.reward = step_result.reward
        new_hist_entry.action = step_result.action
        new_hist_entry.observation = step_result.observation
        new_hist_entry.register_entry(new_hist_entry, None, step_result.next_state)

        if child_belief_node is not None:
            console(2, module + ".update", "Matched " + str(child_belief_node.state_particles.__len__()) + " states")
            # If a child belief node for the root already exists, just copy over its state particles into the new root's
            # particles
            for i in child_belief_node.state_particles:
                new_root.state_particles.append(i.copy())

        else:
            console(2, module + ".update", "No matching node found")

        # If the new root does not yet have the max possible number of particles add some more
        if new_root.state_particles.__len__() < self.model.sys_cfg["max_particle_count"]:

            num_to_add = self.model.sys_cfg["max_particle_count"] - new_root.state_particles.__len__()
            # Generate particles for the new root node
            new_root.state_particles += self.model.generate_particles(self.policy.root, step_result.action,
                                            step_result.observation, num_to_add,
                                            self.policy.root.state_particles)

            # If that failed, attempt to create a new state particle set
            if new_root.state_particles.__len__() == 0:
                new_root.state_particles += self.model.generate_particles_uninformed(self.policy.root, step_result.action,
                                                                                    step_result.observation,
                                                                                    self.model.sys_cfg["min_particle_count"])

        # Failed to continue search
        if new_root.state_particles.__len__() == 0 and (child_belief_node is None or
                                                                    child_belief_node.state_particles.__len__() == 0):
            return True

        # delete old tree and set the new root
        start_time = time.time()
        self.policy.prune_tree(self.policy)
        elapsed = time.time() - start_time
        print "Time spent pruning = ", str(elapsed)
        self.policy.root = new_root

        return False

    ''' --------------- Rollout Search --------------'''
    def rollout_search(self):
        legal_actions = self.model.get_legal_actions()
        history_length = self.history.entry_sequence.__len__()

        for i in range(self.model.sys_cfg["num_sims"]):
            action = legal_actions[i%legal_actions.size()]
            state = self.policy.root.sample_particle()
            step_result = self.model.generate_step(state, action)
            new_hist_entry = self.history.add_entry()
            new_hist_entry.reward = step_result.reward
            new_hist_entry.action = step_result.action
            new_hist_entry.observation = step_result.observation

            if not step_result.is_terminal:
                child_node = self.policy.root.create_or_get_child(action, step_result.observation)
                child_node.state_particles.append(step_result.next_state)

            # Create a new history entry and step the history forward
            new_hist_entry.register_entry(new_hist_entry, None, step_result.next_state)
            delayed_reward = self.rollout(step_result.next_state)

            # TODO Might want to subtract out the current mean_q_value
            total_reward = (step_result.reward + self.model.sys_cfg["discount"] * delayed_reward) * self.step_size

            action_mapping_entry = self.policy.root.action_map.get_entry(step_result.action)
            assert action_mapping_entry is not None

            action_mapping_entry.update_visit_count(1.0)
            action_mapping_entry.update_q_value(total_reward)

            # Truncate the history sequence
            self.history.entry_sequence = self.history.entry_sequence[:history_length]

    # Uniform Random Action Rollout
    def rollout(self, state):
        console(2, module + ".rollout", "Starting rollout")

        is_terminal = False
        total_reward = 0.0
        discount = 1.0
        num_steps = 0
        while num_steps < self.model.sys_cfg["maximum_depth"] and not is_terminal:

            rand_action = self.model.get_random_action()
            step_result, is_legal = self.model.generate_step(state, rand_action)
            is_terminal = step_result.is_terminal

            console(4, module + ".rollout", "Step Result.Action = " + step_result.action.to_string())
            console(4, module + ".rollout", "Step Result.Observation = " + step_result.observation.to_string())
            console(4, module + ".rollout", "Step Result.Next_State = " + step_result.next_state.to_string())
            console(4, module + ".rollout", "Step Result.Reward = " + str(step_result.reward))

            total_reward += step_result.reward * discount
            discount *= self.model.sys_cfg["discount"]

            num_steps += 1

        self.rollout_depth_stats.add(num_steps)
        console(2, module + ".rollout", "Ending rollout after " + str(num_steps) + " steps, with total reward "
                + str(total_reward))
        return total_reward

    ''' --------------- Multi-Armed Bandit Search -------------- '''
    def UCT_search(self):
        """
        Expands the root node via random simulations
        :return:
        """

        self.clear_stats()
        # Create a snapshot of the current information state
        initial_root_data = self.policy.root.data.shallow_copy()

        for i in range(self.model.sys_cfg["num_sims"]):
            # Reset the Simulator
            self.model.reset()

            # Reset the root node to the information state at the beginning of the UCT Search
            # After each simulation
            self.policy.root.data = initial_root_data.shallow_copy()

            state = self.policy.root.sample_particle()
            # Tree depth, which increases with each recursive step
            tree_depth = 0
            self.peak_tree_depth = 0

            console(3, module + ".UCT_search", "Starting simulation at random state = " + state.to_string())

            # initiate
            total_reward = self.simulate_node(state, self.policy.root, tree_depth)

            self.total_reward_stats.add(total_reward)
            self.tree_depth_stats.add(self.peak_tree_depth)

            console(3, module + ".UCT_search", "Total reward = " + str(total_reward))

        # Reset the information state back to the state it was in before the simulations occurred for consistency
        self.policy.root.data = initial_root_data

        console_no_print(3, self.tree_depth_stats.show)
        console_no_print(3, self.rollout_depth_stats.show)
        console_no_print(3, self.total_reward_stats.show)

    def simulate_node(self, state, belief_node, tree_depth):

        action = ActionSelectors.ucb_action(self, belief_node, False)

        self.peak_tree_depth = tree_depth

        # Search horizon reached
        if tree_depth >= self.model.sys_cfg["maximum_depth"]:
            console(4, module + ".simulate_node", "Search horizon reached, getting tf outta here")
            return 0

        if tree_depth == 1:
            # Add a state particle with the new state
            if belief_node.state_particles.__len__() < self.model.sys_cfg["max_particle_count"]:
                belief_node.state_particles += [state]

        # Q value
        total_reward = self.step_node(belief_node, state, action, tree_depth)
        # Add RAVE ?
        return total_reward

    def step_node(self, belief_node, state, action, tree_depth):

        delayed_reward = 0

        step_result, is_legal = self.model.generate_step(state, action)

        console(4, module + ".step_node", "Step Result.Action = " + step_result.action.to_string())
        console(4, module + ".step_node", "Step Result.Observation = " + step_result.observation.to_string())
        console(4, module + ".step_node", "Step Result.Next_State = " + step_result.next_state.to_string())
        console(4, module + ".step_node", "Step Result.Reward = " + str(step_result.reward))

        child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)

        if not step_result.is_terminal:
            if child_belief_node is not None:
                tree_depth += 1
                delayed_reward = self.simulate_node(step_result.next_state, child_belief_node, tree_depth)
            else:
                delayed_reward = self.rollout(state)
            tree_depth -= 1
        else:
            console(3, module + ".step_node", "Reached terminal state.")

        # TODO try subtracting out current Q value for variance-control purposes
        # delayed_reward is "Q maximal"
        Q_value = (step_result.reward + self.model.sys_cfg["discount"] * delayed_reward) * self.step_size

        belief_node.action_map.get_entry(action.bin_number).update_visit_count(1)
        belief_node.action_map.get_entry(action.bin_number).update_q_value(Q_value)

        return Q_value

