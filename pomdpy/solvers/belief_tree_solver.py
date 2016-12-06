from builtins import str
from builtins import range
import time
import random
import abc
from pomdpy.util import console
from pomdpy.pomdp.belief_tree import BeliefTree
from pomdpy.solvers import Solver

module = "BeliefTreeSolver"


class BeliefTreeSolver(Solver):
    """
    All POMDP solvers must implement the abstract methods specified below
    Ex. See POMCP (Monte-Carlo Tree Search)

    Provides a belief search tree and supports on- and off-policy methods
    """
    def __init__(self, agent):
        super(BeliefTreeSolver, self).__init__(agent)
        # The agent owns Histories, the collection of History Sequences.
        # There is one sequence per run of the algorithm
        self.history = agent.histories.create_sequence()
        # flag for determining whether the solver is an on/off-policy learning algorithm
        self.disable_tree = False

        self.belief_tree = BeliefTree(agent)

        # Initialize the Belief Tree
        self.belief_tree.reset()
        self.belief_tree.initialize()

        # generate state particles for root node belief state estimation
        # This is for simulation
        for i in range(self.model.n_start_states):
            particle = self.model.sample_an_init_state()
            self.belief_tree.root.state_particles.append(particle)

        self.belief_tree_index = self.belief_tree.root.copy()

    def monte_carlo_approx(self, eps, start_time):
        """
        Approximate Q(b, pi(b)) via monte carlo simulations, where b is the belief node pointed to by
        the belief tree index, and pi(b) is the action selected by the current behavior policy. For SARSA, this is
        simply pi(b) where pi is induced by current Q values. For Q-Learning, this is max_a Q(b',a')

        Does not advance the policy index
        :param eps
        :param start_time
        :return:
        """
        for i in range(self.model.n_sims):
            # Reset the Simulator
            self.model.reset_for_simulation()
            self.simulate(self.belief_tree_index, eps, start_time)

    @abc.abstractmethod
    def simulate(self, belief, eps, start_time):
        """
        Does a monte-carlo simulation from "belief" to approximate Q(b, pi(b))
        :param belief
        :param eps
        :param start_time
        :return:
        """

    @abc.abstractmethod
    def select_eps_greedy_action(self, eps, start_time):
        """
        Call methods specific to the implementation of the solver
        to select an action
        :param eps
        :param start_time
        :return:
        """

    def prune(self, belief_node):
        """
        Prune the siblings of the chosen belief node and
        set that node as the new "root"
        :param belief_node: node whose siblings will be removed
        :return:
        """
        start_time = time.time()
        self.belief_tree.prune_siblings(belief_node)
        elapsed = time.time() - start_time
        console(3, module, "Time spent pruning = " + str(elapsed) + " seconds")

    def rollout_search(self, belief_node):
        """
        At each node, examine all legal actions and choose the actions with
        the highest evaluation
        :return:
        """
        legal_actions = belief_node.data.generate_legal_actions()

        # rollout each action once
        for i in range(legal_actions.__len__()):
            state = belief_node.sample_particle()
            action = legal_actions[i % legal_actions.__len__()]

            # model.generate_step casts the variable action from an int to the proper DiscreteAction subclass type
            step_result, is_legal = self.model.generate_step(state, action)

            if not step_result.is_terminal:
                child_node, added = belief_node.create_or_get_child(step_result.action, step_result.observation)
                child_node.state_particles.append(step_result.next_state)
                delayed_reward = self.rollout(child_node)
            else:
                delayed_reward = 0

            action_mapping_entry = belief_node.action_map.get_entry(step_result.action.bin_number)

            q_value = action_mapping_entry.mean_q_value

            # Random policy
            q_value += (step_result.reward + self.model.discount * delayed_reward - q_value)

            action_mapping_entry.update_visit_count(1)
            action_mapping_entry.update_q_value(q_value)

    def rollout(self, belief_node):
        """
        Iterative random rollout search to finish expanding the episode starting at belief_node
        :param belief_node:
        :return:
        """
        legal_actions = belief_node.data.generate_legal_actions()

        if not isinstance(legal_actions, list):
            legal_actions = list(legal_actions)

        state = belief_node.sample_particle()
        is_terminal = False
        discounted_reward_sum = 0.0
        discount = 1.0
        num_steps = 0

        while num_steps < self.model.max_depth and not is_terminal:
            legal_action = random.choice(legal_actions)
            step_result, is_legal = self.model.generate_step(state, legal_action)
            is_terminal = step_result.is_terminal
            discounted_reward_sum += step_result.reward * discount
            discount *= self.model.discount
            # advance to next state
            state = step_result.next_state
            # generate new set of legal actions from the new state
            legal_actions = self.model.get_legal_actions(state)
            num_steps += 1

        return discounted_reward_sum

    def update(self, step_result, prune=True):
        """
        Feed back the step result, updating the belief_tree,
        extending the history, updating particle sets, etc

        Advance the policy index to point to the next belief node in the episode

        :return:
        """
        # Update the Simulator with the Step Result
        # This is important in case there are certain actions that change the state of the simulator
        self.model.update(step_result)

        child_belief_node = self.belief_tree_index.get_child(step_result.action, step_result.observation)

        # If the child_belief_node is None because the step result randomly produced a different observation,
        # grab any of the beliefs extending from the belief node's action node
        if child_belief_node is None:
            action_node = self.belief_tree_index.action_map.get_action_node(step_result.action)
            if action_node is None:
                # I grabbed a child belief node that doesn't have an action node. Use rollout from here on out.
                console(2, module, "Reached branch with no leaf nodes, using random rollout to finish the episode")
                self.disable_tree = True
                return

            obs_mapping_entries = list(action_node.observation_map.child_map.values())

            for entry in obs_mapping_entries:
                if entry.child_node is not None:
                    child_belief_node = entry.child_node
                    console(2, module, "Had to grab nearest belief node...variance added")
                    break

        # If the new root does not yet have the max possible number of particles add some more
        if child_belief_node.state_particles.__len__() < self.model.max_particle_count:

            num_to_add = self.model.max_particle_count - child_belief_node.state_particles.__len__()

            # Generate particles for the new root node
            child_belief_node.state_particles += self.model.generate_particles(self.belief_tree_index, step_result.action,
                                                                               step_result.observation, num_to_add,
                                                                               self.belief_tree_index.state_particles)

            # If that failed, attempt to create a new state particle set
            if child_belief_node.state_particles.__len__() == 0:
                child_belief_node.state_particles += self.model.generate_particles_uninformed(self.belief_tree_index,
                                                                                              step_result.action,
                                                                                              step_result.observation,
                                                                                        self.model.min_particle_count)

        # Failed to continue search- ran out of particles
        if child_belief_node is None or child_belief_node.state_particles.__len__() == 0:
            console(1, module, "Couldn't refill particles, must use random rollout to finish episode")
            self.disable_tree = True
            return

        self.belief_tree_index = child_belief_node
        if prune:
            self.prune(self.belief_tree_index)

