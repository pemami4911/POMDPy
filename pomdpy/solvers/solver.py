__author__ = 'patrickemami'

import time
import random
import abc
from pomdpy.util import console
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.belief_tree import BeliefTree

module = "Solver"


class Solver(object):
    """
    All POMDP solvers must implement the interface specified below
    Ex. See MCTS (Monte-Carlo Tree Search)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, agent, on_policy=False):
        self.agent = agent
        self.model = self.agent.model
        # runner owns Histories, the collection of History Sequences.
        # There is one sequence per run of the MCTS algorithm
        self.history = self.agent.histories.create_sequence()
        # flag for determining whether the solver is an on/off-policy learning algorithm
        self.on_policy = on_policy
        self.disable_tree = False
        self.step_size = self.model.sys_cfg["step_size"]
        self.total_reward_stats = Statistic("Total Reward")
        self.belief_tree = BeliefTree(agent)

        # Initialize the Belief Tree
        self.belief_tree.reset()
        self.belief_tree.initialize()

        # generate state particles for root node belief state estimation
        # This is for simulation
        for i in range(self.model.sys_cfg["num_start_states"]):
            particle = self.model.sample_an_init_state()
            self.belief_tree.root.state_particles.append(particle)

        self.policy_iterator = self.belief_tree.root.copy()

    def policy_iteration(self):
        """
        Template-method pattern

        For on-policy learning algorithms such as SARSA, this method will carry out the
        policy iteration. Afterwards, the learned policy can be evaluated by consecutive calls to
        select_action(), which specifies the action selection rule

        For off-policy learning algorithms such as Q-learning, this method will repeatedly be called
        at each step of the policy traversal

        The policy iterator does not advance

        :return:
        """
        start_time = time.time()

        self.total_reward_stats.clear()

        # save the state of the current belief
        # only passing a reference to the action map
        current_belief = self.policy_iterator.copy()

        for i in range(self.model.sys_cfg["num_sims"]):
            # Reset the Simulator
            self.model.reset_for_simulation()

            state = self.policy_iterator.sample_particle()

            console(3, module, "Starting simulation at random state = " + state.to_string())

            approx_value = self.simulate(state, start_time, i)

            self.total_reward_stats.add(approx_value)

            console(3, module, "Approximation of the value function = " + str(approx_value))

            # reset the policy iterator
            self.policy_iterator = current_belief

    @abc.abstractmethod
    def simulate(self, state, start_time, sim_num):
        """
        Initialize the policy iteration algorithm with implementation specific details
        :return:
        """

    @staticmethod
    @abc.abstractmethod
    def reset(agent, model):
        """
        Should return a new instance of a concrete solver class
        :return:
        """

    @abc.abstractmethod
    def select_action(self):
        """
        Call methods specific to the implementation of the solver
        to select an action
        :return:
        """

    @abc.abstractmethod
    def prune(self, belief_node):
        """
        Override to add implementation-specific prune operations, if desired
        :return:
        """

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
                delayed_reward = self.rollout(step_result.next_state, child_node.data.generate_legal_actions())
            else:
                delayed_reward = 0

            action_mapping_entry = belief_node.action_map.get_entry(step_result.action.bin_number)
            q_value = action_mapping_entry.mean_q_value

            q_value += (step_result.reward + self.model.sys_cfg["discount"] * delayed_reward - q_value) * self.step_size

            action_mapping_entry.update_visit_count(1)
            action_mapping_entry.update_q_value(q_value)

    def rollout(self, start_state, legal_actions):
        """
        Iterative random rollout search to finish expanding the episode starting at "start_state"
        """
        if not isinstance(legal_actions, list):
            legal_actions = list(legal_actions)

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

        return total_reward

    def update(self, step_result):
        """
        Feed back the step result, updating the belief_tree,
        extending the history, updating particle sets, etc

        Advance the policy iterator to point to the next belief node in the episode

        :return:
        """
        # Update the Simulator with the Step Result
        # This is important in case there are certain actions that change the state of the simulator
        self.model.update(step_result)

        child_belief_node = self.policy_iterator.get_child(step_result.action, step_result.observation)

        # If the child_belief_node is None because the step result randomly produced a different observation,
        # grab any of the beliefs extending from the belief node's action node
        if child_belief_node is None:
            action_node = self.policy_iterator.action_map.get_action_node(step_result.action)
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
            child_belief_node.state_particles += self.model.generate_particles(self.policy_iterator, step_result.action,
                                                                               step_result.observation, num_to_add,
                                                                               self.policy_iterator.state_particles)

            # If that failed, attempt to create a new state particle set
            if child_belief_node.state_particles.__len__() == 0:
                child_belief_node.state_particles += self.model.generate_particles_uninformed(self.policy_iterator,
                                                                                              step_result.action,
                                                                                              step_result.observation,
                                                                                              self.model.sys_cfg[
                                                                                                  "min_particle_count"])

        # Failed to continue search- ran out of particles
        if child_belief_node is None or child_belief_node.state_particles.__len__() == 0:
            console(1, module, "Couldn't refill particles, must use random rollout to finish episode")
            self.disable_tree = True
            return

        self.policy_iterator = child_belief_node
        self.prune(self.policy_iterator)