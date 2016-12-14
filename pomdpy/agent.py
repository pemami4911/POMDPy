from __future__ import print_function
import time
import logging
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider, select_action
from pomdpy.util.plot_alpha_vectors import plot_gamma
from pomdpy.solvers import POMCP, SARSA, ValueIteration

module = "agent"


class Agent:
    """
    Training loops for learning, and for
    storing statistics on performance
    """

    def __init__(self, model, solver):
        """
        Initialize the POMDPY agent
        :param model:
        :param solver:
        :return:
        """
        self.logger = logging.getLogger('POMDPy.Solver')
        self.model = model
        self.results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset  # Factory method for generating instances of the solver

    def discounted_return(self):

        if not self.model.use_tf:
            self.multi_epoch()
        else:
            self.multi_epoch_tf()

        console(2, module, 'epochs: ' + str(self.model.n_epochs))
        console(2, module, 'ave undiscounted return: ' + str(self.experiment_results.undiscounted_return.mean) +
                ' +- ' + str(self.experiment_results.undiscounted_return.std_err()))
        console(2, module, 'ave discounted return: ' + str(self.experiment_results.discounted_return.mean) +
                ' +- ' + str(self.experiment_results.discounted_return.std_err()))
        console(2, module, 'ave time/epoch: ' + str(self.experiment_results.time.mean))

        self.logger.info('env: ' + self.model.env + '\t' +
                         'epochs: ' + str(self.model.n_epochs) + '\t' +
                         'ave undiscounted return: ' + str(self.experiment_results.undiscounted_return.mean) + ' +- ' +
                         str(self.experiment_results.undiscounted_return.std_err()) + '\t' +
                         'ave discounted return: ' + str(self.experiment_results.discounted_return.mean) +
                         ' +- ' + str(self.experiment_results.discounted_return.std_err()) +
                         '\t' + 'ave time/epoch: ' + str(self.experiment_results.time.mean))

    def multi_epoch_tf(self):
        import tensorflow as tf

        with tf.Session() as sess:
            solver = self.solver_factory(self, sess)

            for epoch in range(self.model.n_epochs):
                epoch_start = time.time()

                self.model.reset_for_epoch()

                # train for 1 epoch
                solver.train()

                if epoch % self.model.test == 0:
                    console(2, module, 'Evaluating agent at epoch {}'.format(epoch))
                    self.results = Results()

                    # evaluate agent
                    reward = 0
                    discounted_reward = 0
                    discount = 1.0
                    belief = self.model.get_initial_belief_state()

                    while True:
                        action, v_b = solver.predict(belief)
                        step_result = self.model.generate_step(action)

                        if not step_result.is_terminal:
                            belief = self.model.belief_update(belief, action, step_result.observation)

                        reward += step_result.reward
                        discounted_reward += discount * step_result.reward
                        discount *= self.model.discount

                        # show the step result
                        self.display_step_result(epoch, step_result)

                        if step_result.is_terminal:
                            break

                    print('Total reward: {} discounted reward: {}'.format(reward, discounted_reward))

                    self.results.update_reward_results(reward, discounted_reward)
                    # self.results.show(epoch)

                    # save model
                    # solver.save_model(step=epoch)
                    plot_gamma('V for epoch {}'.format(epoch), solver.alpha_vectors())

                self.results.time.add(time.time() - epoch_start)

                if self.experiment_results.time.running_total > self.model.timeout:
                    console(2, module, 'Timed out after ' + str(epoch) + ' epochs in ' +
                            self.experiment_results.time.running_total + ' seconds')
                    break

        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

    def multi_epoch(self):
        eps = self.model.epsilon_start
        solver = self.solver_factory(self)

        self.model.reset_for_epoch()

        for i in range(self.model.n_epochs):
            # Reset the epoch stats
            self.results = Results()

            if isinstance(solver, POMCP):
                eps = self.run_pomcp(i + 1, eps)
                self.model.reset_for_epoch()
            elif isinstance(solver, SARSA):
                eps = self.run_episodic(solver, i + 1, eps)
            elif isinstance(solver, ValueIteration):
                self.run_value_iteration(solver, i + 1)
                self.model.reset_for_epoch()

            if self.experiment_results.time.running_total > self.model.timeout:
                console(2, module, 'Timed out after ' + str(i) + ' epochs in ' +
                        self.experiment_results.time.running_total + ' seconds')
                break

    def run_pomcp(self, epoch, eps):
        epoch_start = time.time()

        # Create a new solver
        solver = self.solver_factory(self)

        # Monte-Carlo start state
        state = solver.belief_tree_index.sample_particle()

        reward = 0
        discounted_reward = 0
        discount = 1.0

        for i in range(self.model.max_steps):

            start_time = time.time()

            # action will be of type Discrete Action
            action = solver.select_eps_greedy_action(eps, start_time)

            # update epsilon
            if eps > self.model.epsilon_end:
                eps -= self.model.epsilon_decay

            step_result, is_legal = self.model.generate_step(state, action)

            reward += step_result.reward
            discounted_reward += discount * step_result.reward

            discount *= self.model.discount
            state = step_result.next_state

            # show the step result
            self.display_step_result(i, step_result)

            if not step_result.is_terminal or not is_legal:
                solver.update(step_result)

            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                              step_result.action, step_result.observation, step_result.next_state)

            if step_result.is_terminal or not is_legal:
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break

        self.results.time.add(time.time() - epoch_start)
        self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        # print_divider('large')
        solver.history.show()
        self.results.show(epoch)
        console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        print_divider('medium')

        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

        return eps

    def run_episodic(self, solver, epoch, eps):
        """
        Used for episodic belief tree solvers that update the action-values along the tree after doing each rollout.

        Each
        :param solver:
        :param epoch:
        :param eps:
        :return:
        """
        epoch_start = time.time()

        for _ in range(self.model.n_sims):
            solver.simulate(solver.belief_tree_index, eps, epoch_start)

            # update epsilon
            if eps > self.model.epsilon_end:
                eps -= self.model.epsilon_decay

        if epoch % self.model.test == 0:
            state = solver.belief_tree_index.sample_particle()
            # console(2, module, 'Initial belief state: ' + state.to_string())
            discount = 1.0
            # save the pointer to the root to reset
            root = solver.belief_tree_index.copy()
            # Reset the history
            solver.history = self.histories.create_sequence()

            reward = 0
            discounted_reward = 0

            for i in range(self.model.max_steps):

                start_time = time.time()

                # action will be of type Discrete Action
                action = solver.select_eps_greedy_action(eps, start_time)

                step_result, is_legal = self.model.generate_step(state, action)

                reward += step_result.reward
                discounted_reward += discount * step_result.reward

                discount *= self.model.discount
                state = solver.belief_tree_index.sample_particle()

                # show the step result
                self.display_step_result(i, step_result)

                if not step_result.is_terminal or not is_legal:
                    solver.update(step_result, prune=False)

                # Extend the history sequence
                new_hist_entry = solver.history.add_entry()
                HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                                  step_result.action, step_result.observation, step_result.next_state)

                if step_result.is_terminal or not is_legal:
                    console(3, module, 'Terminated after episode step ' + str(i + 1))
                    break

            solver.belief_tree_index = root

            self.results.time.add(time.time() - epoch_start)
            self.results.update_reward_results(reward, discounted_reward)

            # Pretty Print results
            solver.history.show()
            self.results.show(epoch)
            console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
            print_divider('medium')

            self.experiment_results.time.add(self.results.time.running_total)
            self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
            self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
            self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
            self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

        return eps

    def run_value_iteration(self, solver, epoch):
        run_start_time = time.time()

        reward = 0
        discounted_reward = 0
        discount = 1.0

        solver.value_iteration(self.model.get_transition_matrix(),
                               self.model.get_observation_matrix(),
                               self.model.get_reward_matrix(),
                               self.model.planning_horizon)

        b = self.model.get_initial_belief_state()

        for i in range(self.model.max_steps):

            # TODO: record average V(b) per epoch
            action, v_b = select_action(b, solver.gamma)

            step_result = self.model.generate_step(action)

            b = self.model.belief_update(b, action, step_result.observation)

            reward += step_result.reward
            discounted_reward += discount * step_result.reward
            discount *= self.model.discount

            # show the step result
            self.display_step_result(i, step_result)

            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                              step_result.action, step_result.observation, None)

            if step_result.is_terminal:
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break

        self.results.time.add(time.time() - run_start_time)
        self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        solver.history.show()
        self.results.show(epoch)
        console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        print_divider('medium')

        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

    @staticmethod
    def display_step_result(step_num, step_result):
        """
        Pretty prints step result information
        :param step_num:
        :param step_result:
        :return:
        """
        console(3, module, 'Step Number = ' + str(step_num))
        console(3, module, 'Step Result.Action = ' + step_result.action.to_string())
        console(3, module, 'Step Result.Observation = ' + step_result.observation.to_string())
        # console(3, module, 'Step Result.Next_State = ' + step_result.next_state.to_string())
        console(3, module, 'Step Result.Reward = ' + str(step_result.reward))


class Results(object):
    """
    Maintain the statistics for each run
    """
    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.add(r)
        self.discounted_return.add(dr)

    def reset_running_totals(self):
        self.time.running_total = 0.0
        self.discounted_return.running_total = 0.0
        self.undiscounted_return.running_total = 0.0

    def show(self, epoch):
        print_divider('large')
        print('\tEpoch #' + str(epoch) + ' RESULTS')
        print_divider('large')
        console(2, module, 'discounted return statistics')
        print_divider('medium')
        self.discounted_return.show()
        print_divider('medium')
        console(2, module, 'undiscounted return statistics')
        print_divider('medium')
        self.undiscounted_return.show()
        print_divider('medium')
        console(2, module, 'Time')
        print_divider('medium')
        self.time.show()
        print_divider('medium')
