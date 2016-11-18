import time
import logging
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider
from pomdpy.solvers import POMCP, SARSA

module = "agent"


class Agent(object):
    """
    This class is responsible for initiating a run
    and storing statistics on that run
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
        self.run_results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset  # Factory method for generating instances of the solver

    def discounted_return(self):
        """
        logging and runs
        :return:
        """
        self.multi_run()

        console(2, module, 'runs: ' + str(self.model.sys_cfg['num_runs']))
        console(2, module, 'ave undiscounted return: ' + str(self.experiment_results.undiscounted_return.mean) +
                ' +- ' + str(self.experiment_results.undiscounted_return.std_err()))
        console(2, module, 'ave discounted return: ' + str(self.experiment_results.discounted_return.mean) +
                ' +- ' + str(self.experiment_results.discounted_return.std_err()))
        console(2, module, 'ave time/run: ' + str(self.experiment_results.time.mean))

        self.logger.info('env: ' + self.model.problem_name + '\t' +
                         'runs: ' + str(self.model.sys_cfg['num_runs']) + '\t' +
                         'ave undiscounted return: ' + str(self.experiment_results.undiscounted_return.mean) + ' +- ' +
                         str(self.experiment_results.undiscounted_return.std_err()) + '\t' +
                         'ave discounted return: ' + str(self.experiment_results.discounted_return.mean) +
                         ' +- ' + str(self.experiment_results.discounted_return.std_err()) +
                         '\t' + 'ave time/run: ' + str(self.experiment_results.time.mean))

    def multi_run(self):
        num_runs = self.model.sys_cfg['num_runs']

        eps = self.model.sys_cfg['epsilon_start']
        solver = self.solver_factory(self, self.model)

        for i in range(num_runs):
            # Reset the run stats
            self.run_results = Results()

            # Perform behaviors that must done for each run
            self.model.reset_for_run()

            if isinstance(self.solver_factory, POMCP):
                eps = self.run_mcts(i + 1, eps)
            elif isinstance(self.solver_factory, SARSA):
                eps = self.run_episodic(solver, i + 1, eps)

            if self.experiment_results.time.running_total > self.model.sys_cfg['max_time_out']:
                console(2, module, 'Timed out after ' + str(i) + ' runs in ' +
                        self.experiment_results.time.running_total + ' seconds')
                break

    def run_mcts(self, run_num, eps):
        run_start_time = time.time()
        max_steps = self.model.sys_cfg['max_steps']

        # Create a new solver
        solver = self.solver_factory(self, self.model)

        # Monte-Carlo start state
        state = solver.belief_tree_index.sample_particle()
        console(2, module, 'Initial belief state: ' + state.to_string())

        reward = 0
        discounted_reward = 0
        discount = 1.0

        for i in xrange(max_steps):

            # update epsilon
            if (run_num % self.model.sys_cfg['epsilon_update_frequency']) == 0:
                if eps > self.model.sys_cfg['epsilon_end']:
                    eps -= self.model.sys_cfg['epsilon_delta']

            start_time = time.time()

            # action will be of type Discrete Action
            action = solver.select_action(eps, start_time)

            step_result, is_legal = self.model.generate_step(state, action)

            reward += step_result.reward
            discounted_reward += discount * step_result.reward

            discount *= self.model.sys_cfg['discount']
            state = step_result.next_state

            # show the step result
            self.display_step_result(i, step_result)

            if not step_result.is_terminal:
                solver.update(step_result)

            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                              step_result.action, step_result.observation, step_result.next_state)

            if step_result.is_terminal:
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break

        self.run_results.time.add(time.time() - run_start_time)
        self.run_results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        # print_divider('large')
        solver.history.show()
        self.run_results.show(run_num)
        console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        print_divider('medium')

        self.experiment_results.time.add(self.run_results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.run_results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.run_results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.run_results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.run_results.discounted_return.running_total)

        return eps

    def run_episodic(self, solver, run_num, eps):
        """
        Play 1 episode
        :param solver:
        :param run_num:
        :param eps:
        :return:
        """
        run_start_time = time.time()

        max_steps = self.model.sys_cfg['max_steps']

        # simulate one episode
        solver.simulate(solver.belief_tree_index.sample_particle(), eps, run_start_time)

        # update epsilon
        if (run_num % self.model.sys_cfg['epsilon_update_frequency']) == 0:
            if eps > self.model.sys_cfg['epsilon_end']:
                eps -= self.model.sys_cfg['epsilon_delta']

        if (run_num % self.model.sys_cfg['test_frequency']) == 0:
            state = solver.belief_tree_index.sample_particle()
            # console(2, module, 'Initial belief state: ' + state.to_string())
            discount = 1.0
            # save the pointer to the root to reset
            root = solver.belief_tree_index.copy()
            # Reset the history
            solver.history = solver.agent.histories.create_sequence()

            reward = 0
            discounted_reward = 0

            for i in xrange(max_steps):

                start_time = time.time()

                # action will be of type Discrete Action
                action = solver.select_action(eps, start_time)

                step_result, is_legal = self.model.generate_step(state, action)

                reward += step_result.reward
                discounted_reward += discount * step_result.reward

                discount *= self.model.sys_cfg['discount']
                state = step_result.next_state

                # show the step result
                self.display_step_result(i, step_result)

                if not step_result.is_terminal:
                    solver.update(step_result, prune=False)

                # Extend the history sequence
                new_hist_entry = solver.history.add_entry()
                HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                                  step_result.action, step_result.observation, step_result.next_state)

                if step_result.is_terminal:
                    console(3, module, 'Terminated after episode step ' + str(i + 1))
                    break

            solver.belief_tree_index = root

            self.run_results.time.add(time.time() - run_start_time)
            self.run_results.update_reward_results(reward, discounted_reward)

            # Pretty Print results
            solver.history.show()
            self.run_results.show(run_num)
            console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
            print_divider('medium')

            self.experiment_results.time.add(self.run_results.time.running_total)
            self.experiment_results.undiscounted_return.count += (self.run_results.undiscounted_return.count - 1)
            self.experiment_results.undiscounted_return.add(self.run_results.undiscounted_return.running_total)
            self.experiment_results.discounted_return.count += (self.run_results.discounted_return.count - 1)
            self.experiment_results.discounted_return.add(self.run_results.discounted_return.running_total)

        return eps

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
        console(3, module, 'Step Result.Next_State = ' + step_result.next_state.to_string())
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

    def show(self, run_id):
        print_divider('large')
        print '\tRUN #' + str(run_id) + ' RESULTS'
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
