__author__ = 'patrickemami'

import time
import logging
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories
from pomdpy.util import console, print_divider

module = "agent"


class Results(object):
    """
    Maintain the statistics for each run
    """
    time = Statistic("Total time")
    reward = Statistic("Total reward")
    discounted_return = Statistic("Discounted Reward")
    undiscounted_return = Statistic("Undiscounted Reward")

    @staticmethod
    def reset_running_totals():
        Results.time.running_total = 0.0
        Results.reward.running_total = 0.0
        Results.discounted_return.running_total = 0.0
        Results.undiscounted_return.running_total = 0.0

    @staticmethod
    def show():
        print_divider("large")
        print "\tRUN RESULTS"
        print_divider("large")
        console(2, module, "Discounted Return statistics")
        print_divider("medium")
        Results.discounted_return.show()
        print_divider("medium")
        console(2, module, "Un-discounted Return statistics")
        print_divider("medium")
        Results.undiscounted_return.show()
        print_divider("medium")
        console(2, module, "Time")
        print_divider("medium")
        Results.time.show()
        print_divider("medium")


def display_step_result(step_num, step_result):
    """
    Pretty prints step result information
    :param step_num:
    :param step_result:
    :return:
    """
    print_divider("large")
    console(2, module, "Step Number = " + str(step_num))
    console(2, module, "Step Result.Action = " + step_result.action.to_string())
    console(2, module, "Step Result.Observation = " + step_result.observation.to_string())
    console(2, module, "Step Result.Next_State = " + step_result.next_state.to_string())
    console(2, module, "Step Result.Reward = " + str(step_result.reward))


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
        self.results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset  # Factory method for generating instances of the solver

    def discounted_return(self):
        """
        Encapsulates logging and begins the runs
        :return:
        """
        console(2, module, "Main runs")

        self.logger.info("Simulations\tRuns\tUndiscounted Return\tUndiscounted Error\t" +
                         "\tDiscounted Return\tDiscounted Error\tTime")

        self.multi_run()

        console(2, module, "Simulations = " + str(self.model.sys_cfg["num_sims"]))
        console(2, module, "Runs = " + str(self.results.time.count))
        console(2, module, "Undiscounted Return = " + str(self.results.undiscounted_return.mean) + " +- " +
                str(self.results.undiscounted_return.std_err()))
        console(2, module, "Discounted Return = " + str(self.results.discounted_return.mean) +
                " +- " + str(self.results.discounted_return.std_err()))
        console(2, module, "Time = " + str(self.results.time.mean))

        self.logger.info(str(self.model.sys_cfg["num_sims"]) + '\t' + str(self.results.time.count) + '\t' +
                         '\t' + str(self.results.undiscounted_return.mean) + '\t' +
                         str(self.results.undiscounted_return.std_err()) + '\t' +
                         '\t' + str(self.results.discounted_return.mean) + '\t' +
                         str(self.results.discounted_return.std_err()) + '\t' +
                         '\t' + str(self.results.time.mean))

    def multi_run(self):
        num_runs = self.model.sys_cfg["num_runs"]

        for i in range(num_runs):

            console(2, module, "Starting run " +
                    str(i + 1) + " with " + str(self.model.sys_cfg["num_sims"]) + " simulations")

            self.run()
            total_time = self.results.time.mean * self.results.time.count

            if total_time > self.model.sys_cfg["max_time_out"]:
                console(2, module, "Timed out after " + str(i) + " runs in " + total_time + " seconds")

    def run(self, num_steps=None):
        run_start_time = time.time()
        discount = 1.0

        if num_steps is None:
            num_steps = self.model.sys_cfg["num_steps"]

        # Reset the running total for each statistic for this run
        self.results.reset_running_totals()

        # Create a new solver
        solver = self.solver_factory(self, self.model)

        # Perform sim behaviors that must done for each run
        self.model.reset_for_run()

        console(2, module, "num of particles generated = " + str(solver.belief_tree.root.state_particles.__len__()))

        if solver.on_policy:
            solver.policy_iteration()

        # Monte-Carlo start state
        state = self.model.sample_an_init_state()
        console(2, module, "Initial search state: " + state.to_string())

        for i in range(num_steps):
            start_time = time.time()

            # action will be of type Discrete Action
            action = solver.select_action()

            step_result, is_legal = self.model.generate_step(state, action)

            self.results.reward.add(step_result.reward)
            self.results.undiscounted_return.running_total += step_result.reward
            self.results.discounted_return.running_total += (step_result.reward * discount)

            discount *= self.model.sys_cfg["discount"]
            state = step_result.next_state

            # show the step result
            display_step_result(i, step_result)

            if not step_result.is_terminal:
                solver.update(step_result)

            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            new_hist_entry.reward = step_result.reward
            new_hist_entry.action = step_result.action
            new_hist_entry.observation = step_result.observation
            new_hist_entry.register_entry(new_hist_entry, None, step_result.next_state)

            if step_result.is_terminal:
                console(2, module, "Terminated after episode step " + str(i))
                break

            console(2, module, "MCTS step forward took " + str(time.time() - start_time) + " seconds")

        self.results.time.add(time.time() - run_start_time)
        self.results.discounted_return.add(self.results.discounted_return.running_total)
        self.results.undiscounted_return.add(self.results.undiscounted_return.running_total)

        # Pretty Print results
        print_divider("large")
        solver.history.show()
        self.results.show()
        console(2, module, "Max possible total Un-discounted Return: " + str(self.model.get_max_undiscounted_return()))
        print_divider("medium")