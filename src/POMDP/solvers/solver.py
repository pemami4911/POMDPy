__author__ = 'patrickemami'

import logging
from MCTS import *
from POMDP.statistic import Statistic
from POMDP.history import *
from util.console import *

module = "Solver"


class Results():
    time = Statistic("Total time")
    reward = Statistic("Total reward")
    discounted_return = Statistic("Discounted Reward")
    undiscounted_return = Statistic("Undiscounted Reward")

    def reset_running_totals(self):
        Results.time.running_total = 0.0
        Results.reward.running_total = 0.0
        Results.discounted_return.running_total = 0.0
        Results.undiscounted_return.running_total = 0.0


class Solver(object):
    """
    This class is responsible for initiating a run
    and storing statistics on that run
    """

    def __init__(self, model):
        self.logger = logging.getLogger('POMDPy.Solver')
        self.model = model
        self.action_pool = None
        self.observation_pool = None
        self.results = Results()
        # Collection of history sequences
        self.histories = Histories()

        self.initialize()

    def initialize(self):
        # reset these
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)

    def discounted_return(self):

        console(2, module, self.discounted_return.__name__, "Main runs")

        self.logger.info("Simulations\tRuns\tUndiscounted Return\tUndiscounted Error\t" +
                         "\tDiscounted Return\tDiscounted Error\tTime")

        self.multi_run()

        console(2, module, self.discounted_return.__name__, "Simulations = " + self.model.sys_cfg["num_sims"])
        console(2, module, self.discounted_return.__name__, "Runs = " + self.results.time.count)
        console(2, module, self.discounted_return.__name__,
                "Undiscounted Return = " + self.results.undiscounted_return.mean + " +- " +
                self.results.undiscounted_return.std_err())
        console(2, module, self.discounted_return.__name__,
                "Discounted Return = " + self.results.discounted_return.mean +
                " +- " + self.results.discounted_return.std_err())
        console(2, module, self.discounted_return.__name__, "Time = " + self.results.time.mean)

        self.logger.info(str(self.model.sys_cfg["num_sims"]) + '\t' + str(self.results.time.count) + '\t' +
                         '\t' + str(self.results.undiscounted_return.mean) + '\t' +
                         str(self.results.undiscounted_return.std_err()) + '\t' +
                         '\t' + str(self.results.discounted_return.mean) + '\t' +
                         str(self.results.discounted_return.std_err()) + '\t' +
                         '\t' + str(self.results.time.mean))

    def multi_run(self):
        num_runs = self.model.sys_cfg["num_runs"]

        for i in range(num_runs):

            console(2, module, self.multi_run.__name__, "Starting run " +
                    str(i + 1) + " with " + str(self.model.sys_cfg["num_sims"]) + " simulations")

            self.run()
            total_time = self.results.time.mean * self.results.time.count

            if total_time > self.model.sys_cfg["max_time_out"]:
                console(2, module, self.multi_run.__name__,
                        "Timed out after " + str(i) + " runs in " + total_time + " seconds")

    def run(self, num_steps=None):
        run_start_time = time.time()
        discount = 1.0
        out_of_particles = False

        if num_steps is None:
            num_steps = self.model.sys_cfg["num_steps"]

        # Reset the running total for each statistic for this run
        self.results.reset_running_totals()

        # Monte-Carlo start state
        state = self.model.sample_an_init_state()

        console(2, module, self.run.__name__, "Initial search state: " + state.to_string())

        # Create a new MCTS solver
        mcts = MCTS(self, self.model)

        # Perform sim behaviors that must done for each run
        self.model.reset_for_run()

        console(2, module, self.run.__name__,
                "num of particles generated = " + str(mcts.policy.root.state_particles.__len__()))

        for i in range(num_steps):
            start_time = time.time()

            if out_of_particles:
                console(2, module, self.run.__name__, "Out of particles; finishing sequence with rollout strategy")
                mcts.disable_tree = True

            # action will be of type Discrete Action
            action = mcts.select_action()

            step_result, is_legal = self.model.generate_step(state, action)

            self.results.reward.add(step_result.reward)
            self.results.undiscounted_return.running_total += step_result.reward
            self.results.discounted_return.running_total += (step_result.reward * discount)

            discount *= self.model.sys_cfg["discount"]
            state = step_result.next_state

            print_divider("large")
            console(2, module, self.run.__name__, "Step Number = " + str(i))
            console(2, module, self.run.__name__, "Step Result.Action = " + step_result.action.to_string())
            console(2, module, self.run.__name__, "Step Result.Observation = " + step_result.observation.to_string())
            console(2, module, self.run.__name__, "Step Result.Next_State = " + step_result.next_state.to_string())
            console(2, module, self.run.__name__, "Step Result.Reward = " + str(step_result.reward))

            if not step_result.is_terminal:
                out_of_particles = mcts.update(step_result)

            # Extend the history sequence
            new_hist_entry = mcts.history.add_entry()
            new_hist_entry.reward = step_result.reward
            new_hist_entry.action = step_result.action
            new_hist_entry.observation = step_result.observation
            new_hist_entry.register_entry(new_hist_entry, None, step_result.next_state)

            if step_result.is_terminal:
                console(2, module, self.run.__name__, "Terminated after episode step " + str(i))
                break

            if not out_of_particles:
                console(2, module, self.run.__name__,
                        "num of particles pushed over to new root = " + str(mcts.policy.root.state_particles.__len__()))

            console(2, module, self.run.__name__,
                    "MCTS step forward took " + str(time.time() - start_time) + " seconds")

        self.results.time.add(time.time() - run_start_time)
        self.results.discounted_return.add(self.results.discounted_return.running_total)
        self.results.undiscounted_return.add(self.results.undiscounted_return.running_total)

        # Pretty Print
        print_divider("large")
        mcts.history.show()
        print_divider("large")
        print "\tRUN RESULTS"
        print_divider("large")
        console(2, module, self.run.__name__, "Discounted Return statistics")
        print_divider("medium")
        self.results.discounted_return.show()
        print_divider("medium")
        console(2, module, self.run.__name__, "Un-discounted Return statistics")
        print_divider("medium")
        self.results.undiscounted_return.show()
        print_divider("medium")
        console(2, module, self.run.__name__, "Time")
        print_divider("medium")
        self.results.time.show()
        print_divider("medium")
        console(2, module, self.run.__name__,
                "Max possible total Un-discounted Return: " + str(self.model.get_max_undiscounted_return()))
        print_divider("medium")
