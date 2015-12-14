__author__ = 'patrickemami'

from pomdpy.examples.rock_problem import RockModel
from pomdpy.solvers import Solver, MCTS

'''
TEST 0 - Pruning
TEST 1 - Action Selection after simulation
'''
TEST = 1

model = RockModel("Objgraph Test")
solver = Solver(model)
mcts = MCTS(solver, model)

def add_to_tree():
        global mcts
        global model

        # sample an init state
        rand_state = model.sample_an_init_state()
        # Sample a random action
        rand_action = solver.action_pool.sample_random_action()
        # Generate an observation by generating a step
        step_result, is_legal = model.generate_step(rand_state, rand_action)
        # Create a child belief node
        mcts.policy.root.create_or_get_child(rand_action, step_result.observation)

def simulation():

        # Run the solver on 1 step
        solver.run(1)

if __name__ == '__main__':
        if TEST == 0:
            # Add 5 belief nodes to the tree
            for i in range(5):
                    add_to_tree()

            # Prune the tree
            mcts.policy.prune_tree(mcts.policy)

            assert mcts.policy.root is None
        elif TEST == 1:
            simulation()
        else:
            print "No such test defined yet"

