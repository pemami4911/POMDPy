__author__ = 'patrickemami'

import objgraph
import RockModel
import Solver
import MCTS

model = RockModel.RockModel("Objgraph Test")
solver = Solver.Solver(model)
mcts = MCTS.MCTS(solver, model)

def add_to_tree():
        """
        Testing a pruning feature of the search tree
        :return:
        """
        global mcts
        global model

        # sample an init state
        rand_state = model.sample_an_init_state()
        # Sample a random action
        rand_action = model.get_random_action()
        # Generate an observation by generating a step
        step_result, is_legal = model.generate_step(rand_state, rand_action)
        # Create a child belief node
        child, added = mcts.policy.root.create_or_get_child(rand_action, step_result.observation)
        # clear out the entire tree ?
        #objgraph.show_chain(objgraph.find_backref_chain(child, objgraph.is_proper_module), filename='chain.png')

if __name__ == '__main__':

        for i in range(5):
                add_to_tree()

        #objgraph.show_refs(mcts.policy.root, filename='policy1.png')
        mcts.policy.prune_tree(mcts.policy)
        #mcts.policy.root.action_map.owner = None
        #mcts.policy.root = None
        #objgraph.show_backrefs(child, filename='child_node.png')
        #objgraph.show_refs(mcts.policy.root, filename='policy2.png')
