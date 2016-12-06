from builtins import object
import random


class BeliefNode(object):
    """
    Represents a single node in a belief tree.
    *
    * The key functionality is a set of all the state particles associated with this belief node, where
    * each particle is a pointer to a DiscreteState.
    *
    * Additionally, a belief node owns an ActionMapping, which stores the actions that have been
    * taken from this belief, as well as their associated statistics and subtrees.
    *
    * Key method is create_or_get_child()
    """
    def __init__(self, solver, id=None, parent_entry=None):
        if id is None:
            self.id = -1
        else:
            self.id = id

        self.solver = solver
        self.data = None    # The smart history-based data, to be used for history-based policies.
        self.depth = -1
        self.action_map = None
        self.state_particles = []   # The set of states that comprise the belief distribution of this belief node

        if parent_entry is not None:
            self.parent_entry = parent_entry
            # Correctly calculate the depth based on the parent node.
            self.depth = self.get_parent_belief().depth + 1
        else:
            self.parent_entry = None
            self.depth = 0

    def copy(self):
        bn = BeliefNode(self.solver, self.id, self.parent_entry)
        # copy the data
        bn.data = self.data.copy()
        # share a reference to the action map
        bn.action_map = self.action_map
        bn.state_particles = self.state_particles
        return bn

    # Randomly select a History Entry
    def sample_particle(self):
        return random.choice(self.state_particles)

    # -------------------- Tree-related getters  ---------------------- #
    def get_parent_action_node(self):
        if self.parent_entry is not None:
            return self.parent_entry.map.owner
        else:
            return None

    def get_parent_belief(self):
        if self.parent_entry is not None:
            return self.parent_entry.map.owner.parent_entry.map.owner
        else:
            return None

    # Returns the last observation received before this belief
    def get_last_observation(self):
        if self.parent_entry is not None:
            self.parent_entry.get_observation()
        else:
            return None

    def get_last_action(self):
        if self.parent_entry is not None:
            return self.parent_entry.map.owner.parent_entry.get_action()
        else:
            return None

    def get_child(self, action, obs):
        node = self.action_map.get_action_node(action)
        if node is not None:
            return node.get_child(obs)
        else:
            return None

    def child(self, action, obs):
        node = self.action_map.get_action_node(action)
        if node is not None:
            child_node = node.get_child(obs)
            if child_node is None:
                return None
            child_node.data.update(child_node.get_parent_belief())
            return child_node
        else:
            return None

    # ----------- Core Methods -------------- #

    def create_or_get_child(self, action, obs):
        """
        Adds a child for the given action and observation, or returns a pre-existing one if it
        already existed.

        The belief node will also be added to the flattened node vector of the policy tree, as
        this is done by the BeliefNode constructor.
        :param action:
        :param obs:
        :return: belief node
        """
        action_node = self.action_map.get_action_node(action)
        if action_node is None:
            action_node = self.action_map.create_action_node(action)
            action_node.set_mapping(self.solver.observation_pool.create_observation_mapping(action_node))
        child_node, added = action_node.create_or_get_child(obs)

        if added:   # if the child node was added - it is new
            if self.data is not None:
                child_node.data = self.data.create_child(action, obs)
            child_node.action_map = self.solver.action_pool.create_action_mapping(child_node)
        else:
            # Update the current action mapping to reflect the state of the simulation
            # child_node.action_map.update()
            # self.solver.model.num_reused_nodes += 1

            # Update the re-used child belief node's data
            child_node.data.update(child_node.get_parent_belief())
        return child_node, added
