__author__ = 'patrickemami'

# see search_interface.hpp for basic-search-strategy, which I need to customize

class BeliefNode:
    """
    Represents a single node in a belief tree.
    *
    * The key functionality is a set of all the particles associated with this belief node, where
    * each particle is a pointer to a HistoryEntry.
    *
    * Additionally, a belief node owns an ActionMapping, which stores the actions that have been
    * taken from this belief, as well as their associated statistics and subtrees.
    *
    * The belief node can also store a vector of cached values, which is convenient if you want
    * to cache values that are derived from the contents of the belief via a relatively expensive
    * calculation.
    *
    * This caching is also particularly useful for incremental updates - the cached value can be
    * compared to its new value after recalculation, and then the change in value can be easily
    * back-propagated.
    """
    def __init__(self, solver, id=None, parent_entry=None):
        if id is None:
            self.id = -1
        else:
            self.id = id

        self.solver = solver

        self.data = None    # The smart history-based data, to be used for history-based policies.
        self.particles = []     # The set of particles belonging to this node.
        self.depth = -1
        self.n_starting_sequences = 0
        self.action_map = None

        if parent_entry is not None:
            self.parent_entry = parent_entry
            # Correctly calculate the depth based on the parent node.
            self.depth = self.get_parent_belief().depth + 1
        else:
            self.parent_entry = None
            self.depth = 0

        # Add this node to the index in the tree.
        self.solver.policy.add_node(self)


    def get_number_of_particles(self):
        return self.particles.__len__()

    def get_states(self):
        states = []
        for entry in self.particles:
            states.append(entry.get_state())
        return states

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

    # ----------- Value estimation and action selection -------------- #

    # def get_recommended_action

    # ----------- Core Method -------------- #

    def create_or_get_child(self, action, obs):
        """
        Adds a child for the given action and observation, or returns a pre-existing one if it
        already existed.

        The belief node will also be added to the flattened node vector of the policy tree, as
        this is done by the BeliefNode constructor.
        :param action:
        :param obs:
        :return:
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
        #else:
            # Update the current action mapping to reflect the state of the simulation
            # child_node.action_map.update()
        return child_node

    def add_particle(self, new_history_entry):
        self.particles.append(new_history_entry)
        if new_history_entry.id is 0:
            self.n_starting_sequences += 1

    def remove_particle(self, hist_entry):
        self.particles.remove(hist_entry)
        if hist_entry.id is 0:
            self.n_starting_sequences -= 1



