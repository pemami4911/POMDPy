__author__ = 'patrickemami'

import random

class BeliefNode:
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

        # Nearest Neighbor heuristic
        # self.max_nn_distance = self.solver.model.sys_cfg["max_nn_distance"]
        # self.max_n_comparisons = self.solver.model.sys_cfg["max_n_comparisons"]
        # The closest neighbor found so far for this node
        # self.neighbor = None

        if parent_entry is not None:
            self.parent_entry = parent_entry
            # Correctly calculate the depth based on the parent node.
            self.depth = self.get_parent_belief().depth + 1
        else:
            self.parent_entry = None
            self.depth = 0

        # Add this node to the index in the tree.
        #self.solver.policy.add_node(self)

    def copy(self, id=None, parent_entry=None):
        return BeliefNode(self.solver, id, parent_entry)

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
    '''
    def distance(self, other_belief_node):
        """
        Distance metric defined on belief nodes to estimate the relative "distance" between each other
        for a nearest-neighbor search heuristic
        :param other_belief_node:
        :return: average_dist (double)
        """
        non_trans_action = self.solver.model.config["non-transferable-action"]

        dist = 0.0
        for entry1 in self.particles:
            for entry2 in other_belief_node.particles:
                # Have to have the same position
                if not entry2.state.position.equals(entry1.state.position):
                    dist = BIG_NUM
                    break

                # For the RockProblem, Sampling a rock is non-transferable, since it is
                # illegal in all states except those in which the grid contains a rock
                if entry2.action.bin_number == non_trans_action:
                    dist = BIG_NUM
                    break

                # L1 norm between states
                dist += entry1.state.distance_to(entry2.state)

        average_dist = dist / (self.particles.__len__() + other_belief_node.particles.__len__())
        assert average_dist >= 0
        return average_dist
    '''
    '''
    def add_particle(self, new_history_entry):
        self.particles.append(new_history_entry)
        if new_history_entry.id is 0:
            self.n_starting_sequences += 1

    def remove_particle(self, hist_entry):
        self.particles.remove(hist_entry)
        if hist_entry.id is 0:
            self.n_starting_sequences -= 1

    def nearest_neighbor_probability(self):
        """
        Determines whether the nearest neighbor heuristic should be used
        :return:
        """
        #return 1.0 /(self.solver.n_episodes**2)*self.solver.current_episode**2
        return 1/(1 + np.exp(-self.solver.sigmoid_steepness *
                               (self.solver.current_episode - (self.solver.n_episodes/2))))

    def find_neighbor(self):
        """
        Finds an approximate nearest neighbor for the given belief node
        :param belief:
        :return:
        """
        if self.nearest_neighbor_probability() < np.random.uniform(0, 1):
            return None

        # This function is disabled
        if self.max_nn_distance < 0:
            return None

        min_dist = np.inf
        if self.neighbor is not None:
            min_dist = self.distance(self.neighbor)

        num_tried = 0
        nearest_belief = None
        for other_belief in self.solver.policy.all_nodes.values():
            # Ignore this belief
            if self == other_belief:
                continue

            # Couldn't find a NN
            if num_tried >= self.max_n_comparisons:
                break
            else:
                dist = self.distance(other_belief)
                if dist < min_dist:
                    min_dist = dist
                    nearest_belief = other_belief
                num_tried += 1

        if min_dist > self.max_nn_distance:
            return None

        self.neighbor = nearest_belief
        return nearest_belief
    '''
