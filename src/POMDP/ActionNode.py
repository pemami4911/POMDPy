__author__ = 'patrickemami'

import ObservationMapping as Om

class ActionNode(object):
    """
    Contains the ActionNode class, which represents a belief-action pair (i.e. the part of the
    belief tree corresponding to taking a given action from a specific belief.
    """
    def __init__(self, parent_entry=None):
        # parent_entry is type ActionMappingEntry
        if parent_entry is not None:
            self.parent_entry = parent_entry
        else:
            parent_entry = None
        self.observation_map = None

    def get_parent_belief(self):
        return self.parent_entry.get_mapping().get_owner()

    def get_child(self, obs):
        return self.observation_map.get_belief(obs)

    # -------------- internal methods ---------------- #

    # setter for Tree methods
    def set_mapping(self, obs_mapping):
        assert isinstance(obs_mapping, Om.ObservationMapping)
        self.observation_map = obs_mapping

    # returns belief node, boolean
    def create_or_get_child(self, obs):
        child_node = self.get_child(obs)
        added = False
        if child_node is None:
            child_node = self.observation_map.create_belief(obs)
            added = True
        return child_node, added



