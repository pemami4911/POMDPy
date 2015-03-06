__author__ = 'patrickemami'

import ObservationMapping as Om
import BeliefNode as Bn
import ActionNode as An
import DiscreteObservation as Do
import logging

class DiscreteObservationMap(Om.ObservationMapping):
    """
    A concrete class implementing ObservationMapping for a discrete set of observations.
    *
    * The mapping entries are stored in a dictionary, which maps each observation
    * to its associated entry in the mapping
    * The dictionary returns None if an observation is not yet stored in the dictionary
    """
    def __init__(self, action_node, solver):
        super(DiscreteObservationMap, self).__init__(action_node)
        self.solver = solver
        self.child_map = {}
        self.total_visit_count = 0
        self.logger = logging.getLogger('Model.DiscreteObservationMapping')

    def get_belief(self, disc_observation):
        assert isinstance(disc_observation, Do.DiscreteObservation)

        entry = self.get_entry(disc_observation)
        if entry is None:
            return None
        else:
            return entry.child_node

    def create_belief(self, disc_observation):
        assert isinstance(disc_observation, Do.DiscreteObservation)

        entry = DiscreteObservationMapEntry()
        entry.map = self
        entry.observation = disc_observation
        entry.child_node = Bn.BeliefNode(self.solver, None, entry)

        self.child_map.__setitem__(entry.observation, entry)
        return entry.child_node

    def get_n_children(self):
        return self.child_map.__len__()

    def delete_child(self, obs_mapping_entry):
        assert isinstance(obs_mapping_entry, DiscreteObservationMapEntry)

        self.total_visit_count -= obs_mapping_entry.visit_count
        del self.child_map[obs_mapping_entry.observation]

    def get_child_entries(self):
        return_entries = []
        for key in self.child_map.keys():
            return_entries.append(self.child_map.get(key))
        return return_entries

    def get_entry(self, obs):
        entries = self.child_map.values()
        for i in entries:
            if obs.equals(i.observation):
                return i
        return None


class DiscreteObservationMapEntry(Om.ObservationMappingEntry):
    """
    A concrete class implementing ObservationMappingEntry for a discrete set of observations.
    *
    * Each entry stores a pointer back to its parent map, the actual observation it is associated with,
    * and its child node and visit count.
    """

    def __init__(self):
        self.map = None  # DiscreteObservationMap
        self.observation = None     # DiscreteObservation
        # The child node of this entry (should always be non-null).
        self.child_node = None  # belief node
        self.visit_count = 0

    def get_observation(self):
        return self.observation.copy()

    def update_visit_count(self, delta_n_visits):
        self.visit_count += delta_n_visits
        self.map.total_visit_count += delta_n_visits


