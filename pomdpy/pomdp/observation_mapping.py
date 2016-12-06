from builtins import object
from future.utils import with_metaclass
import abc


class ObservationMapping(with_metaclass(abc.ABCMeta, object)):
    """
    An abstract base class that defines a mapping of observations to subsequent beliefs in the
    belief tree.

    Each of these edges must also store the statistics for that edge - in this case, this only
    consists of the visit count for that edge.
    """

    def __init__(self, action_node):
        self.owner = action_node

    #  -------------- Access to and management of child nodes. ---------------- #
    @abc.abstractmethod
    def get_belief(self, observation):
        """
        Retrieves the belief node (if any) corresponding to this observation
        :param observation:
        :return:
        """

    @abc.abstractmethod
    def create_belief(self, observation):
        """
        Creates a new belief node for the given observation
        :param observation:
        :return:
        """

    @abc.abstractmethod
    def delete_child(self, obs_mapping_entry):
        """
        Deletes the given entry from this mapping, as well as the entire corresponding subtree.
        :param obs_mapping_entry:
        :return:
        """

    # -------------- Retrieval of mapping entries. ----------------
    @abc.abstractmethod
    def get_child_entries(self):
        """
        Returns a list of all the entries in this mapping that have an associated child node.
        :return:
        """

    @abc.abstractmethod
    def get_entry(self, obs):
        """
        Returns the mapping entry associated with the given observation
        :param obs:
        :return:
        """

class ObservationMappingEntry(with_metaclass(abc.ABCMeta, object)):
    """
    An interface that represents an edge in the belief tree between an action node and a
    subsequent belief node; this interface is provided so that observations can be grouped together
    in custom ways.

    Conceptually, this corresponds to a (belief, action, observation) triplet (b, a, o), or,
    equivalently, it can be seen as the parent edge of the resulting belief (b').

    Apart from grouping observations together, the primary purpose of this entry is to store
    a visit count - i.e. the number of times this edge has been visited during searching.
    """


    @abc.abstractmethod
    def get_observation(self):
        """
        Returns the observation for this entry.
        :return: Observation
        """

    @abc.abstractmethod
    def update_visit_count(self, delta_n_visits):
        """
        Updates the visit count for this observation.
        :param delta_n_visits:
        :return:
        """
