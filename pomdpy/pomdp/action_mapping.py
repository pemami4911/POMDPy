from builtins import object
from future.utils import with_metaclass
import abc


class ActionMapping(with_metaclass(abc.ABCMeta, object)):
    """
    Action Mapping abc for discrete and continuous actions
    """

    def __init__(self, belief_node):
        self.owner = belief_node

    @abc.abstractmethod
    def get_action_node(self, action):
        """
        Retrieves the action node (if any) corresponding to the given action.
        :param action:
        :return action_node:
        """


    @abc.abstractmethod
    def create_action_node(self, action):
        """
        Creates a new action node for the given action.
        :param action:
        :return action_node:
        """

    @abc.abstractmethod
    def delete_child(self, action_mapping_entry):
        """
        Deletes the child in the given entry, as well as the entire corresponding subtree
        :param action_mapping_entry:
        :return:
        """

    @abc.abstractmethod
    def get_child_entries(self):
        """
        Returns all entries in this mapping that have a child node associated with them
        :return: List of ActionMappingEntry's
        """

    @abc.abstractmethod
    def get_visited_entries(self):
        """
        Returns a vector of all of the visited entries in this mapping
        Some of those entries might have null action nodes if the visit counts were initialized
        with nonzero values.
        :return:
        """

    @abc.abstractmethod
    def get_entry(self, action):
        """
        Returns the mapping entry associated with the given action, or None if there is none.
        :param action:
        :return:
        """

    @abc.abstractmethod
    def get_next_action_to_try(self):
        """
        Returns the next unvisited action to be tried for this node, or None if there are no
        more unvisited actions (that are legal).
        :param:
        :return action:
        """


class ActionMappingEntry(with_metaclass(abc.ABCMeta, object)):
    """
    An interface for discrete and continuous actions
    that represents a (belief, action) edge in the belief tree.

    There are two core pieces of functionality - a number of getter methods returning various
    properties of this edge, as well as, more importantly
    update_visit_count(), update_q_value(), which updates the visit count and/or Q-value for this edge
    """

    @abc.abstractmethod
    def update_visit_count(self, delta_n_visits):
        """
        :param delta_n_visits:
        :return: visit count
        """

    @abc.abstractmethod
    def update_q_value(self, delta_total_q):
        """
        :param delta_n_visits:
        :param delta_total_Q:
        :return true iff the Q value changed:
        """

    @abc.abstractmethod
    def set_legal(self, legal):
        """
        :param legal: bool
        Sets the legality of this action - this determines whether or not it will be taken in the
        course of *future* searches.

        In and of itself, making this action illegal will not delete previous histories that have
        already taken this action. In order to achieve this the associated history entries also
        need to be marked for updating via the model-changing interface.

        This feature is currently not used
        :return:
        """

    @abc.abstractmethod
    def get_action(self):
        """
        Returns the action for this entry.
        :return Action:
        """

