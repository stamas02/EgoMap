import numpy as np
from egomap.map.math_helpers import sum_log_probs, normalize_log_likelihoods
from egomap.map.graph import Graph
from egomap.map.view import BOVWView

__author__ = "Tamás Süveges"
__copyright__ = "Copyright 2019"
__credits__ = ["Tamás Süveges"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Tamás Süveges"
__email__ = "stamas01@gmail.com"
__status__ = "Prototype"


class Location:
    """Contains logic for location including probability calculation of each view and likelihood of the location.

    Parameters
    ----------
    observations: numpy array, optional
        A set of observations to initialize the location.
    min_view_observation_count: int, optional
        Minimum update count. Views with less updates are deleted.
    smoothing_factor: float, optional
        Smoothing factor used to smooth the output probability.

    Attributes
    ----------
    min_view_observation_count: int
        Minimum update count for views.
    smoothing_factor: float
        Smoothing factor for output probability.
    graph: graph.Graph
        A graph object containing nodes and edges.
    _previous_alfa: float
        Stores the previous log likelihood of the location for recursive use in the forward algorithm.
    """

    def __init__(self,
                 observations=None,
                 min_view_observation_count=0,
                 detector_true_negative_rate=0.68,
                 detector_true_positive_rate=0.68,
                 smoothing_factor=0.99):
        self.min_view_observation_count = min_view_observation_count
        self.smoothing_factor = smoothing_factor
        self.detector_true_negative_rate = detector_true_negative_rate
        self.detector_true_positive_rate = detector_true_positive_rate
        self.graph = Graph(constant_self_transition_probability=0.99)
        self._previous_alfa = None

        if observations is not None:
            new_view = BOVWView(np.mean(observations, axis=0),
                                detector_true_negative_rate=self.detector_true_negative_rate,
                                detector_true_positive_rate=self.detector_true_positive_rate)

            self.graph.add_node(new_view)
            self.update(observations)

    def _update_step(self, observation, log_prior):
        """Performs one update step given an observation.

        Parameters
        ----------
        observation: numpy array
            The new observation.
        log_prior: numpy array
            Prior belief on the views.

        Returns
        -------
        numpy array
            Log posterior on each view.
        """
        log_posterior = self.probaility_of_views(observation, log_prior)
        best_view_index = np.argmax(log_posterior)

        if best_view_index == 0:
            best_view_index = len(self.graph.nodes)
            new_view = BOVWView(observation,
                                detector_true_negative_rate=self.detector_true_negative_rate,
                                detector_true_positive_rate=self.detector_true_positive_rate)

            self.graph.add_node(new_view)
            log_posterior = np.append(log_posterior, log_posterior[0])
            log_posterior[0] = np.log(0.001)
        else:
            self.graph.nodes[best_view_index].update(observation)

        if log_prior is not None:
            from_node = np.argmax(log_prior)
            if from_node != best_view_index:
                self.graph.edges[from_node, best_view_index] += 1

        return log_posterior

    def update(self, observations):
        """Updates the location with a set of observations.

        Parameters
        ----------
        observations: numpy array
            New observations.
        """
        if len(self.graph.nodes) == 0:
            new_view = BOVWView(np.mean(observations, axis=0),
                                detector_true_negative_rate=self.detector_true_negative_rate,
                                detector_true_positive_rate=self.detector_true_positive_rate)
            self.graph.add_node(new_view)
        else:
            self.graph.nodes[0].update(np.mean(observations, axis=0))
        log_prior = None
        for observation in observations:
            log_prior = self._update_step(observation, log_prior)
        self._clean_short_views()

    def _clean_short_views(self):
        """Deletes views with less than minimum update count."""
        update_cnts = np.array([view.update_cnt for view in self.graph.nodes[1::]])
        to_delete = np.argwhere(update_cnts < self.min_view_observation_count).flatten()
        to_delete = -np.sort(-to_delete)
        for i in to_delete:
            self.graph.delete_node(i)

    def reset(self):
        """Resets the previous likelihood."""
        self._previous_alfa = None

    def log_likelihood(self, observation):
        """Calculates the log likelihood of the location using the recursive forward algorithm.

        Parameters
        ----------
        observation: numpy array
            New observation.

        Returns
        -------
        numpy array
            Log location likelihood.
        """
        self._previous_alfa = sum_log_probs(self.graph.log_likelihood_of_nodes(observation, self._previous_alfa)[1::])
        return self._previous_alfa

    def probaility_of_views(self, observation, log_prior=None):
        """Returns the probability for each view in this location.

        Parameters
        ----------
        observation: numpy array
            Observation.
        log_prior: numpy array, optional
            Prior belief.

        Returns
        -------
        numpy array
            Log probability distribution over all views given an observation.
        """
        un_log_view_probabilities = self.graph.log_likelihood_of_nodes(observation, log_prior)
        # un_log_view_probabilities[0] =  -5500
        log_posterior = normalize_log_likelihoods(un_log_view_probabilities, self.smoothing_factor)

        return log_posterior
