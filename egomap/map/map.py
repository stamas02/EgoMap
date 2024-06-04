"""This module is responsible for storing map capabilities.

More precisely a Map class is defined. It is responisble for:
    - Store location objects
    - Infer current location
    - Add new location 
    - Update location
"""

import numpy as np
from egomap.map.math_helpers import sum_log_probs, normalize_log_likelihoods
from egomap.map.location import Location

__author__ = "Tamás Süveges"
__copyright__ = "Copyright 2019"
__credits__ = ["Tamás Süveges"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Tamás Süveges"
__email__ = "tsuveges@dundee.ac.uk"
__status__ = "Prototype"


class EgoMap:
    """This class manages a map of locations during runtime."""

    def __init__(self,
                 new_location_likelihood_threshold,
                 minimum_observation_count = 3,
                 detector_true_negative_rate=0.68,
                 detector_true_positive_rate=0.68,
                 smoothing_factor=0.99):
        """Initializes the EgoMap.

        Parameters
        ----------
        new_location_likelihood_threshold: float
            Threshold for likelihood of new locations.
        smoothing_factor: float, optional
            Smoothing factor for probability calculations.
        """
        self.smoothing_factor = smoothing_factor
        self.new_location_likelihood_threshold = new_location_likelihood_threshold
        # self.graph_location = Graph()
        self.visit_locations = []
        self.transition_locations = []
        self.minimum_observation_count = minimum_observation_count
        self.observations = []
        self.log_location_prior = None
        self.log_prior = None
        self.log_posterior = None
        self.previous_location = None
        self.previous_transition = None
        self.detector_true_negative_rate = detector_true_negative_rate
        self.detector_true_positive_rate = detector_true_positive_rate
        self.location_transition_mapping = np.empty([0, 0], dtype=int)
        self.transition_location_mapping = np.empty([0, 0], dtype=int)
        self.in_transition = False

    def _reset(self):
        """Resets the alpha parameter for each location."""
        for location in self.visit_locations:
            location.reset()
        for location in self.transition_locations:
            location.reset()

    def infer(self, observation):
        """Infers the location based on the observation.

        Parameters
        ----------
        observation: numpy array
            The observation.

        Returns
        -------
        numpy array
            Log posterior probability distribution.
        """
        nodes = self.transition_locations if self.in_transition else self.visit_locations
        return self._infer(observation, nodes)

    def _infer(self, observation, nodes):
        """Performs inference based on the observation and nodes.

        Parameters
        ----------
        observation: numpy array
            The observation.
        nodes: list
            List of nodes for inference.

        Returns
        -------
        numpy array
            Log posterior probability distribution.
        """
        # Add observation to observations
        self.observations.append(observation)
        # Calculate log observation likelihood
        log_observation_likelihood = np.array([l.log_likelihood(observation) for l in nodes])
        # Concatenate log likelihoods with threshold for new locations
        bayse_nominator = log_observation_likelihood
        bayse_nominator = np.concatenate(
            [[self.new_location_likelihood_threshold * len(self.observations)], bayse_nominator])
        # Normalize log likelihoods
        self.log_posterior = normalize_log_likelihoods(bayse_nominator, self.smoothing_factor)
        return self.log_posterior

    def _log_posterior_to_log_prior(self, mapping):
        """Converts log posterior to log prior.

        Parameters
        ----------
        mapping: numpy array
            Transition mapping.
        """
        # t = np.clip(mapping, 0, 1)
        if 0 in mapping.shape:
            return
        np.seterr(divide='ignore', invalid='ignore')
        t = np.log(mapping / np.sum(mapping))
        t = np.nan_to_num(t)
        log_probability_matrix = t + np.reshape(self.log_posterior, (-1, 1))
        log_prior_likelihood = np.array(
            [sum_log_probs(column) for column in np.transpose(log_probability_matrix)])
        self.log_prior = normalize_log_likelihoods(log_prior_likelihood, self.smoothing_factor)
        pass

    def start_transition(self):
        """Starts a transition."""
        # Set the transition flag to True
        self.in_transition = True

        if self.log_posterior is None:
            self.observations = []
            self._log_posterior_to_log_prior(self.location_transition_mapping)
            self._reset()
            return

        # Find the best location based on log posterior
        best_location = np.argmax(self.log_posterior)

        # If the best location is a new location
        if best_location == 0:
            best_location = len(self.log_posterior)
            self._add_visit_location(self.observations)
            self.log_posterior = np.roll(self.log_posterior, -1)
        else:
            # Update the best location with observations
            self.visit_locations[best_location - 1].update(self.observations)
            # Normalize log posterior excluding the new location
            self.log_posterior = normalize_log_likelihoods(self.log_posterior[1::], self.smoothing_factor)

        self.previous_location = best_location

        # Update the mapping if there's a previous transition
        if self.previous_transition is not None:
            self.transition_location_mapping[self.previous_transition - 1, best_location - 1] += 1

        # Reset observations and update prior
        self.observations = []
        self._log_posterior_to_log_prior(self.location_transition_mapping)
        self._reset()

    def start_visit(self):
        """Starts a visit."""
        # Set the transition flag to False
        self.in_transition = False

        if self.log_posterior is None:
            self.observations = []
            self._log_posterior_to_log_prior(self.transition_location_mapping)
            self._reset()
            return

        # Find the best location based on log posterior
        best_location = np.argmax(self.log_posterior)

        # If the best location is a new location
        if best_location == 0:
            best_location = len(self.log_posterior)
            self._add_transition_location(self.observations)
            self.log_posterior = np.roll(self.log_posterior, -1)
        else:
            # Update the best location with observations
            self.transition_locations[best_location - 1].update(self.observations)
            # Normalize log posterior excluding the new location
            self.log_posterior = normalize_log_likelihoods(self.log_posterior[1::], self.smoothing_factor)

        self.previous_transition = best_location

        # Update the mapping if there's a previous location
        if self.previous_location is not None:
            self.location_transition_mapping[self.previous_location - 1, best_location - 1] += 1

        # Reset observations and update prior
        self.observations = []
        self._log_posterior_to_log_prior(self.transition_location_mapping)
        self._reset()

    def _add_transition_location(self, observations):
        """Adds a new transition location.

        Parameters
        ----------
        observations: numpy array
            Observations.
        """
        new_location = Location(observations=observations,
                                detector_true_negative_rate=self.detector_true_negative_rate,
                                detector_true_positive_rate=self.detector_true_positive_rate,
                                min_view_observation_count=self.minimum_observation_count)

        self.transition_locations.append(new_location)
        self.location_transition_mapping = np.lib.pad(self.location_transition_mapping, ((0, 0), (0, 1)),
                                                      'constant',
                                                      constant_values=0)
        self.transition_location_mapping = np.lib.pad(self.transition_location_mapping, ((0, 1), (0, 0)),
                                                      'constant',
                                                      constant_values=0)

    def _add_visit_location(self, observations, ):
        """Adds a new visit location.

        Parameters
        ----------
        observations: numpy array
            Observations.
        """
        new_location = Location(observations=observations,
                                detector_true_negative_rate=self.detector_true_negative_rate,
                                detector_true_positive_rate=self.detector_true_positive_rate,
                                min_view_observation_count=self.minimum_observation_count)
        self.visit_locations.append(new_location)
        self.location_transition_mapping = np.lib.pad(self.location_transition_mapping, ((0, 1), (0, 0)),
                                                      'constant',
                                                      constant_values=0)
        self.transition_location_mapping = np.lib.pad(self.transition_location_mapping, ((0, 0), (0, 1)),
                                                      'constant',
                                                      constant_values=0)
