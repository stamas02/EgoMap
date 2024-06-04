import numpy as np
import logging
from abc import ABC, abstractmethod

__author__ = "Tamás Süveges"
__copyright__ = "Copyright 2019"
__credits__ = ["Tamás Süveges"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Tamás Süveges"
__email__ = "stamas01@gmail.com"
__status__ = "Prototype"


class View(ABC):
    """Abstract view class containing basic view logic.

    Parameters
    ----------
    observation : numpy array
        A numpy matrix with dimensionality equal to the feature dimension.

    Attributes
    ----------
    view : numpy array
        The view model, a numpy matrix with dimensionality equal to the feature dimension.
    """

    def __init__(self, observation):
        self.view = np.array(observation).astype(float)

    @abstractmethod
    def update(self, observation):
        """Update the view with a new observation.

        Parameters
        ----------
        observation : numpy array
            The new observation.
        """
        pass

    @abstractmethod
    def log_likelihood(self, observation):
        """Calculate the log likelihood of the observation given this view.

        Parameters
        ----------
        observation : numpy array
            The new observation.
        """
        pass


class BOVWView(View):
    """Bag-of-Visual-Words view class containing relevant algorithms for bag-of-visual-words features.

    Parameters
    ----------
    observation : numpy array
        A numpy matrix with dimensionality equal to the feature dimension.
    detector_true_positive_rate : float, optional
        True positive rate of the detector.
    detector_true_negative_rate : float, optional
        True negative rate of the detector.

    Attributes
    ----------
    observation : numpy array
        A numpy matrix with dimensionality equal to the feature dimension.
    detector_true_positive_rate : float
        True positive rate of the detector.
    detector_true_negative_rate : float
        True negative rate of the detector.
    update_cnt : int
        The number of updates performed on the view so far.
    view : numpy array
        The view model, a numpy matrix with dimensionality equal to the feature dimension.
    """

    def __init__(self,
                 observation,
                 detector_true_positive_rate=1.0,
                 detector_true_negative_rate=1.0):
        super().__init__(observation)
        self.detector_true_positive_rate = detector_true_positive_rate
        self.detector_true_negative_rate = detector_true_negative_rate
        self.update_cnt = 1

    def update(self, observation):
        """Update the view with a new observation.

        Parameters
        ----------
        observation : numpy array
            The new observation.
        """
        self.update_cnt += 1
        self.view *= 1 - (1 / self.update_cnt)
        self.view += (1 / self.update_cnt) * observation

    def log_likelihood(self, observation):
        """Calculate the log likelihood of the observation given this view.

        Parameters
        ----------
        observation : numpy array
            The new observation.
        """
        observation = observation * self.detector_true_positive_rate + \
                      (1 - observation) * (1 - self.detector_true_negative_rate)

        observation_likelihood_of_bits = \
            observation * self.view + (1 - observation) * (1 - self.view)

        log_observation_likelihood = np.sum(np.log(observation_likelihood_of_bits), axis=0)

        if np.min(log_observation_likelihood) == float("-inf"):
            logging.warning("-inf occurred! This happens if the view and observation have opposite bits (0 and 1).")

        return log_observation_likelihood