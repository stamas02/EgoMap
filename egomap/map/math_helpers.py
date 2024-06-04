"""This module contains math related helper functions

More precisely it implements the summation of probabilities in log scale. 
"""

import numpy as np

__author__ = "Tamás Süveges"
__copyright__ = "Copyright 2019"
__credits__ = ["Tamás Süveges"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Tamás Süveges"
__email__ = "tsuveges@dundee.ac.uk"
__status__ = "Prototype"


def normalize_log_likelihoods(likelihoods, smoothing_factor=0.99):
    log_normalising_term = sum_log_probs(likelihoods)
    log_posterior = likelihoods - log_normalising_term
    posterior = np.exp(log_posterior)
    posterior = smoothing_factor * posterior + (1 - smoothing_factor) / len(posterior)
    return np.log(posterior)


def sum_log_probs(log_probs):
    """sum_log_prob is function that recieves a list of probabilities
    in log scale and performs an operation equivalent to the followinf expression:
        
        log(sum(exp(log_probs)))

    However this measure might lead to overflow when numbers are too small. To overcome
    this limitation this function makes use of sum_log_prob() implementation.

    Args:
        log_probs: list of floats

    Return:
    """
    if len(log_probs) == 1:
        return log_probs[0]

    x = sum_log_prob(log_probs[0], log_probs[1])
    for i in range(2, len(log_probs)):
        x = sum_log_prob(x, log_probs[i])
    return x


def sum_log_prob(a, b):
    """sum_log_prob is a function that summs two probabilities in log scale.
    Given tow log scale probabilites a and be it performs an operation equalent to
    log(exp(a)+exp(b)). In practice this expression can easily lead to overflow
    when a and/or b are very small. This function implements a safe way to calculate
    this expression even when a and/or b are small. 

    Args:
        a: float,
            log scale probability
        b: float,
            log scale probability

    Return:
        return a float number equalent to log(exp(a)+exp(b))
    """
    log1pexp = lambda x: np.log1p(np.exp(x))
    return a + log1pexp(b - a) if a > b else b + log1pexp(a - b)
