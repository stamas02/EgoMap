import numpy as np
from egomap.map.math_helpers import sum_log_probs

__author__ = "Tamás Süveges"
__copyright__ = "Copyright 2019"
__credits__ = ["Tamás Süveges"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Tamás Süveges"
__email__ = "tsuveges@dundee.ac.uk"
__status__ = "Prototype"


class Graph():
    """ Contains nodes and the connections between them.

    Parameters
    ----------
    default_edge : float
        The default edge weight

    Attributes
    ----------
    nodes : list of view.View object
        A list of view object.
    edges : numpy array
        an adjacency matrix.
    """

    def __init__(self, default_edge = 10, constant_self_transition_probability = None):
        self.nodes = []
        self.default_edge = default_edge
        self.edges = np.empty([0, 0])
        self.constant_self_transition_probability = constant_self_transition_probability
        pass

    def log_likelihood_of_nodes(self, observation, log_prior=None):
        """Calculates the likelihood of each node given an observation and a prior.

        Args:
            observation: numpy array,
                query observation
            log_prior: numpy array,
                It should be a log probability distribution. If None
                then a uniform distribution will be used. 

        Return:
            A numpy array of likelihoods of each node.
        """

        if len(self.nodes) == 0:
            return np.array([])

        # log likelihood based on appearance
        log_observation_likelihood = [node.log_likelihood(observation) for node in self.nodes]

        # a stochastic matrix where the (i,j) element gives the probability of
        # transitioning from the ith node to jth node based on the transition matrix.

        log_transition_probability_matrix = np.log(self._edge_normals())
        if not log_prior is None and log_prior.size > 0:
            log_transition_probability_matrix += log_prior

        # summing the log_prior_matrix column-wise in the probability space (NOT LOG) gives the provability
        # of being in each node based only on the transition matrix and prior belief.
        # Note that since log space is used a summation is applied instead a product.
        log_transition_probability = np.array(
            [sum_log_probs(column) for column in np.transpose(log_transition_probability_matrix)])

        # The product of observation_likelihood and transition_probability gives the likelihood of each node
        # Summation is used instead of product because calculation takes place in the log space.
        return np.array(log_observation_likelihood + log_transition_probability)

    def delete_node(self, index):
        """ Deletes a node at index

        Parameters
        ----------
        index : int
            the index of the node to be deleted
        """
        self.edges = np.delete(self.edges, index, axis=0)
        self.edges = np.delete(self.edges, index, axis=1)
        del self.nodes[index]


    def add_node(self, node):
        """add_node adds a node to the hmm

        Args:
            node: Object,
                the new node desired to be added to the hmm
        """
        self.nodes.append(node)
        self.edges = np.lib.pad(self.edges,
                                ((0, 1), (0, 1)),
                                'constant',
                                constant_values=self.default_edge)

    def _edge_normals(self):
        """ Normalizes the edges of the Graph. If constant_self_transition_probability is not None
            then self transition (diag of the matrix) will always be constant.

        Returns
        -------
        numpy array
            The normalized edges.
        """
        if self.constant_self_transition_probability is None:
            return self.edges / np.sum(self.edges)
        else:
            edges_wo_self_transition = np.array(self.edges)
            np.fill_diagonal(edges_wo_self_transition, 0)
            np.seterr(divide='ignore', invalid='ignore')
            normalizing_factor = np.expand_dims(np.sum(edges_wo_self_transition, axis=1), axis=1) / (1-self.constant_self_transition_probability)
            norm = edges_wo_self_transition / normalizing_factor
            np.fill_diagonal(norm, self.constant_self_transition_probability)
            return norm / np.sum(norm)
        #return self.edges / np.expand_dims(np.sum(self.edges, axis=1), axis=1)
