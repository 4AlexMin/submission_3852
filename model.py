import kldloss

import math
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from prda import prep, stats
from prda.ml.neighbors import propogation_algorighm
from scipy.optimize import LinearConstraint



__version__ = '2.0.0'
class SocialBucks(BaseEstimator):
    """
    When using the word `Vertex`, it will only be referred to `class Vertex` in `vertex.py`.
    """

    def __init__(
        self,
        kappa: float = 10,
        eta: float = 0.5,
        distance_measurement: str ='minkowski',
        propagation_algo = 'mode',
        kernel_method: str = 'kld_constrained',
        k_formula = 'linear',
        upper_sliding = 1.5,
        mutual_graph = False,
        output_info=False,
        ) -> None:
        """_summary_

        Parameters
        ----------
        kappa : float, optional
            _description_, by default 10
        eta : float, optional
            _description_, by default 0.5
        distance_measurement : str, optional
            _description_, by default 'minkowski'
        propagation_algo : str, optional
            _description_, by default 'mode'
        kernel_method : str, optional
            _description_, by default 'kld_constrained'
        k_formula : str, optional
            _description_, by default 'linear'
        upper_sliding : float, optional
            the upper limit for F, with unit of k-multiple, by default 0.5
            if not None, F will be between ((upper_sliding-1) * kappa, upper_sliding * kappa)
        """

        # Network params
        self.kappa = kappa
        self.eta = eta
        self.distance_measurement = distance_measurement
        self.propagation_algo = propagation_algo
        self.kernel_method = kernel_method
        self.k_formula = k_formula
        self.upper_sliding = upper_sliding
        self.mutual_graph = mutual_graph
        self.__output_info = output_info

        # Attrs.
        self.graph = nx.Graph()    #  Vertices are represented as 0, 1, 2, ..., which is also the index of X, hence the indice of `fitness_kernel`
        self.fitness_kernel = None
        self.knnModel = None

        
    

    def __initialize_knn(self, input_matrix) -> NearestNeighbors:
        """ No data preprocessing will be made here.
        """
        if self.distance_measurement == 'mahalanobis':
            metric_params = {'VI': np.linalg.inv(np.cov(input_matrix.T))}
        elif self.distance_measurement == 'minkowski':
            metric_params = None
        else:
            raise KeyError(self.distance_measurement, ' is not a valid measurement.')
        knnModel = NearestNeighbors(metric=self.distance_measurement, metric_params=metric_params).fit(input_matrix)
        return knnModel



        
    def __add_edge_(self, alpha, beta):
        if self.mutual_graph:
            if (beta, alpha) not in self._direct_edges:
                self._direct_edges.add((alpha, beta))
            else:
                edge_weight = self.fitness_kernel[alpha] * self.fitness_kernel[beta] / self.kernel_sum
                self.graph.add_edge(alpha, beta, weight=edge_weight)
        else:
            if not self.graph.has_edge(alpha, beta):
                edge_weight = self.fitness_kernel[alpha] * self.fitness_kernel[beta] / self.kernel_sum
                self.graph.add_edge(alpha, beta, weight=edge_weight)
        
    def __update_knn(self):
        """At the moment, do nothing. BecauseOF:: AttributeError: 'NearestNeighbors' object has no attribute 'partial_fit'
        """
        pass

    def __update_g(self):
        pass



    def __k_method(self, alpha)-> int:
        """
        Returns
        -------
        int
            k, the potential neighbor number, also `k` in `KNN`.
        """
        if self.k_formula.lower() == 'exp':
            k =  min(len(self.fitness_kernel), math.ceil(self.fitness_kernel[alpha]**self.eta)*self.kappa)
        elif self.k_formula.lower() == 'linear':
            k = min(len(self.fitness_kernel), math.ceil(linear_kernel(self.kappa, self.eta, self.fitness_kernel[alpha])))
        
        # k should be at least 1
        k = max(1, k)
        return k
    
    def __knng_single(self, alpha, x_alpha) -> None:
        """

        """
        x_alpha = x_alpha.reshape(1, -1)
        k_alpha = self.__k_method(alpha)
        nbr_nodes = self.knnModel.kneighbors(x_alpha, n_neighbors=k_alpha+1, return_distance=False)[0][1:]    # self excluded
        for beta in nbr_nodes:
            self.__add_edge_(alpha, beta)
        if self.__output_info:
            print('alpha:', alpha, 'k:', k_alpha, '#'+str(len(nbr_nodes))+'_pnbrs:', nbr_nodes)
    
    def construct_knng(self, X) -> None:
        self._direct_edges = set()
        for alpha in self.graph.nodes:
            x_alpha = X[alpha]
            self.__knng_single(alpha, x_alpha)


    
    def fit(self, X, y):
        """Graph Consruction

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Graph as Classifier
        """
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.DataFrame:
            y = y.values
        
        # Generate unnormalized fitness kernel
        F_init = None
        if 'kld' in self.kernel_method:
            unique_values, counts = np.unique(y, return_counts=True)
            counts_dict = dict(zip(unique_values, counts))
            count_mapping = np.vectorize(lambda x: np.log(counts_dict[x]))
            F_init = count_mapping(y)
        raw_kernels = generate_fitness_kernel(X, y, method=self.kernel_method, F_init=F_init)

        if self.upper_sliding:
            kernel_values = prep.normalization(raw_kernels, min_=(self.upper_sliding - 1)*self.kappa, max_=self.upper_sliding*self.kappa, bias=0)
        else:
            kernel_values = raw_kernels
        
        self.fitness_kernel =  {_: kernel_values[_][0]  for _ in range(len(X))}
        self.kernel_sum = sum(self.fitness_kernel.values())
        
        # Construct kNN
        self.knnModel = self.__initialize_knn(X)

        # Construct kNN graph
        self.graph.add_nodes_from([(_, dict(label=y[_])) for _ in range(len(X))])    # fit samples are represented as 0, 1, 2, ..., which is also the index of X
        pcs = prep.pca(data=X, npcs=2)
        self.__graph_pos = {i: pcs[i].tolist() for i in range(len(pcs))}
        self.construct_knng(X)
        



    def predict(self, X):
        if self.knnModel is None:
            raise RuntimeError("fit() method must be called before predict().")
        X_closest = self.knnModel.kneighbors(X, n_neighbors=1, return_distance=False)    # `X_closest` contains the indices of the closest data point in the training data(i.e., self.X) to each data point in the test data(i.e., X), which is also the node representation in self.graph
        y_pred = propogation_algorighm(graph=self.graph, X_closest=X_closest)

        # If decide to update knn and graph after each prediction, the `X_closest` approach may contain false indices.
        self.__update_knn()
        self.__update_g()
        return y_pred
    
    
    def get_params(self, deep=True):
        return {
            "eta": self.eta,
            "kappa": self.kappa,
            "kernel_method": self.kernel_method, 
            "k_formula": self.k_formula, 
            "propagation_algo": self.propagation_algo, 
            'upper_sliding': self.upper_sliding,
            'mutual_graph': self.mutual_graph
            }
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        


def generate_fitness_kernel(X, y=None, method: str = 'kld', bias=0.1, F_init: np.ndarray = None) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    X : _type_
        _description_
    method : str, optional
        _description_, by default 'pca'
    bias : float, optional
        Because E(edge_weight) for normal nodes are almost zero, by default 0.1
    F_init : numpy,ndarray, optional
        Currently, only support when `method` set to 'kld'
    Returns
    -------
    np.ndarray
        array of kernel values in shape (length_of_X, 1)
    """
    
    if method.lower() == 'pca':
        raw_kernels = prep.normalization(prep.pca(data=X, npcs=1), bias=bias)
        
    elif method.lower() == 'linear':
        raw_kernels = prep.apply_linear_func(X, bias=bias)
    
    elif method.lower() == 'kld':
        raw_kernels = kldloss.KLProjection(init='customized', F_init=F_init).fit_transform(X)
    
    elif ''.join(method.lower().split('_')) == 'kldconstrained':
        custom_constraints = []
        distances = compute_distances(X, y)
        X_sub, y_sub, indices_sub = get_border_samples(X, y, return_indices=True)
        
        # Generate constraints
        
        for indice in indices_sub:
            A = np.zeros(shape=(X.shape[0]))
            A[indice] = 1    
            constraint = LinearConstraint(A, lb=F_init[indice], ub=np.inf)    # bigger k
            distance_nn, distance_center, distance_nn_and_center = distances[indice]
            if distance_nn + distance_nn_and_center <= distance_center:
                if distance_nn < distance_nn_and_center:
                    constraint = LinearConstraint(A, lb=1, ub=F_init[indice])
            custom_constraints.append(constraint)

        raw_kernels = kldloss.KLProjectionConstrained(init='customized', F_init=F_init).fit_transform(X, custom_constraints=custom_constraints)

    
    # Directly employing overall distribution estimation as $F$.
    elif method.lower() == 'direct':
        raw_kernels = stats.compute_probabilities(X)
    
    else:
        raise ValueError('Invalid kernel method. ', method)

    return raw_kernels.reshape(-1, 1)
    


def linear_kernel(kappa, eta, F, sigma=1/3):
    """
    Computes the result of the linear kernel function: (1-eta)k + eta*F + epsilon, where
    epsilon is a normal distributed noise with mean 0 and standard deviation sigma.
    
    Parameters:
    -----------
    kappa : float or array-like
        A scalar or an array of shape (n_samples,) representing the input data.
    eta : float
        A scalar representing the weight of F in the kernel.
    F : float or array-like
        A scalar or an array of shape (n_samples,) representing the fitness kernel.
    sigma : float, default=1
        Standard deviation of the normal distributed noise added to the kernel result.
    
    Returns:
    --------
    kernel_result : float or array-like
        A scalar or an array of shape (n_samples,) representing the result of the linear kernel function.
    """
    epsilon = np.random.normal(0, sigma, size=1)[0]
    # return (1-eta)*kappa + eta*F + epsilon
    return (1-eta)*kappa + eta*F



def compute_distances(X, y, k=3, distance_measurement='euclidean'):
    """
    Compute three distances for each data point in the input matrix X:
    1. The distance between x_i and its nearest neighbor within the same class.
    2. The distance between x_i and the center of other neighbors within the same class (excluding the nearest neighbor).
    3. The distance between the center of other neighbors within the same class and the nearest neighbor within the same class.

    Parameters:
    -----------
    X : numpy array, shape (n_samples, n_features)
        The input matrix containing data points.
    y : numpy array, shape (n_samples,)
        The array of class labels for each data point in X.

    distance_measurement : str, optional (default='euclidean')
        The distance measurement to use. Supported options: 'euclidean', 'manhattan', 'cosine', etc.
    
    Returns:
    --------
    numpy array, shape (n_samples, 3)
        An array where each row contains the three distances for the corresponding data point in X.
    """
    distances = []

    for i in range(len(X)):
        x_i = X[i]
        class_label = y[i]

        # Filter data points with the same class label
        same_class_indices = np.where(y == class_label)[0]

        # Create a NearestNeighbors object for the same class data points
        nn = NearestNeighbors(n_neighbors=2, metric=distance_measurement)
        nn.fit(X[same_class_indices])

        # Find k+1 nearest neighbors for the current data point (including itself)
        _, indices = nn.kneighbors(x_i.reshape(1, -1))

        # Exclude the nearest neighbor (itself) from the list of neighbors
        nearest_neighbor_index = indices[0, 1]
        other_neighbor_indices = [_ for _ in range(same_class_indices.shape[0]) if _ not in indices[0, :2]]

        # Extract the data points corresponding to the nearest neighbor and other neighbors
        nearest_neighbor = X[same_class_indices[nearest_neighbor_index]]
        other_neighbors = X[same_class_indices[other_neighbor_indices]]
        center_of_other_neighbors = np.mean(other_neighbors, axis=0)

        # Compute the three distances
        distance_to_nearest_neighbor = np.linalg.norm(x_i - nearest_neighbor)
        distance_to_center_of_other_neighbors = np.linalg.norm(x_i - center_of_other_neighbors)
        distance_between_center_and_nearest = np.linalg.norm(center_of_other_neighbors - nearest_neighbor)

        distances.append([distance_to_nearest_neighbor, distance_to_center_of_other_neighbors, distance_between_center_and_nearest])

    return np.array(distances)


def get_border_samples(X, y, quarter=0.2, return_indices=False):
    """Extract a subset of X and their labels based on probabilities estimation.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data.
    y : array-like, shape (n_samples,)
        The labels for the input data.
    quarter : float, optional
        The fraction of samples to be extracted (e.g., 0.2 for 20%), by default 0.2.

    Returns
    -------
    X_subset : array-like, shape (n_subset_samples, n_features)
        The subset of X containing samples with lowest probabilities.
    y_subset : array-like, shape (n_subset_samples,)
        The labels corresponding to the samples in X_subset.
    lowest_prob_indices : Optional
    """
    # Compute probabilities for each sample in X
    probabilities = stats.compute_probabilities(X)

    # Get the number of samples to be extracted
    n_subset_samples = int(len(X) * quarter)

    # Sort probabilities and get the indices of the samples with the lowest probabilities
    lowest_prob_indices = np.argsort(probabilities)[:n_subset_samples]

    # Extract the subset of X and y based on the lowest probabilities
    X_subset = X[lowest_prob_indices]
    y_subset = y[lowest_prob_indices]
    if return_indices:
        return X_subset, y_subset, lowest_prob_indices
    else:
        return X_subset, y_subset


