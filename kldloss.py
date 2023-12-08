import numpy as np
import math
from sklearn.neighbors import KernelDensity
from prda import prep
from scipy.optimize import minimize, LinearConstraint

class KLProjection:
    def __init__(
        self,
        num_dimensions=1,
        h=0.5,
        learning_rate=0.005,
        max_iterations=500,
        loss_threshold=1e-2,
        init='customized',
        use_sgd=False,
        batch_size=0.2,
        use_adam=True,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        F_init: np.ndarray = None):

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.loss_threshold = loss_threshold
        self.num_dimensions = num_dimensions
        self.F = F_init
        self.init = init
        self.h = h

        # Param for sgd
        self.use_sgd = use_sgd
        self.batch_size = batch_size

        # Params for Adam
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def _compute_probabilities(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Create a Kernel Density Estimator
        kde = KernelDensity(kernel='gaussian', bandwidth=self.h)

        # Fit the KDE model to the data
        kde.fit(X)

        # Evaluate the KDE model on the data points
        log_probabilities = kde.score_samples(X)

        # Convert log probabilities to probabilities
        probabilities = np.exp(log_probabilities)
        normalized_probabilities = probabilities / np.sum(probabilities)

        return normalized_probabilities

    

    def _initialize_F(self, X, n_samples):
        if self.init.lower() == 'random':
            self.F = np.abs(np.random.randn(n_samples, self.num_dimensions))
        elif self.init.lower() == 'pca':
            self.F = prep.pca(data=X, npcs=self.num_dimensions)
        elif self.init.lower() == 'linear':
            self.F = prep.apply_linear_func(X)
        elif self.init.lower() == 'auto':
            lowest_loss = np.inf
            best_init = None
            for init_method in ['pca', 'linear', 'random']:
                F_init = None
                if init_method == 'pca':
                    F_init = prep.pca(data=X, npcs=self.num_dimensions)
                elif init_method == 'linear':
                    F_init = prep.apply_linear_func(X)
                elif init_method == 'random':
                    F_init = np.abs(np.random.randn(n_samples, self.num_dimensions)) 
                F_sample_probs = self._compute_probabilities(F_init)
                loss = compute_kl_divergence(self.X_sample_probs, F_sample_probs)
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_init = F_init
            self.F = best_init
        elif self.init.lower() == 'log':    #  F_{\text{init}} = \log \left( X \cdot \mathbf{1}_{d,1} \right) + \delta

            # Not support for `num_dimensions` != 1.
            row_sums = np.sum(X, axis=1)
            noise = np.random.normal(0, sigma=0.05, size=row_sums.shape)
            self.F = np.log(row_sums + noise)

        elif self.init.lower() == 'customized':
            self.F = initialize_fitness_kernel(self.F)
        else:
            raise KeyError(self.init, 'is not a valid method for F initialization.')
        
        self.F = self.F.reshape(n_samples, self.num_dimensions)
    
    def fit_transform(self, X):
        losses = []
        n_samples, n_features = X.shape
        X_sample_probs = self._compute_probabilities(X)

        # Initialize projection matrix F
        self._initialize_F(X, n_samples)


        for _itr in range(self.max_iterations):
            F_sample_probs = self._compute_probabilities(self.F)
            loss = compute_kl_divergence(X_sample_probs, F_sample_probs)
            losses.append(loss)

            # print(f"Iteration {iteration}: Loss = {loss}")
            if loss < self.loss_threshold:
                break

            # Descent
            if self.use_sgd == 'sgd':
                indices = np.random.choice(n_samples, math.ceil(self.batch_size*n_samples), replace=False)
            else:    # Newton iteration
                indices = [_ for _ in range(n_samples)]    
            for m in indices:

                # Calculate gradient
                diff = self.F[m] - self.F
                sum1 = np.sum(X_sample_probs * (diff))
                sum2_upper = np.sum(diff * np.exp(-0.5 * (diff ** 2) / (self.h ** 2)))
                sum2_lower = np.sum(np.exp(-0.5 * (diff ** 2) / (self.h ** 2)))
                gradient = (1 / (self.h ** 2)) * (sum1 + X_sample_probs[m] * (sum2_upper / sum2_lower))
                
                # Adam update
                if self.use_adam:
                    self.t += 1
                    self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
                    self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
                    m_hat = self.m / (1 - self.beta1 ** self.t)
                    v_hat = self.v / (1 - self.beta2 ** self.t)
                    self.F[m] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                else:
                    self.F[m] -= self.learning_rate * gradient
        
        self.__draw_loss(losses)
                

        return self.F
    
    def __draw_loss(self, losses):
        import seaborn as sns
        sns.lineplot(data=losses)




class KLProjectionConstrained(KLProjection):
    

    def fit_transform(self, X, custom_constraints=None):
        n_samples, n_features = X.shape
        self.X_sample_probs = self._compute_probabilities(X)

        # Initialize projection matrix F
        self._initialize_F(X, n_samples=X.shape[0])

        # Define the optimization function to minimize
        def optimization_function(F_flat):
            F = F_flat.reshape(n_samples, self.num_dimensions)
            F_sample_probs = self._compute_probabilities(F)
            return compute_kl_divergence(self.X_sample_probs, F_sample_probs)
        

        # Perform optimization using SciPy
        initial_guess = self.F.flatten()
        result = minimize(optimization_function, initial_guess, constraints=custom_constraints,
                          options={'maxiter': self.max_iterations, 'ftol': self.loss_threshold})
        self.F = result.x.reshape(n_samples, self.num_dimensions)

        return self.F
    

    

    
def compute_kl_divergence(X_sample_probs, F_sample_probs):
    """Calculate sample-wise KLD.
    """
    kld = np.sum(X_sample_probs * np.log(X_sample_probs / F_sample_probs))
    return kld

def initialize_fitness_kernel(F_init):
    if type(F_init) != np.ndarray:
        if not F_init:
            raise AttributeError('When setting `init` to "customized", `F_init` must be assigned.')
    return F_init
