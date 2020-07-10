import pickle 

import matplotlib.pyplot as plt 
import numpy as np 
import numpy.random as npr 
from scipy.special import digamma

from picard.models.base import Model
from picard.distributions.base import MixtureDistribution
from picard.distributions.categorical import Categorical, Dirichlet 
from picard.distributions.gaussian import NormalInverseWishart, MultivariateGaussian
from picard.utils.linear_algebra import safe_divide, mahalanobis
from picard.utils.io import load_object

# TODO: Solve the century old inference problem, for now we do bespoke like everyone else 
# TODO: set a model obs dim, don't infer it during fit 

# TODO: convert to mixture model (from base)
# TODO: initialize mvn from hyperprior if mu and sigma are not provided 

# TODO: on initialization: if they give means and covariances in the prior just initialize with those, maybe unless otherwise stated we assume we're 
# initializing by sampling from the prior? 

class GaussianMixtureModel(Model): 

    def __init__(self, n_components : int, hyperpriors : dict): 
        self.prior = dict() 
        self.n_components, self.hyperpriors = n_components, hyperpriors
        self.initialize_components() # right now, we can only initialize from the prior

    def __repr__(self): 
        return self.__class__.__name__ + f"(n_components={self.n_components})"

    @property
    def hyperpriors(self): 
        return self.hyperpriors_
    
    @hyperpriors.setter
    def hyperpriors(self, hyperpriors : dict): 
        # TODO: things getting sketchy here, need to think about how this should work
        self.hyperpriors_ = hyperpriors
        self.prior['hyperpriors'] = self.hyperpriors_
        self.prior['mixture_prior'] = Dirichlet(self.hyperpriors)
        self.prior['component_priors'] = [NormalInverseWishart(self.hyperpriors) for _ in range(self.n_components)]

    def show(self): 
        pass 

    def sample(self, size : int):
        return self.likelihood.sample(size=size) 

    def initialize_components(self, method : str = 'from_prior', **kwargs): 
        if method == 'from_prior': 
            self.likelihood = MixtureDistribution(dict(
                mixture_weights=self.prior['mixture_prior'].sample(), 
                mixture_components=[self.prior['component_priors'][k].sample(return_dist=True) for k in range(self.n_components)]
            ))
        elif method == 'on_datum': 
            raise NotImplementedError
        elif method == 'kmeans++': 
            raise NotImplementedError
        elif method == 'gibbs': 
            raise NotImplementedError
        else: 
            raise NotImplementedError

    def fit(self, obs : np.ndarray, method : str = 'mean_field_vb'):
        # initialize variational params (TODO: alternative initializations?)
        n_obs, obs_dim = obs.shape
        self.obs_dim = obs_dim
        q_z = np.array([Dirichlet(dict(concentrations=self.likelihood.mixture_weights)).sample() for _ in range(n_obs)])

        if method is 'mean_field_vb': 
            # M step 
            # TODO: use multiple processes 
            sufficient_stats = self.collect_sufficient_stats(obs, q_z)
            for k in range(self.n_components): 
                self.prior['component_priors'][k].absorb(self.likelihood, sufficient_stats[k])
            
            # E step 
            expected_energy = self._compute_expected_energy(obs)
            expected_log_det_precisions = self._compute_log_det_precision() 
            expected_log_pi = self._compute_expected_log_pi(sufficient_stats['n_measurements'])
            q_z = self._update_variational_params(expected_log_pi, expected_log_det_precisions, expected_energy, n_obs)
        
        else: 
            raise NotImplementedError

    def _compute_expected_energy(self, obs): 
        n_obs, obs_dim = obs.shape
        expected_energy = np.zeros((n_obs, self.n_components))
        for i in range(n_obs): 
            for k in range(self.n_components): 
                expected_energy[i][k] = obs_dim * (1/self.prior[k].n_measurements) + self.prior[k].dof * mahalanobis(obs[i], self.components[k].mean, self.components[k].covariance)

        return expected_energy

    def _compute_log_det_precision(self): 
        log_det_precisions = [None for _ in range(self.n_components)]
        for k in range(self.n_components):
            det_precision = np.linalg.det(self.prior['component_priors'][k].scale_matrix) # TODO: should this be inverse scale matrix? 
            if det_precision > 1e-30: 
                log_det = np.log(det_precision)
            else: 
                log_det = 0.0
            log_det_precisions[k] = np.sum([digamma((self.prior['component_priors'][k].dof + 1 - i) / 2.) for i in range(self.obs_dim)]) + self.obs_dim * np.log(2) + log_det
        return log_det_precisions

    def _compute_expected_log_pi(self, n_k : np.ndarray): 
        self.likelihood['mixture_weights'].concentrations += n_k 
        expected_log_pi = digamma(self.likelihood['mixture_weights'].concentrations) - digamma(self.likelihood['mixture_weights'].concentrations.sum())
        return expected_log_pi

    def _update_variational_params(self, expected_log_pi : np.ndarray, log_det_precisions : list, energy : np.ndarray, n_obs : int) -> np.ndarray:
        ln_q_z = np.zeros((n_obs, self.n_components))  # ln q_z 
        for k in range(self.n_components):
            ln_q_z[:, k] = expected_log_pi[k] + 0.5 * log_det_precisions[k] - 0.5 * self.obs_dim * np.log(2 * np.pi) - 0.5 * energy[:, k]

        #normalise ln Z:
        ln_q_z -= ln_q_z.max(axis=1).reshape(n_obs, 1)
        q_z = np.exp(ln_q_z) / np.exp(ln_q_z).sum(axis=1).reshape(n_obs, 1)
        return q_z

    def collect_sufficient_stats(self, obs : np.ndarray, assignments : np.ndarray): 
        n_obs, obs_dim = obs.shape
        total_responsibilities = assignments.sum(axis=0).reshape(-1, 1)
        normalizer = safe_divide(1, total_responsibilities) 
        means = normalizer * obs.T.dot(assignments).T
        s = np.tile(obs, (self.n_components, 1, 1)) - np.array([np.tile(mean, (n_obs, 1)) for mean in means])
        scatter_matrices = s.reshape(self.n_components, obs_dim, n_obs) @ s
        sufficient_stats = [
            dict(
            mean=means[k], 
            scatter_matrix=scatter_matrices[k], 
            n_measurements=total_responsibilities[k]) for k in range(self.n_components)
            ]
        return sufficient_stats

    def predict(self): 
        pass

if __name__ == "__main__":
    experiment_params = load_object('/Users/nickrichardson/Desktop/personal/projects/picard/tests/comparators/gmm.pkl') 

    x_t = experiment_params['observations']
    z_t = experiment_params['latent_variables']
    k = experiment_params['n_components']
    hyperpriors = experiment_params['hyperpriors']

    gmm = GaussianMixtureModel(n_components=k, hyperpriors=hyperpriors)
    gmm.fit(x_t)

