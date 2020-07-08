import matplotlib.pyplot as plt 
import numpy as np 
import numpy.random as npr 
from scipy.special import digamma

from picard.models.base import Model
from picard.distributions.base import MixtureDistribution
from picard.distributions.categorical import Categorical, Dirichlet 
from picard.distributions.gaussian import NormalInverseWishart, MultivariateGaussian
from picard.utils.linear_algebra import safe_divide, mahalanobis

# TODO: Solve the century old inference problem, for now we do bespoke like everyone else 
# TODO: set a model obs dim, don't infer it during fit 

# TODO: convert to mixture model (from base)
# TODO: initialize mvn from hyperprior if mu and sigma are not provided 

# TODO: on initialization: if they give means and covariances in the prior just initialize with those, maybe unless otherwise stated we assume we're 
# initializing by sampling from the prior? 

class GaussianMixtureModel(Model): 

    def __init__(self, n_components : int, hyperpriors : dict): 
        self.n_components, self.hyperpriors_ = n_components, hyperpriors
        self.prior = dict(
            mixture_prior=Dirichlet(hyperpriors), 
            component_priors=[NormalInverseWishart(hyperpriors) for _ in range(self.n_components)]
        )
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
        self.prior['mixture_prior'] = Dirichlet(self.hyperpriors_)
        self.prior['component_priors'] = [NormalInverseWishart(hyperpriors) for _ in range(self.n_components)]

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
        q_z = np.array([self.prior['mixture_prior'].sample() for _ in range(n_obs)])

        if method is 'mean_field_vb': 
            # M step 
            # TODO: use multiple processes 
            sufficient_stats = self.collect_sufficient_stats(obs, q_z)
            for k in range(self.n_components): 
                self.prior['component_priors'][k].absorb(self.likelihood.mixture_components[k], sufficient_stats[k])
            
            # E step 
            expected_energy = self._compute_expected_energy(obs)
            expected_log_det_precision = self._compute_log_det_precision() 
        
        else: 
            raise NotImplementedError

    def _compute_expected_energy(self, obs): 
        n_obs, obs_dim = obs.shape
        expected_energy = np.zeros((n_obs, self.n_components))
        for i in range(n_obs): 
            for k in range(self.n_components): 
                expected_energy[i][k] = obs_dim * (1/self.prior.n_measurements) + self.prior.dof * mahalanobis(obs[i], self.components[k].mean, self.components[k].covariance)

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
        self.prior['mixture_prior'] += n_k 
        expected_log_pi = digamma(self.likelihood['mixture_weights']) - digamma(self.likelihood['mixture_weights'].concentrations.sum())
        return expected_log_pi

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
    # model parameters 
    n_components = 3 
    obs_dim = 2 

    hyperpriors = dict(
        concentrations=np.ones(n_components),
        mean=npr.randn(obs_dim),
        scale_matrix=np.eye(obs_dim), 
        prior_measurements=5, 
        dof=obs_dim + 1
        )

    gmm = GaussianMixtureModel(n_components=n_components, hyperpriors=hyperpriors)
    x, z = gmm.sample(1000)

    gmm.fit(x)


    

    
