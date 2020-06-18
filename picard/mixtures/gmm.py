import matplotlib.pyplot as plt 
import numpy as np 
import numpy.random as npr 

from picard.models.base import Model
from picard.distributions.categorical import Categorical, Dirichlet 
from picard.distributions.gaussian import NormalInverseWishart, MultivariateGaussian
from picard.utils.linear_algebra import safe_divide, tensor_outer_product

# TODO: Solve the century old inference problem, for now we do bespoke like everyone else 
# TODO: set a model obs dim, don't infer it during fit 

class GaussianMixtureModel(Model): 

    def __init__(self, n_components : int, hyperpriors : dict): 
        self.n_components = n_components
        self.hyperpriors_ = hyperpriors
        self.prior = {
            'hyperpriors': self.hyperpriors_, 
            'mixture_prior': Dirichlet(hyperpriors), 
            'component_priors': [NormalInverseWishart(hyperpriors) for _ in range(self.n_components)]
            }
        init_likelihoods = [self.prior['component_priors'][k].sample() for k in range(self.n_components)]
        init_params = [dict(mean=likelihood[0], covariance=likelihood[1]) for likelihood in init_likelihoods]
        self.likelihood = {
            'mixture_weights': Categorical(dict(pis=self.prior['mixture_prior'].sample())), 
            'components': [MultivariateGaussian(init_params[k]) for k in range(self.n_components)]
        }

    def __repr__(self): 
        return self.__class__.__name__ + f"(n_components={self.n_components})"

    @property
    def hyperpriors(self): 
        return self.hyperpriors_
    
    @hyperpriors.setter
    def hyperpriors(self, hyperpriors : dict): 
        # TODO: things geting sketchy here, need to think about how this should work
        self.hyperpriors_ = hyperpriors
        self.prior['hyperpriors'] = self.hyperpriors_
        self.prior['mixture_prior'] = Dirichlet(self.hyperpriors_)
        self.prior['component_priors'] = [NormalInverseWishart(hyperpriors) for _ in range(self.n_components)]

    def show(self): 
        pass 

    def sample(self, n_samples : int):
        assert n_samples > 1

        z = [self.likelihood['mixture_weights'].sample()] 
        x = [self.likelihood['components'][z[0]].sample()]

        for _ in range(n_samples - 1): 
            z.append(self.likelihood['mixture_weights'].sample())
            x.append(self.likelihood['components'][z[-1]].sample())
        
        return np.array(x), np.array(z)

    def fit(self, obs : np.ndarray, method : str = 'mean_field_vb'):
        n_obs, obs_dim = obs.shape
        q_z = np.array([self.prior['mixture_prior'].sample() for _ in range(n_obs)])

        if method is 'mean_field_vb': 
            # M step (TODO: remember to normalize by n_k)
            # TODO: use multiple processes 

            total_responsibilities = q_z.sum(axis=0).reshape(-1, 1)
            normalizer = safe_divide(1, total_responsibilities) 
            means = normalizer * obs.T.dot(q_z).T
            # TODO: tensordot 
            scatter_matrices = [] 
            s = np.tile(obs, (self.n_components, 1, 1)) - np.array([np.tile(mean, (n_obs, 1)) for mean in means])
            for scatter in s: 
                scatter_matrices.append(scatter.T.dot(scatter))
            sufficient_stats = [dict(
                mean=means[k], 
                scatter_matrix=scatter_matrices[k], 
                n_measurements=total_responsibilities[k]) for k in range(self.n_components)]
            for k in range(self.n_components): 
                self.prior['component_priors'][k].absorb(self.likelihood['components'][k], sufficient_stats[k])

        else: 
            raise NotImplementedError

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


    

    
