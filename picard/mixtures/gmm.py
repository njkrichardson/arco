import numpy as np 
import numpy.random as npr 

from picard.models.base import Model
from picard.distributions.categorical import Categorical, Dirichlet 
from picard.distributions.gaussian import MatrixNormalInverseWishart, MultivariateGaussian

# TODO: Solve the century old inference problem, for now we do bespoke like everyone else 

class GaussianMixtureModel(Model): 

    def __init__(self, n_components : int, hyperpriors : dict): 
        self.n_components = n_components
        self.hyperpriors_ = hyperpriors
        self.prior = {
            'hyperpriors': self.hyperpriors_, 
            'mixture_prior': Dirichlet(hyperpriors), 
            'component_priors': [MatrixNormalInverseWishart(hyperpriors) for _ in range(self.n_components)]
            }
        self.likelihood = {
            'mixture_weights': Categorical(self.prior['mixture_prior'].sample()), 
            'components': [MultivariateGaussian(self.prior['component_priors'].sample()) for _ in range(self.n_components)]
        }
        pass 

    @property
    def hyperpriors(self): 
        return self.hyperpriors_
    
    @hyperpriors.setter
    def hyperpriors(self, hyperpriors : dict): 
        # TODO: things geting sketchy here, need to think about how this should work
        self.hyperpriors_ = hyperpriors
        self.prior['hyperpriors'] = self.hyperpriors_
        self.prior['mixture_prior'] = Dirichlet(self.hyperpriors_)
        self.prior['component_priors'] = [MatrixNormalInverseWishart(hyperpriors) for _ in range(self.n_components)]

    def show(self): 
        pass 

    def sample(self): 
        pass 

    def fit(self, obs : np.ndarray, method : str = 'mean_field_vb'):
        q_z = np.array([Categorical(self.prior['mixture_prior'].sample()) for _ in range(self.n_components)])

        if method is 'mean_field_vb': 
            # M step (TODO: remember to normalize by n_k)
            # TODO: use multiple processes 
            n_k = q_z.sum(axis=0)
            for k in range(self.n_components): 
                self.prior['component_priors'][k].absorb(n_k[k])
                self.likelihood['components'][k].vb_update(self.prior['component_priors'][k], obs, q_z) 

        else: 
            raise NotImplementedError

    def predict(self): 
        pass

