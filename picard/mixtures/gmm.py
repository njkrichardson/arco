import numpy as np 
import numpy.random as npr 

from picard.models.base import Model
from picard.distributions.categorical import Categorical, Dirichlet 
from picard.distributions.gaussian import MatrixNormalInverseWishart, MultivariateGaussian

# TODO: Solve the century old inference problem, for now we do bespoke like everyone else 
# TODO: set a model obs dim, don't infer it during fit 

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
        n_obs, obs_dim = obs.shape[1]
        q_z = np.array([Categorical(self.prior['mixture_prior'].sample()) for _ in range(self.n_components)])

        if method is 'mean_field_vb': 
            # M step (TODO: remember to normalize by n_k)
            # TODO: use multiple processes 

            # know this works, vectorize in a bit 
            weighted_means = np.zeros((self.n_components, obs_dim))
            for i in range(n_obs): 
                for k in range(self.n_components): 
                    weighted_means[k] += q_z[i][k] * obs[i]
            total_responsibilities = q_z.sum(axis=0)
            scatter_matrices = [np.eye(obs_dim)] 
            for k in range(self.n_components): 
                self.prior['component_priors'][k].absorb(total_responsibilities[k])
                # self.likelihood['components'][k].absorb(self.prior['component_priors'][k], obs, weights=q_z[:, k]) # maybe a sort of generic updates where the q_zs are just weights? 
                self.likelihood['components'][k].absorb(self.prior['component_priors'][k], total_responsibilities[k] * (obs * q_z[:, k]))

        else: 
            raise NotImplementedError

    def predict(self): 
        pass

if __name__ == "__main__":
    # model parameters 
    n_components = 3 
    obs_dim = 2 

    hyperpriors = dict(
        mean=npr.randn(obs_dim),
        scale=np.eye(obs_dim), 
        prior_measurements=3, 
        dof=obs_dim + 1
        )
