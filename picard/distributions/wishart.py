import numpy as np 
import numpy.random as npr 
import scipy 
import scipy.stats as stats 

from picard.distributions.base import Distribution

# TODO: abstraction for exponential families and the notion of sufficiency for a distribution's absorb method. I.e. what 
# does the distribution need to _know_ from the data and a likeihood model in order to produce a posterior distribution? 
# Maybe the right thing to do is actually have priors absorb data and likelihoods and produce their parameter updates that way? 
# I guess regardless of who "encapsulates" who we are still going to have some interesting depth going on. Just need to think about 
# what's better from a design pov. 

class InverseWishart(Distribution): 
    """
    Inverse Wishart distribution, a distribution over real-valued positive-definite matrices.
    Used as the conjugate prior for the covariance matrix of a multivariate normal.  

    References: 
      * rb.gy/rrjtmo
    """

    def __init__(self, params : dict): 
        self.prior_scale_matrix = params['scale_matrix'] 
        self.posterior_scale_matrix = None 
        self.dim = self.prior_scale_matrix.shape[0]
        self.prior_dof = params['dof']
        self.posterior_dof = None 

    @property 
    def scale_matrix(self): 
        return self.posterior_scale_matrix if self.posterior_scale_matrix is not None else self.prior_scale_matrix

    @property 
    def dof(self): 
        return self.posterior_dof if self.posterior_dof is not None else self.prior_dof

    def sample(self, size : int = 1) -> np.ndarray: 
        cholesky = np.linalg.cholesky(self.scale_matrix) 
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(self.dof - np.arange(self.dim)))))
        x[np.triu_indices_from(x, 1)] = npr.randn(self.dim * (self.dim - 1) // 2)
        r = np.linalg.qr(x, 'r')
        t = scipy.linalg.solve_triangular(r.T, cholesky.T, lower=True).T
        return t.dot(t.T)

    def density(self, obs : np.ndarray) -> float: 
        return stats.invwishart.pdf(obs, df=self.dof, scale=self.scale_matrix)

    def absorb(self, obs : np.ndarray): 
        """
        Condition on observations (summary statistics, in this case) to produce posterior parameter estimates. 
        """


class Wishart(Distribution): 
    """
    Distribution over symmetric, nonnegative-definite matrices. Used as the conjugate prior 
    for the precision matrix of a multivariate normal. 
    """

    # TODO: update this to have the prior/posterior property model 

    def __init__(self, params : dict): 
        self.scale_matrix = params['scale_matrix']
        self.dim = self.scale_matrix.shape[0]
        self.dof = params['dof']

    def sample(self, size : int = 1) -> np.ndarray: 
        cholesky = np.linalg.cholesky(self.scale_matrix)
        a = np.diag(np.sqrt(npr.chisquare(self.dof - np.arange(self.dim))))
        a[np.tri(self.dim, dtype=bool)] = npr.normal(size=(self.dim * (self.dim - 1) / 2.))
        x = np.dot(cholesky, a)
        return x.dot(x.T)

    def density(self, obs : np.ndarray) -> float: 
        return stats.wishart.pdf(obs, df=self.dof, scale=self.scale_matrix)