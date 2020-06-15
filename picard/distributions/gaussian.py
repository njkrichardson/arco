import math 
import numpy as np 
import numpy.random as npr 
import scipy
import scipy.stats as stats

from picard.distributions.base import Distribution

# TODO: should be more efficient way of maintaining the cov/precision isomorphism 

class UnivariateGaussian(Distribution): 
    dim = 1
    def __init__(self, params : dict = None): 
        self.mean = params['mean']

        if 'variance' in list(params.keys()): 
            self.variance = params['variance'] 
        elif 'precision' in list(params.keys()): 
            self.precision = params['precision']
        else: 
            raise KeyError('Univariate normal must be instantiated with either a variance or a precision')

    def __repr__(self): 
        return self.__class__.__name__ + f"(mean={self.mean}, variance={self.variance}"
    
    @property
    def variance(self): 
        return self.variance_

    @property
    def precision(self): 
        return self.precision_ 

    @variance.setter
    def variance(self, variance : float): 
        self.variance_ = variance
        self.precision_ = 1/variance
    
    @precision.setter 
    def precision(self, precision : float): 
        self.precision_ = precision
        self.variance_ = 1/precision

    def sample(self, size : int = 1): 
        return self.variance * npr.randn(size) + self.mean if size > 1 else self.variance * npr.randn() + self.mean
    
    def energy(self, obs : float): 
        return 0.5 * self.precision * (obs - self.mean)**2

    def density(self, obs : np.ndarray): 
        normalizer = math.sqrt((2 * math.pi * self.variance))
        return (1/normalizer) * math.exp(-self.energy(obs))

class MultivariateGaussian(Distribution): 

    def __init__(self, params : dict): 
        self.mean = params['mean']
        self.dim = self.mean.size

        if 'covariance' in list(params.keys()): 
            self.covariance = params['covariance']
        elif 'precision' in list(params.keys()): 
            self.precision = params['precision']
        else: 
            raise KeyError('Multivariate normal must be instantiated with either a covariance or precision')
    
    def __repr__(self): 
        return self.__class__.__name__ + f"(mean={self.mean}, covariance={self.covariance_}"

    @property
    def covariance(self): 
        return self.covariance_ 

    @property
    def precision(self): 
        return self.precision_ 

    @covariance.setter
    def covariance(self, covariance : np.ndarray): 
        self.covariance_ = covariance
        self.precision_ = np.linalg.inv(covariance)
    
    @precision.setter 
    def precision(self, precision : np.ndarray): 
        self.precision_ = precision
        self.covariance_ = np.linalg.inv(precision)

    def sample(self, size : int = 1): 
        return npr.multivariate_normal(self.mean, self.covariance, size=size)

    def density(self, obs : np.ndarray): 
        normalizer = 1 / np.sqrt(((2 * np.pi) ** self.dim) * np.linalg.det(self.covariance_)) 
        return normalizer * np.exp(-self.energy(obs))

    def energy(self, obs : np.ndarray): 
        return 0.5 * (obs - self.mean).T.dot(self.precision_.dot(obs - self.mean))

    def entropy(self): 
        return 0.5 * np.log(np.linalg.det(2 * np.pi * math.e * self.covariance_))

    def absorb(self, prior : Distribution, obs : np.ndarray): 
        """
        Absorb a prior and observations and update the distribution's parameters
        """
        self.mean = (prior.prior_precision_scale * prior.mean + obs) / prior.precision_scale

class NormalInverseWishart(Distribution): 

    def __init__(self, params : dict): 
        self.mean = params['mean']
        self.dof = params['dof']
        self.prior_measurements = params['prior_measurements']
        self.scale_matrix = params['scale_matrix']
        
    def __repr__(self): 
        return self.__class__.__name__ + f"(mean={self.mean}, dof={self.dof}, prior measurements={self.prior_measurements}, scale matrix={self.scale_matrix}"

    def sample(self, size : int = 1) -> tuple: 
        covariance = InverseWishart(dict(scale_matrix=self.scale_matrix, dof=self.dof)).sample() 
        mean = MultivariateGaussian(dict(mean=self.mean, covariance=covariance/self.prior_measurements)).sample() 
        return (mean, covariance)

    def density(self, obs : np.ndarray) -> float: 
        normal_pdf = stats.multivariate_normal.pdf(obs, mean=self.mean, cov=self.scale_matrix)  
        inv_wishart_pdf = stats.invwishart.pdf(obs, df=self.dof, scale=self.scale_matrix)
        return normal_pdf * inv_wishart_pdf

    def absorb(self, obs : np.ndarray, count : float): 
        # TODO: how is this going to work? 
        self.prior_measurements += count 
        self.dof += count + 1

class InverseWishart(Distribution): 
    """
    Inverse Wishart distribution, a distribution over real-valued positive-definite matrices.
    Used as the conjugate prior for the covariance matrix of a multivariate normal.  

    References: 
      * rb.gy/rrjtmo
    """

    def __init__(self, params : dict): 
        self.scale_matrix = params['scale_matrix'] 
        self.dim = self.scale_matrix.shape[0]
        self.dof = params['dof']

    def sample(self, size : int = 1) -> np.ndarray: 
        cholesky = np.linalg.cholesky(self.scale_matrix)
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(self.dof - np.arange(self.dim)))))
        x[np.triu_indices_from(x, 1)] = npr.randn(self.dim * (self.dim - 1) // 2)
        r = np.linalg.qr(x, 'r')
        t = scipy.linalg.solve_triangular(r.T, cholesky.T, lower=True).T
        return t.dot(t.T)

    def density(self, obs : np.ndarray) -> float: 
        return stats.invwishart.pdf(obs, df=self.dof, scale=self.scale_matrix)

class Wishart(Distribution): 
    """
    Distribution over symmetric, nonnegative-definite matrices. Used as the conjugate prior 
    of the precision matrix of a multivariate normal. 
    """

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

