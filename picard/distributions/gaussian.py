import math 
import numpy as np 
import numpy.random as npr 
import scipy
import scipy.stats as stats

from picard.distributions.base import Distribution
from picard.distributions.wishart import InverseWishart

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
        return self.__class__.__name__ + f"(mean={self.mean}, variance={self.variance})"
    
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
        # TODO: initialize from prior if prior params are given instead of mean and covariance 
        self.mean = params['mean']
        self.dim = self.mean.size

        if 'covariance' in list(params.keys()): 
            self.covariance = params['covariance']
        elif 'precision' in list(params.keys()): 
            self.precision = params['precision']
        else: 
            raise KeyError('Multivariate normal must be instantiated with either a covariance or precision')
    
    def __repr__(self): 
        return self.__class__.__name__ + f"(mean={self.mean}, covariance={self.covariance_})"

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
        # TODO: keep this DRY! 
        return npr.multivariate_normal(self.mean, self.covariance, size=size) if size != 1 else npr.multivariate_normal(self.mean, self.covariance, size=size)[0]

    def density(self, obs : np.ndarray): 
        normalizer = 1 / np.sqrt(((2 * np.pi) ** self.dim) * np.linalg.det(self.covariance_)) 
        return normalizer * np.exp(-self.energy(obs))

    def energy(self, obs : np.ndarray): 
        # TODO: 0.5 * mahalanobis(obs, self.mean, self.precision)
        return 0.5 * (obs - self.mean).T.dot(self.precision.dot(obs - self.mean))

    def entropy(self): 
        return 0.5 * np.log(np.linalg.det(2 * np.pi * math.e * self.covariance_))

    def absorb(self, prior : Distribution, obs : np.ndarray): 
        """
        Absorb a prior and observations and update the distribution's parameters
        """
        self.mean = (prior.prior_precision_scale * prior.mean + obs) / prior.precision_scale

class NormalInverseWishart(Distribution): 

    def __init__(self, params : dict): 
        self.prior_mean = params['mean']
        self.posterior_mean = None 
        self.prior_dof = params['dof']
        self.posterior_dof = None
        self.prior_measurements = params['prior_measurements']
        self.posterior_measurements = None
        self.prior_scale_matrix = params['scale_matrix']
        self.posterior_scale_matrix = None

    def __repr__(self): 
        return self.__class__.__name__ + f"(mean={self.mean}, dof={self.dof}, prior measurements={self.prior_measurements}, scale matrix={self.scale_matrix})"

    @property 
    def mean(self): 
        return self.posterior_mean if self.posterior_mean is not None else self.prior_mean

    @property 
    def dof(self): 
        return self.posterior_dof if self.posterior_dof is not None else self.prior_dof

    @property 
    def measurements(self): 
        return self.posterior_measurements if self.posterior_measurements is not None else self.prior_measurements

    @property 
    def scale_matrix(self): 
        return self.posterior_scale_matrix if self.posterior_scale_matrix is not None else self.prior_scale_matrix

    def sample(self, size : int = 1, return_dist : bool = False) -> tuple: 
        covariance = InverseWishart(dict(scale_matrix=self.scale_matrix, dof=self.dof)).sample() 
        mean = MultivariateGaussian(dict(mean=self.mean, covariance=covariance/self.measurements)).sample() 
        return (mean, covariance) if return_dist is False else MultivariateGaussian(dict(mean=mean, covariance=covariance))

    def density(self, obs : np.ndarray) -> float: 
        normal_pdf = stats.multivariate_normal.pdf(obs, mean=self.mean, cov=self.scale_matrix)  
        inv_wishart_pdf = stats.invwishart.pdf(obs, df=self.dof, scale=self.scale_matrix)
        return normal_pdf * inv_wishart_pdf

    def absorb(self, likelihood : Distribution, sufficient_stats : dict): 
        # TODO: Dangerous, be careful about using the properties like this, remember to test this later 
        # TODO: lmao at this ordering, definitely will need to be tested 
        # TODO: why does 10.62 in Bishop contradict p.73 of Gelman? 
        assert all(stat in list(sufficient_stats.keys()) for stat in ['scatter_matrix', 'n_measurements', 'mean']), 'insufficient stats provided'
        self.posterior_scale_matrix =  self.scale_matrix + sufficient_stats['scatter_matrix'] + \
                                        ((self.measurements * sufficient_stats['n_measurements']) / (self.measurements + sufficient_stats['n_measurements'])) * \
                                        np.outer((sufficient_stats['mean'] - self.mean), (sufficient_stats['mean'] - self.mean))
        self.posterior_mean = (self.measurements * self.mean + sufficient_stats['n_measurements'] * sufficient_stats['mean']) / (self.measurements + sufficient_stats['n_measurements'])
        self.posterior_dof = self.dof + sufficient_stats['n_measurements']
        self.posterior_measurements = self.measurements + sufficient_stats['n_measurements']

        # likelihood.absorb(obs, sufficient_stats)
        # TODO: how is this going to work? I think actually we should delegate the updates to the constituent distributions
