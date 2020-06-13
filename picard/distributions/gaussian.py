import math 
import numpy as np 
import numpy.random as npr 

from picard.distributions.base import Distribution

class UnivariateGaussian(Distribution): 

    def __init__(self, params : dict = None): 
        self.params = params
    
    def sample(self, size :int): 
        pass 

    def density(self, obs : np.ndarray): 
        pass

class MultivariateGaussian(Distribution): 

    def __init__(self, params : dict): 
        self.mean = params['mean']
        self.dim = self.mean.size

        if 'covariance' in list(params.keys()): 
            self.covariance = params['covariance']
        elif 'precision' in list(params.keys()): 
            self.precision = params['precision']
        else: 
            raise KeyError('Multivariate normal must be initialized with either a covariance or precision')
    
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

    def sample(self, size : int): 
        return npr.multivariate_normal(self.mean, self.covariance, size=size)

    def density(self, obs : np.ndarray): 
        normalizer = 1 / np.sqrt(((2 * np.pi) ** self.dim) * np.linalg.det(self.covariance_)) 
        return normalizer * np.exp(-self.energy(obs))

    def energy(self, obs : np.ndarray): 
        return 0.5 * (obs - self.mean).T.dot(self.precision_.dot(obs - self.mean))

    def entropy(self): 
        return 0.5 * np.log(np.linalg.det(2 * np.pi * math.e * self.covariance_))

if __name__ == "__main__":
    gaussian = UnivariateGaussian() 

