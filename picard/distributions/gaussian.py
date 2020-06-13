import math 
import numpy as np 
import numpy.random as npr 

from picard.distributions.base import Distribution

class UnivariateGaussian(Distribution): 

    def __init__(self, params : dict = None): 
        self.params = params
    
    def sample(self, size : int = 1): 
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

    def vb_update(self, hyperpriors : dict, obs : np.ndarray, variational_params : np.ndarray, n_k : np.ndarray): 
        pass

class MatrixNormalInverseWishart(Distribution): 

    def __init__(self, params : dict): 
        self.mean = params['mean']
        self.dof = params['dof']
        
        if 'covariance_scale' in list(params.keys()): 
            self.covariance_scale = params['covariance_scale']
            self.covariance = params['covariance']
        elif 'precision_scale' in list(params.keys()): 
            self.precision_scale = params['precision_scale']
            self.precision = params['precision']
        else: 
            raise KeyError('Matrix normal inverse Wishart must be instantiated with either a covariance scale or a precision scale')

    @property 
    def covariance(self): 
        return self.covariance_ 
    
    @property 
    def precision(self): 
        return self.precision_ 

    @property 
    def covariance_scale(self): 
        return self.covariance_scale_

    @property
    def precision_scale(self): 
        return self.precision_scale_ 

    @covariance.setter
    def covariance(self, covariance : np.ndarray): 
        self.covariance_ = covariance
        self.precision_ = np.linalg.inv(covariance)

    @precision.setter 
    def precision(self, precision : np.ndarray): 
        self.precision_ = precision
        self.covariance_ = np.linalg.inv(precision)

    @covariance_scale.setter 
    def covariance_scale(self, covariance_scale): 
        self.covariance_scale_ = covariance_scale
        self.precision_scale_ = 1/covariance_scale

    @precision_scale.setter 
    def precision_scale(self, precision_scale): 
        self.precision_scale_ = precision_scale
        self.covariance_scale_ = 1/precision_scale

    def sample(self, size : int = 1): 
        pass 

    def density(self, obs : np.ndarray): 
        pass

    def absorb(self, count : float): 
        self.precision_scale += count 
        self.dof += count + 1

class Wishart(Distribution): 

    def __init__(self, params : dict): 
        pass 

    def sample(self, size : int = 1): 
        pass 

    def density(self, obs : np.ndarray): 
        pass

