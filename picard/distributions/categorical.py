import numpy as np 
import numpy.random as npr 

from picard.distributions.base import Distribution

# TODO: discrete versus continous distributions should extend a different abstract class 

class Categorical(Distribution): 

    def __init__(self, params : dict = None): 
        self.params = params
    
    def sample(self, size : int): 
        pass 

    def density(self, obs : np.ndarray): 
        pass

class Dirichlet(Distribution): 

    def __init__(self, params : dict = None): 
        self.params = params
    
    def sample(self, size : int = 1): 
        pass 

    def density(self, obs : np.ndarray): 
        pass