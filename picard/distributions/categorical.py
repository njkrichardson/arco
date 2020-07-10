import numpy as np 
import numpy.random as npr 

from picard.distributions.base import Distribution

class Categorical(Distribution): 

    def __init__(self, params : dict = None): 
        self.pis = params['pis']

    def __repr__(self): 
        return self.__class__.__name__ + f"(distribution={np.round(self.pis, 2)})"
    
    def sample(self, size : int = 1): 
        return npr.choice(np.arange(self.pis.size), size=size, p=self.pis) if size != 1 else npr.choice(np.arange(self.pis.size), size=size, p=self.pis)[0]

    def density(self, obs : np.ndarray): 
        raise NotImplementedError

class Dirichlet(Distribution): 

    def __init__(self, params : dict = None): 
        self.concentrations = params['concentrations']

    def __repr__(self): 
        return self.__class__.__name__ + f"(concentrations={self.concentrations})"
    
    def sample(self, size : int = 1): 
        # TODO: 
        return npr.dirichlet(self.concentrations, size=size)[0] if size == 1 else npr.dirichlet(self.concentrations, size=size) 

    def density(self, obs : np.ndarray): 
        raise NotImplementedError