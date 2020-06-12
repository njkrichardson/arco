import numpy as np 

from picard.distributions.base import Distribution

class UnivariateGaussian(Distribution): 

    def __init__(self, params : dict = None): 
        self.params = params
    
    def sample(self, size :int): 
        pass 

    def density(self, obs : np.ndarray): 
        pass

if __name__ == "__main__":
    gaussian = UnivariateGaussian() 

