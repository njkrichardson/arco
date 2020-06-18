import numpy as np 
import numpy.random as npr 
from abc import ABC, abstractmethod

class Distribution(ABC): 
    """
    Distribution base class. 
    """

    @abstractmethod
    def __init__(self): 
        pass

    @abstractmethod 
    def sample(self, size : int = 1): 
        pass 

    @abstractmethod
    def density(self, obs : np.ndarray): 
        pass 

class MixtureDistribution(Distribution): 
    """
    Mixture distribution base class. 
    """

    def __init__(self, params : dict): 
        self.mixture_components = params['mixture_components']
        self.mixture_weights = params['mixture_weights']
        self.n_components = len(self.mixture_components)

    def sample(self, size : int = 2): 
        assert size > 1 

        component_labels = [npr.choice(np.arange(self.n_components), size=size, p=self.mixture_weights)]
        observations = [self.mixture_components[component_labels[0]].sample()]

        for _ in range(size - 1): 
            component_labels.append(npr.choice(np.arange(self.n_components), size=size, p=self.mixture_weights))
            observations.append(self.mixture_components[component_labels[-1]].sample())
        
        return np.array(observations), np.array(component_labels)
    
    def density(self): 
        raise NotImplementedError

