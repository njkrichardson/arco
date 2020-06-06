import numpy as np 
from abc import ABC, abstractmethod

class Distribution(ABC): 
    """
    Distribution base class. 
    """

    @abstractmethod
    def __init__(self): 
        pass

    @abstractmethod 
    def sample(self, size : int): 
        pass 

    @abstractmethod
    def density(self, obs : np.ndarray): 
        pass 
