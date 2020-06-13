import numpy as np 
from abc import ABC, abstractmethod

class Model(ABC): 
    """
    Model base class. 
    """

    @abstractmethod
    def __init__(self): 
        pass

    @abstractmethod
    def show(self): 
        pass

    @abstractmethod 
    def fit(self): 
        pass 

    @abstractmethod
    def predict(self): 
        pass