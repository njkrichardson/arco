import os
import pickle 

def save_object(py_object, path : str):
    """
    Save a generic Python object. 

    Reference: rb.gy/hi1vqw

    Parameters
    ----------
    py_object : [type]
        any Python object
    path : str
        path to save the object to 
    """
    with open(path, 'wb') as f:
        pickle.dump(py_object, f, pickle.HIGHEST_PROTOCOL)

def load_object(path : str): 
    """
    Load a generic Python object. 

    Reference: rb.gy/hi1vqw

    Parameters
    ----------
    path : str
        path to load the object from 
    """
    with open(path, 'rb') as f:
        return pickle.load(f)