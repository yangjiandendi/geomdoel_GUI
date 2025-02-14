"""
utils.py

This module contains additional utility functions that may be used across different
modules in the project.
"""

def combine_funcs(*funcs):
    """
    Return a function that calls all provided functions sequentially.
    
    Parameters:
        funcs: Functions to be combined.
        
    Returns:
        A function that calls each input function in sequence.
    """
    def inner(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return inner
