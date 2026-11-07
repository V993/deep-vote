import numpy as np
import whalrus

def to_whalrus(profile:np.ndarray):
    '''
    RETURN
    ----------
    (whalrus.Profile): whalrus profile object converted from np.ndarray of rank ballots
    
    NOTES
    ------ 
    Whalrus can't create profile from numpy array so the profile has to be cast to a dict of int:list first'''
    return whalrus.Profile(profile.tolist())
