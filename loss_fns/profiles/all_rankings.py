import numpy as np
from itertools import permutations
from itertools import product

def all_single_rankings(n_cands:int)->np.ndarray: #iterator
    '''
    NOTES
    ------------ 
    Cand ids are ints [0, ..., n_cands-1] so that they can be used for indexing
    '''
    return np.asarray(list(permutations(range(n_cands))))

def all_single_rankings_gen(n_cands:int)->np.ndarray:
    '''Generator for yielding all possible rank ballots (rankings by single voter) iteratively'''
    for r in permutations(range(n_cands)):
        yield np.array(r)

def all_rankings_profiles(n_voters:int, n_cands:int)->np.ndarray:
    '''Return all possible profiles with rank ballots'''
    return np.array(list(product(permutations(range(n_cands), n_cands), repeat=n_voters)))

def all_rankings_profiles_gen(n_voters:int, n_cands:int)->np.ndarray:
    '''Generator for returning all possible profiles of rank ballots'''
    for profile in product(permutations(range(n_cands), n_cands), repeat=n_voters):
        yield np.asarray(profile)