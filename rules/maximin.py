import numpy as np
import whalrus
from whalrus import RuleMaximin
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def maximin_vote(profile:np.ndarray):
    '''
    Returns Maximin winner, breaking ties lexicographically

    NOTES:
    Using profile[0] to get sorted list of cands will NOT work if orders can be incomplete
    '''
    cands = sorted(profile[0])
    return to_indices(RuleMaximin(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).winner_, cands)

def maximin_scf(profile:np.ndarray):
    '''
    Returns the set of cowinners with highest Maximin score
    '''
    cands = sorted(profile[0])
    return to_indices(RuleMaximin(tw.to_whalrus(profile)).cowinners_, cands)

def maximin_pscf(profile):
    '''
    Returns a uniform distribution over candidates with maximum borda score.
    Note that it does NOT give the distribution proportionate to borda score. 
    '''
    return indices_to_lottery(maximin_scf(profile))