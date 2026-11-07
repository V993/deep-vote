import numpy as np
import whalrus
from whalrus import RuleCopeland
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def copeland_vote(profile:np.ndarray):
    '''
    Returns cand with highest Copeland score, breaking ties lexicographically.
    If Condorcet winner is unique, then they are the unique winner

    NOTES:
    Using profile[0] to get sorted list of cands will NOT work if orders can be incomplete
    '''
    cands = sorted(profile[0])
    return to_indices(RuleCopeland(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).winner_, cands)

def copeland_scf(profile:np.ndarray):
    '''
    Returns set of cands with highest Copeland scores as cowinners.
    If Condorcet winner is unique, then they are the unique winner
    All copeland winners are within the Top Cycle / Smith Set
    '''
    cands = sorted(profile[0])
    return to_indices(RuleCopeland(tw.to_whalrus(profile)).cowinners_, cands)

def copeland_pscf(profile:np.ndarray):
    '''
    Returns uniform lottery over cands with highest Copeland scores.
    If Condorcet winner is unique, then all porb mass on the unique winner
    All copeland winners are within the Top Cycle / Smith Set
    '''
    return indices_to_lottery(copeland_scf(profile))