import numpy as np
import whalrus
from whalrus import RuleSchulze
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def schulze_vote(profile:np.ndarray):
    '''
    Returns single Schulze winner, breaking ties lexicographically

    NOTES:
    Using profile[0] to get sorted list of cands will NOT work if orders can be incomplete

    '''
    cands = sorted(profile[0])
    return to_indices(RuleSchulze(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).winner_, cands)

def schulze_scf(profile:np.ndarray):
    '''
    Returns Schulze cowinners
    '''
    cands = sorted(profile[0])
    return to_indices(RuleSchulze(tw.to_whalrus(profile)).cowinners_, cands)

def schulze_pscf(profile:np.ndarray):
    '''
    Returns uniform distribution over Schulze cowinners
    '''
    return indices_to_lottery(schulze_scf(profile))