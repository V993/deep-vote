import numpy as np
import whalrus
from whalrus import RuleCondorcet
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def condorcet_vote(profile:np.ndarray):
    '''
    If there is no unique Condorcet winner, it returns cand that comes first in lexicographic order
    '''
    cands = sorted(profile[0])
    return to_indices(RuleCondorcet(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).winner_, cands)

def condorcet_scf(profile:np.ndarray):
    '''
    If there is no unique Condorcet winner, it returns all cands as cowinners

    NOTES:
    Using profile[0] to get sorted list of cands will NOT work if orders can be incomplete
    '''
    cands = sorted(profile[0])
    return to_indices(RuleCondorcet(tw.to_whalrus(profile)).cowinners_, cands)

def condorcet_pscf(profile):
    '''
    Returns lottery with either all mass on Condorcet winner if it exists, otherwise uniform distribution over all candidates
    '''
    return indices_to_lottery(condorcet_scf(profile))