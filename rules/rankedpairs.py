import numpy as np
import whalrus
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def rankedpairs_vote(profile:np.ndarray):
    '''
    Returns single Ranked Pairs (Tideman) winner, breaking ties lexicographically.
    '''
    cands = sorted(profile[0])
    return to_indices(whalrus.RuleRankedPairs(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).winner_, cands)

def rankedpairs_scf(profile:np.ndarray):
    '''
    Can't define rankedpairs without tie-breaking so it always returns a unique winner. But still use .cowinners_ so that it returns an iterable set.

    NOTES:
    Using profile[0] to get sorted list of cands will NOT work if orders can be incomplete

    '''
    cands = sorted(profile[0])
    return to_indices(whalrus.RuleRankedPairs(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).cowinners_, cands)

def rankedpairs_pscf(profile:np.ndarray):
    return indices_to_lottery(rankedpairs_scf(profile))