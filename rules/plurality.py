import numpy as np
import whalrus
from whalrus import RulePlurality
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def plurality_vote(profile:np.ndarray):
    '''
    Returns single plurality winner, breaking ties lexicographically
    '''
    cands = sorted(profile[0])
    return to_indices(RulePlurality(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).winner_, cands)


def plurality_scf(profile:np.ndarray):
    '''
    Returns set of cands with highest Plurality scores as cowinners

    NOTES:
    Using profile[0] to get sorted list of cands will NOT work if orders can be incomplete

    '''
    cands = sorted(profile[0])
    return to_indices(RulePlurality(tw.to_whalrus(profile)).cowinners_, cands)

def plurality_pscf(profile:np.ndarray):
    '''
    Returns lottery that is uniform over Plurality winners
    '''
    return indices_to_lottery(plurality_scf(profile))