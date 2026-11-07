import numpy as np
import whalrus
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def irv_vote(profile:np.ndarray):
    '''
    NOTES:
    ------
    IRV works by rounds of elimination which require tie-breaking. There is no way to really define it without tie-breaking. Not sure if whalrus implementation ever returns more than one winner

    NOTES:
    Using profile[0] to get sorted list of cands will NOT work if orders can be incomplete

    '''
    cands = sorted(profile[0])
    return to_indices(whalrus.RuleIRV(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).winner_, cands)

def irv_scf(profile:np.ndarray):
    '''
    NOTES
    -------
    Currently uses tie-breaking. Does not return all possible winners from the profile under differetn tie-breaking because this is not implemented in whalrus and computationally too expensive. So the winner is always unique. But still use .cowinners_ so that it returns an iterable set.
    '''
    cands = sorted(profile[0])
    return to_indices(whalrus.RuleIRV(tw.to_whalrus(profile), tie_break = whalrus.Priority.ASCENDING).cowinners_, cands)

def irv_pscf(profile:np.ndarray):
    '''
    NOTES:
    ------
    IRV works by rounds of elimination which require tie-breaking. There is no way to really define it without tie-breaking. So this returns all probability mass on a single winner.
    '''
    return indices_to_lottery(irv_scf(profile))