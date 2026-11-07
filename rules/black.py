import numpy as np
from whalrus import RuleBorda
from whalrus import RuleCondorcet
from . import numpy_ranking_to_whalrus as tw
from rules.cowinners_to_indices import to_indices
from rules.indices_to_lottery import indices_to_lottery

def black(profile:np.ndarray):
    cands = sorted(profile[0])
    c = RuleCondorcet(tw.to_whalrus(profile)).cowinners_
    if len(c) == 1:
        return to_indices(c, cands)
    else:
        return to_indices(RuleBorda(tw.to_whalrus(profile)).cowinners_, cands)
    
def black_pscf(profile):
    return indices_to_lottery(black(profile))