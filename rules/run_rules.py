from collections.abc import Callable
import numpy as np
from utils.callable_to_str import callable_to_str
# from . import numpy_ranking_to_whalrus as tw

def run_rules_one_profile(rules:list[Callable], profile:np.ndarray)->dict:
    '''
    Given a list of rules and a given profile, return the outcomes from every rule on that profile.
    '''
    outcomes = {callable_to_str(rule):rule(profile) for rule in rules}
    return outcomes