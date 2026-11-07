import numpy as np

def lex(profile:np.ndarray)->np.ndarray:
    '''Anonymize a profile by lexicographically sorting the rank ballots (rows). This prevents learning a dictatorial rule.
    
    ARGS
    ----------
    profile (np.ndarray): Each row is the rank ballot of a single voter, given as an ordered list of candidates. profile[i][j] = the candidate that voter i ranks in position j, where their top choice is j=0

    RETURNS
    -----------
    sorted_profile (np.ndarray): Permutation of rows of the original profile so that the rankings are in lexicograpic order according to the cand names/ids
    '''
    indices = np.lexsort(profile.T[::-1])
    sorted_profile = profile[indices]
    return sorted_profile