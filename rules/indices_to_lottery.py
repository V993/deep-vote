import numpy as np
def indices_to_lottery(indices:np.ndarray)->np.ndarray:
    '''
    Takes in a binary indicator array of length n_cands, and returns a uniform distribution over the cowinners
    '''
    return indices.astype(float) / sum(indices)