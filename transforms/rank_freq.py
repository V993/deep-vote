import torch
import numpy as np

# def rankfreq(profile:np.ndarray)->np.ndarray:
#     '''
    
#     RETURN
#     -------------
#     rank_freq_matrix (np.ndarray of dimension n_cands x n_cands): each row is a candidate, and each column is a rank position. rank_freq_matrix[c,i] is the total number of voters that rank candidate c in position i
    
#     '''
#     n_cands = len(profile[0])
#     rank_freq_matrix = np.zeros((n_cands, n_cands), dtype=int) #int now, make float for matrix multiplication later
#     for ranking in profile:
#         for i, cand in enumerate(ranking):
#             rank_freq_matrix[int(cand),i] += 1 #ensure id is int to use as index (must be unique)

#     rank_freq_matrix = rank_freq_matrix * 10 / len(profile)
#     return rank_freq_matrix

def rankfreq(profile:torch.Tensor)->torch.Tensor:
    '''
    
    RETURN
    -------------
    rank_freq_matrix (np.ndarray of dimension n_cands x n_cands): each row is a candidate, and each column is a rank position. rank_freq_matrix[c,i] is the total number of voters that rank candidate c in position i
    
    '''
    if not isinstance(profile, torch.Tensor):
        profile = torch.tensor(profile)
    
    profile = profile.clone()
    n_cands = len(profile[0])
    rank_freq_matrix = torch.zeros((n_cands, n_cands), dtype=float, device=profile.device) 
    for ranking in profile:
        for i, cand in enumerate(ranking):
            rank_freq_matrix[int(cand),i] += 1 #ensure id is int to use as index (must be unique)

    rank_freq_matrix = rank_freq_matrix / len(profile)
    return rank_freq_matrix