import torch
import numpy as np
import transforms.weighted_tournament as wt

# def tournament(profile:np.ndarray):
#     n_voters = len(profile)
#     w_tournament = wt.weighted_tournament(profile)
#     return w_to_tournament(w_tournament, n_voters)

# def w_to_tournament(w_tournament:np.ndarray, n_voters:int)->np.ndarray:
#     '''
#     Computes the (unweighted) tournament transform from a weighted tournament transform

#     RETURN
#     ----------
#     tournament (np.ndarray of dimension n_cands x n_cands): tournament[i,j] = 1 if more voters prefer i to j, 1/2 if equal number of voters prefer each, and 0 if more voters prefer j to i.
#     '''
#     tournament = w_tournament.astype(float)
#     tournament[w_tournament == n_voters/2.0] = 0.5
#     tournament[w_tournament < n_voters/2.0] = 0
#     tournament[w_tournament > n_voters/2.0] = 1
#     for c in range(len(tournament)): #set diagonal to 0.5, so cadns tie with themselves
#         tournament[c,c] = 0.5
#     return tournament

def tournament(profile:torch.Tensor)->torch.Tensor:
    n_voters = len(profile)
    w_tournament = wt.weighted_tournament(profile)
    return w_to_tournament(w_tournament, n_voters)

def w_to_tournament(w_tournament:torch.Tensor, n_voters:int)->torch.Tensor:
    '''
    Computes the (unweighted) tournament transform from a weighted tournament transform

    RETURN
    ----------
    tournament (torch.Tensor of dimension n_cands x n_cands): tournament[i,j] = 1 if more voters prefer i to j, 1/2 if equal number of voters prefer each, and 0 if more voters prefer j to i.
    '''
    if not isinstance(w_tournament, torch.Tensor):
        tournament = torch.tensor(w_tournament).clone().float()
    else:
        tournament = w_tournament.clone().float()
    tournament[w_tournament == 0.5] = 0.5
    tournament[w_tournament < 0.5] = 0
    tournament[w_tournament > 0.5] = 1
    for c in range(len(tournament)): #set diagonal to 0.5, so cands tie with themselves
        tournament[c,c] = 0.5
    return tournament