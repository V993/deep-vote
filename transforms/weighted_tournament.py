
import copy
import torch
import numpy as np

# def weighted_tournament(profile:np.ndarray)->np.ndarray:
#     '''
#     Computes the weighted tournament transform from a complete rank ballots profile.

#     ARGS
#     ------------
#     profile (np.ndarray): Each row is the rank ballot of a single voter, given as an ordered list of candidates. profile[i][j] = the candidate that voter i ranks in position j, where the voter's top choice is j=0

#     RETURN
#     -------------
#     w_tournament (np.ndarray): tournament[i,j] = is the number of voters who (weakly) prefer i to j. It is (weakly), even though preference orders are strict, so that the elements on the diagonal are equal to the number of voters.
#     '''
#     n_cands = len(profile[0])
#     w_tournament = np.zeros((n_cands, n_cands), dtype=int)
#     for ranking in profile:
#         for i, cand in enumerate(ranking):
#             for _, cand2 in enumerate(ranking[i:]):
#                 w_tournament[int(cand),int(cand2)] += 1 #needs to be int to use cand IDs as indices

#     w_tournament = w_tournament * 10 / len(profile)
#     return w_tournament

def weighted_tournament(profile:torch.Tensor)->torch.Tensor:
    '''
    Computes the weighted tournament transform from a complete rank ballots profile.

    ARGS
    ------------
    profile (torch.Tensor): Each row is the rank ballot of a single voter, given as an ordered list of candidates. profile[i][j] = the candidate that voter i ranks in position j, where the voter's top choice is j=0

    RETURN
    -------------
    w_tournament (torch.Tensor): tournament[i,j] = is the number of voters who (weakly) prefer i to j. It is (weakly), even though preference orders are strict, so that the elements on the diagonal are equal to the number of voters.
    '''

    if not isinstance(profile, torch.Tensor):
        profile = torch.tensor(profile)
    n_cands = len(profile[0])
    w_tournament = torch.zeros((n_cands, n_cands), dtype=float, device=profile.device)
    for ranking in profile:
        for i, cand in enumerate(ranking):
            for cand2 in ranking[i+1:]:
                if cand >=0 and cand2 >= 0:
                    w_tournament[int(cand),int(cand2)] += 1 #needs to be int to use cand IDs as indices

    w_tournament = w_tournament / len(profile)
    return w_tournament

def w_tournament_after_swap(w_tournament:torch.Tensor, cand_down:int, cand_up:int, n_voters:int, normalized=True)->torch.Tensor:
    '''
    Computes the updated weighted tournament when two adjacent candidates have been swapped
    '''
    if not torch.is_tensor(w_tournament):
        w_tournament = torch.tensor(w_tournament)
    

    new_tournament = w_tournament.clone() # Keeps gradients

    new_tournament[cand_down, cand_up] = new_tournament[cand_down, cand_up]-(normalized/n_voters)
    new_tournament[cand_up, cand_down] = new_tournament[cand_up, cand_down]+(normalized/n_voters)

    return new_tournament


# w_tournament=tensor([[0.5000, 0.0000, 0.4000],
#         [1.0000, 0.5000, 0.8000],
#         [0.6000, 0.2000, 0.5000]])
# cand_down=1
# cand_up=0
# normalized=True
# n_voters=5
# modified w_tournament=tensor([[0.5000, 0.2000, 0.4000],
#         [0.8000, 0.5000, 0.8000],
#         [0.6000, 0.2000, 0.5000]])

def w_tournament_after_abstention(w_tournament:torch.Tensor, original_ranking:torch.Tensor, n_voters:int, normalized=True)->torch.Tensor:
    '''

    Computes the updated weighted tournament based on original weighted tournament when a voter from the original profile abstains

    n_voters is numebr of voter in the original profile, not accounting for the abstention
    '''
    if not torch.is_tensor(w_tournament):
        w_tournament = torch.tensor(w_tournament)
    if not torch.is_tensor(original_ranking):
        original_ranking = torch.tensor(original_ranking)
    if normalized:
        w_tournament  = w_tournament * n_voters
    original_ranking = original_ranking.clone()

    for i, cand in enumerate(original_ranking[:-1]):
        for cand2 in original_ranking[i+1:]:
            w_tournament[int(cand), int(cand2)] = w_tournament[int(cand), int(cand2)] - 1
    return w_tournament / (n_voters-1)
