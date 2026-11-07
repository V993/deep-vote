# import numpy as np
import torch

# def create_all_swaps(ranking:np.ndarray)->np.ndarray:
#     '''
#     RETURNS:
#     all_swaps (list of 1D np.ndarrays) all rankings that could be obtained by adjacent swaps of original ranking
#     does not include the original ranking in its returns
#     '''
#     all_swaps = []
#     for i in range(len(ranking)-1):
#         swap = np.copy(ranking)
#         swap[i], swap[i+1] = swap[i+1], swap[i]
#         all_swaps.append(swap)
#     return all_swaps
    
# # create_all_swaps(np.array([1,2,3,4]))


# def create_all_voter_swaps(profile:np.ndarray, voter:int):
#     '''
    
#     RETURNS:
#     all_voter_swaps (list of 2D np.ndarrays) all profiles that could be obtained if given voter made a single swap
#     '''
    
#     all_swaps = []
#     ranking = profile[voter]
#     for swapped_ranking in create_all_swaps(ranking):
#         swapped_profile = np.copy(profile)
#         swapped_profile[voter] = swapped_ranking
#         all_swaps.append(swapped_profile)
#     return np.asarray(all_swaps)
    
# # create_all_voter_swaps(np.array([[1,2,3,4],[4,3,2,1]]),0)

# def create_all_swaps_profiles(profile:np.ndarray):
#     '''returns all possible profiles obtainable by a single swap from any voter from the original profile'''
#     all_swaps = []
#     for voter, _ in enumerate(profile):
#         for prof in create_all_voter_swaps(profile,voter):
#             all_swaps.append(prof)
#     return np.asarray(all_swaps)

def all_swaps(ranking:torch.tensor)->torch.tensor:
    '''
    RETURNS:
    all_swaps (tensor of 1D tensors) all rankings that could be obtained by adjacent swaps of original ranking
    does not include the original ranking in its returns

    NOTES:
    Can't do swap[i],swap[i+1] = swap[i+1],swap[i] with tensors. Works with numpy.
    '''
    swaps = torch.zeros_like(ranking.unsqueeze(0), dtype=int)
    for i in range(len(ranking)-1):
        swap = torch.clone(ranking)
        temp = torch.clone(swap[i])
        swap[i] = swap[i+1]
        swap[i+1] = temp
        swaps = torch.cat((swaps,swap.unsqueeze(0)))
    return swaps[1:]

def all_voter_swaps(profile:torch.tensor, voter:int)->torch.tensor:
    '''
    
    RETURNS:
    all_voter_swaps (tensor of 2D tensors) all profiles that could be obtained if given voter made a single swap
    '''
    
    swaps = torch.zeros_like(profile.unsqueeze(0), dtype=int)
    ranking = profile[voter]
    for swapped_ranking in all_swaps(ranking):
        swapped_profile = torch.clone(profile)
        swapped_profile[voter] = swapped_ranking
        swaps = torch.cat((swaps,swapped_profile.unsqueeze(0)))
    return swaps[1:]

def all_swaps_profiles(profile:torch.tensor)->torch.tensor:
    '''returns all possible profiles obtainable by a single swap from any voter from the original profile'''
    # swaps = torch.zeros_like(profile.unsqueeze(0), dtype=int)
    d_alts = {}
    for voter, _ in enumerate(profile):
        for profid, prof in enumerate(all_voter_swaps(profile,voter)):
            # all_swaps = torch.cat((all_swaps,prof.unsqueeze(0)))
            d_alts[voter, profid] = prof
    return d_alts


def all_swaps_generator(ranking:torch.tensor)->torch.tensor:
    '''
    YIELDS:
    swap (1D tensor) ranking that could be obtained by adjacent swaps of original ranking

    NOTES:
    Can't do swap[i],swap[i+1] = swap[i+1],swap[i] with tensors. Gives wrong answer. Might be a better way :/
    '''
    for i in range(len(ranking)-1):
        swap = torch.clone(ranking)
        temp = torch.clone(swap[i])
        swap[i] = swap[i+1]
        swap[i+1] = temp
        yield swap

def all_swaps_profiles_generator(profile:torch.tensor)->torch.tensor:
    '''yields all possible profiles obtainable by a single swap from any voter from the original profile'''
    for voter, _ in enumerate(profile):
        for prof in all_voter_swaps(profile,voter):
            yield prof

    
# x = create_all_swaps_profiles(torch.tensor([[1,2,3,4],[4,3,2,1]]))
# print(x)