#create uniformly random preference rankings and ranking profiles using numpy
import numpy as np
# from itertools import permutations
from collections.abc import Callable
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def load_random_labeled_profiles(rule, transform, n_voters, n_cands, n_profiles, batch_size, rng, device="cpu"):
    if device=="cpu":
        print("RUNNING ON CPU!")
    dataset = RandomLabeledProfiles(n_voters, n_cands, n_profiles, rng, rule, transform, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_random_profiles(transform, n_voters, n_cands, n_profiles, batch_size, rng, device="cpu"):
    if device=="cpu":
        print("RUNNING ON CPU!")
    dataset = RandomProfiles(n_voters, n_cands, n_profiles, rng, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_random_labeled_profiles_full(rule, transform, n_voters, n_cands, n_profiles, batch_size, rng, device="cpu", **kwargs):
    if device=="cpu":
        print("RUNNING ON CPU!")
    
    dataset = [rand_rankings_profile(n_voters, n_cands, rng, device)]*n_profiles
    outcomes = [rule(item) for item in dataset]
    if transform is not None:
        dataset = [transform(item) for item in dataset]
    
    dataset = torch.tensor(zip(outcomes, dataset), device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


class RandomLabeledProfiles(Dataset):
    def __init__(self, n_voters:int, n_cands:int, n_profiles:int, rng, rule:Callable, transform:Callable=None, device:str="cpu"):
        self.n_profiles = n_profiles
        self.n_voters = n_voters
        self.n_cands = n_cands
        self.rng = rng
        self.rule = rule
        self.transform = transform
        self.device = device

    def __len__(self):
        return self.n_profiles

    def __getitem__(self, idx):#->tuple[np.ndarray, np.ndarray]
        profile = rand_rankings_profile(self.n_voters, self.n_cands, self.rng, self.device)
        outcome = self.rule(profile.numpy())
        if self.transform is not None:
            return self.transform(profile), outcome
        else:
            return profile, outcome
    
class RandomProfiles(Dataset):
    def __init__(self, n_voters:int, n_cands:int, n_profiles:int, rng, transform:Callable=None):
        self.n_profiles = n_profiles
        self.n_voters = n_voters
        self.n_cands = n_cands
        self.rng = rng
        self.transform = transform

    def __len__(self):
        return self.n_profiles

    def __getitem__(self, idx):
        profile = rand_rankings_profile(self.n_voters, self.n_cands, self.rng)
        return profile
    
        # Legacy code, repurposing to return profiles and transform later
        if self.transform is not None:
            return self.transform(profile)
        else:
            return profile


def rand_rankings_profile(n_voters:int, n_cands:int, rng)->np.ndarray:
    '''Generates a single rankings profile uniformly at random with n_voters number of complete rank ballots
    
    RETURN
    ------------
    permutations (np.ndarray): Each row is a ballot of a single voter (ordered list of candidates)
    '''

    permutations = np.zeros((n_voters, n_cands), dtype=int)
    for i in range(n_voters):
        permutations[i] = rng.permutation(n_cands)
    return permutations

def rand_rankings_profiles_gen(n_voters:int, n_cands:int, n_profiles:int, rng=None)->np.ndarray:
    '''Generator for yielding n_profiles number of uniformly random profiles
    
    YIELD
    -------------
    (np.ndarray): Each row is a ballot of a single voter (ordered list of candidates)
    '''
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(n_profiles):
        yield rand_rankings_profile(n_voters, n_cands, rng)