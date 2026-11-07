import torch
from os import cpu_count

def check_sd(dist1:torch.Tensor, dist2:torch.Tensor):
    '''Checks if dist1 stochastically dominates dist2

    NOTES:
    Assumes the distributions are already in the correct order
    '''
    dist1_sums, dist2_sums = torch.cumsum(torch.tensor(dist1),dim=0), torch.cumsum(torch.tensor(dist2),dim=0)
    diffs = torch.subtract(dist1_sums, dist2_sums)
    return torch.all(diffs >= 0)


def check_sd_ref(original_lottery, alternative_lottery, reference_order):
    '''
    Checks if the alternative lottery stochastically dominates the original lottery according to the reference order (e.g. preference order of a particular voter)
    '''

    #1. reorder the lotteries according to the reference order
    #2. call check_sd

    pass

def get_sd_loss_old(original_lottery, alternative_lottery, voter_agg_func):

    if original_lottery.shape != alternative_lottery.shape:
        raise ValueError(f"Input distributions cannot be of different shapes. Recieved lengths: \n\t{original_lottery.shape}, {alternative_lottery.shape}")

    # print(f"get_sd_loss, original_lottery: {original_lottery}")
    # print(f"get_sd_loss, alternative_lottery: {alternative_lottery}")

    orig_pre, abst_pre = torch.tensor([0.0], device=original_lottery.device), torch.tensor([0.0], device=original_lottery.device)

    # print(f"get_sd_loss, orig_pre: {orig_pre.device}")
    # print(f"get_sd_loss, abst_pre: {abst_pre.device}")

    prefixes = torch.tensor([0.0]*len(original_lottery), device=original_lottery.device)

    # print(f"get_sd_loss, prefixes: {prefixes.device}")

    # Iterate over all candidates in lottery
    # Ignore last index for efficiency, which will always have a diff of 0.
    for i in range(len(original_lottery)-1):
        # print("Iterating over candidates in lottery")
        orig_pre += original_lottery[i]
        abst_pre += alternative_lottery[i]

        # by subtracting the original from the abstaining lottery, we have a result where a
        # strong positive value denotes a higher degree of "failure to stochastically dominate"
        dist_loss = torch.max(torch.tensor([0.0], device=original_lottery.device), abst_pre-orig_pre)

        prefixes[i] = dist_loss
    
    return voter_agg_func(prefixes)


def get_sd_loss(original_lottery, alternative_lottery, voter_agg_func, pref_order):
    if original_lottery.shape != alternative_lottery.shape:
        raise ValueError(
            f"Input distributions cannot be of different shapes. Recieved lengths: \n\t{original_lottery.shape}, {alternative_lottery.shape}")

    # Reorder both lotteries according to preference order
    orig_ordered = original_lottery[pref_order]
    abst_ordered = alternative_lottery[pref_order]

    # Compute cumulative sums along the preference order
    orig_cumsum = torch.cumsum(orig_ordered, dim=0)
    abst_cumsum = torch.cumsum(abst_ordered, dim=0)

    # Compute distance loss for all positions at once
    dist_loss = torch.clamp(abst_cumsum - orig_cumsum, min=0.0)

    # # Create output tensor and scatter results back to original positions
    # prefixes = torch.zeros_like(original_lottery)
    # prefixes[pref_order] = dist_loss

    return voter_agg_func(dist_loss)