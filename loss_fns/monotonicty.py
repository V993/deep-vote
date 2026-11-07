

import re
import torch
from os import cpu_count
from transforms.rank_freq import rankfreq
from transforms.tournament import w_to_tournament
from .stochasticdominance import get_sd_loss
from transforms.transform_alts import transform_alts
from loss_fns.profiles.adjacent_swaps import all_swaps_profiles
from transforms.weighted_tournament import w_tournament_after_swap
from rules import WhalrusModule

def _normalize_transform_name(transform_name):
    if isinstance(transform_name, bool):
        return "concatenate" if transform_name else None
    return transform_name

def _resolve_transform_key(transform_name, container):
    name = _normalize_transform_name(transform_name)
    if not name or name == "concatenate":
        return None
    if name in ("weighted_tournament", "weighted_tournament-normalized"):
        return "weighted_tournament-normalized" if "weighted_tournament-normalized" in container else "weighted_tournament"
    if name in ("rankfreq", "rank_matrix", "rank_matrix-normalized"):
        return "rank_matrix-normalized" if "rank_matrix-normalized" in container else "rankfreq"
    return name


def get_swap_loss(original, swapped, cand, agg_func):
    """Returns the loss from a single swap

    Args:
        original (torch.tensor): original profile's distribution
        swapped (torch.tensor): single-swapped profile's distribution
        cand (int): the candidate who stands to gain, swapped to a higher ranking. Equivalent to index of distribution
    
    Returns:
        torch.tensor: Loss over the single swap 
    """

    # get probability of candidate before their swap
    original_prob = original[cand]

    # get probability of candidate after they are swapped up
    swapped_prob = swapped[cand]

    # return the positive loss as max of 0 tensor and the difference between swapped and original (positive denotes error if the original is greater than the swapped)
    return torch.max(original_prob-swapped_prob,torch.tensor([0.], device=original.device))

class MonotonictyLoss(torch.nn.Module):
    def __init__(self, profile_agg_func=torch.mean, voter_agg_func=torch.max, scalar=1, verbose=False):
        self.profile_agg_func = profile_agg_func
        self.voter_agg_func = voter_agg_func # Should always be max
        self.verbose = verbose
        self.scalar = scalar
        super(MonotonictyLoss, self).__init__()

    def forward(self, data, model, concatenate=False, device=None):
        """Takes original distribution and abstain distribution (ordered by preference) and outputs max of the differences

        Args:
            profile (tensor or np.ndarray): profile over which we compute loss
            model (nn.Module): model being trained
        
        Returns:
            tensor: loss over the profile
        """

        # print("Is this loss?")

        # Precompute all the alternate profiles from adjacent swaps, as dict
        # alts = all_swaps_profiles(profile)
        # t_original_profile, transformed_alts = transform_alts(profile, alts, model.transform)

        # For each alternative profile, get model output distribution
        # swapped_dists = {key:model(alt_prof) for key,alt_prof in transformed_alts.items()}
        
        all_profile = data["Profile"]  # [B, n_voters, n_cands]
        evaluating_whalrus = isinstance(model, WhalrusModule)
        batch_losses = torch.zeros(all_profile.shape[0], device=device)
        for batch_idx in range(all_profile.shape[0]):
            profile = all_profile[batch_idx].to(device)

            transform_key = _resolve_transform_key(concatenate, data)
            if transform_key is None:
                orig_parts = [
                    data['tournament'][batch_idx].to(device),
                    data['weighted_tournament-normalized'][batch_idx].to(device),
                    data['rank_matrix-normalized'][batch_idx].to(device),
                ]
                orig_parts = [p if p.dim() > 1 else p.unsqueeze(0) for p in orig_parts]
                t_original_profile = torch.cat(orig_parts, dim=1)
                if t_original_profile.dim() == 2:
                    t_original_profile = t_original_profile.unsqueeze(0)  # add batch dim for model
            else:
                t_original_profile = data[transform_key][batch_idx].to(device)
                if t_original_profile.dim() == 2:
                    t_original_profile = t_original_profile.unsqueeze(0)

            # concatenate each topk adjacent swap's features
            batch_dict = data["topk_swaps"][batch_idx]
            if not isinstance(batch_dict, dict) or len(batch_dict) == 0:
                # Nothing profitable to evaluate; keep training but note it for debugging.
                print(f"[MonotonicityLoss] No profitable_abstentions for batch {batch_idx}; loss set to 0.")
                batch_losses[batch_idx] = torch.tensor(0.0, device=device)
                continue
                
            # concatenate each swap's features
            topk_inputs = []
            swap_order = [] # Used to retrieve the candidate which is being swapped up (k+1)
            profile_list = []
            for (voter, candidate_swapped_down), abst in batch_dict.items():
                # add one key for each voter/k-th swap. e.g., (2,3) indicates voter 2 swapping ranks 3/4 in their ordering
                abst_key = _resolve_transform_key(concatenate, abst)
                if abst_key is None:
                    parts = [
                        abst['tournament'].to(device),
                        abst['rank_matrix-normalized'].to(device),
                        abst['weighted_tournament-normalized'].to(device)
                    ]
                    parts = [p if p.dim() > 1 else p.unsqueeze(0) for p in parts]
                    concatenated = torch.cat(parts, dim=1)
                else:
                    concatenated = abst[abst_key].to(device)
                    if concatenated.dim() == 1:
                        concatenated = concatenated.unsqueeze(0)
                topk_inputs.append(concatenated)
                swap_order.append((voter, candidate_swapped_down))
                if evaluating_whalrus:
                    profile_list.append(abst['profile'].squeeze())
            
            # swap_dists = {}
            # for vk, swap in data["topk_swaps"].items():
            #     # add one key for each voter/k-th swap. e.g., (2,3) indicates voter 2 swapping ranks 3/4 in their ordering
            #     concatenated = torch.cat(
            #         [swap[vk]['tournament'], swap[vk]['rank_matrix-normalized'],
            #         swap[vk]['weighted_tournament-normalized']], dim=1
            #     )
            #     swap_dists[vk] = model(concatenated)

            if len(topk_inputs) == 0:
                print(f"[MonotonicityLoss] Profitable abstentions dict present but empty after filtering at batch {batch_idx}.")
                batch_losses[batch_idx] = torch.tensor(0.0, device=device)
                continue

            topk_batch = torch.stack(topk_inputs, dim=0).to(device)

            # Compute original distribution for comparison:

            # Compute distributions in batch
            if not evaluating_whalrus:
                orig_dist = model(t_original_profile) # shape: (1, n_cands)
                swap_dists = model(topk_batch) # shape: (n_abst, n_cands)
            else:
                orig_dist = model([profile])
                swap_dists = model(profile_list)

            # Compute swap loss per adjacent swap (vectorized cumsums when possible)
            orig_flat = torch.flatten(orig_dist)
            swaps_flat = swap_dists.view(swap_dists.size(0), -1)

            # Here we compute the swap loss w.r.t. the ith swap's 2nd index (candidate swapped down) +1, which is the candidate swapped up.
            profile_losses = torch.stack(
                [get_swap_loss(
                    orig_flat,
                    swaps_flat[i],
                    (profile[swap_order[i][0]][swap_order[i][1]+1]),
                    self.voter_agg_func)
                    for i in range(swaps_flat.size(0))])

            # Aggregate all of the batch losses into a single batch loss:
            batch_losses[batch_idx] = self.profile_agg_func(profile_losses)

        # aggregate the voter losses into single loss for entire profile
        return torch.multiply(self.profile_agg_func(batch_losses), torch.tensor([self.scalar], device=device))




        # # Instantiate losses as 1d tensor matching profile size (entry per voter )
        # profile_losses = torch.zeros(len(all_profile), device=all_device)

        # no_transform = False
        # if model.transform == None:
        #     no_transform = True
        #     transform = lambda x: x
        # else:
        #     transform = model.transform

        # # We will apply the transform from the model's state to profiles after modification:
        # original_profile = transform(all_profile)

        # # Compute original distribution for comparison:
        # orig_dist = model(torch.tensor(original_profile))

        # # For each voter in the profile:
        # for i in range(len(all_profile)):
        #     if self.verbose:
        #         print()
        #         print(f"--------------------------\nComputing swaps for {i}th voter...")

        #     adjacent_swap_losses = torch.zeros(len(all_profile[i]), device=all_device)
        #     # Transform the profile with all possible adjacent swaps:
        #     for j in range(len(all_profile[i])-1):
        #         swapped = all_profile.clone()

        #         # swap the adjacent rankings, where j+1 is lower in rank than j
        #         swapped[i][[j,j+1]] = swapped[i][[j+1,j]]
                
        #         # j+1 is the index of the candidate who stands to gain:
        #         cand = all_profile[i][j+1]

        #         swapped_profile = transform(swapped)
        #         swapped_dist = model(torch.tensor(swapped_profile))

        #         # compute loss for single swap:
        #         loss = get_swap_loss(torch.flatten(orig_dist), torch.flatten(swapped_dist), cand)

        #         if self.verbose:
        #             print()
        #             print(f"Original")
        #             print(all_profile[i])
        #             print(orig_dist)
        #             print()
        #             print(f"Swapped {j}, {j+1}")
        #             print(swapped[i])
        #             print(f"Candidate: {cand}")
        #             print(swapped_dist)
        #             print(f"loss {loss}")
    
        #         adjacent_swap_losses[j] = loss
            
        #     # compute loss over a voter as the aggregate of all adjacent swap losses:
        #     profile_losses[i] = self.voter_agg_func(adjacent_swap_losses)
    
        # # compute loss over the profile as the aggregate of all profile losses:
        # return torch.multiply(self.profile_agg_func(profile_losses), torch.tensor([self.scalar]))
