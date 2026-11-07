import torch
from sympy.integrals.rde import no_cancel_b_large

from rules import WhalrusModule
from .stochasticdominance import get_sd_loss
from .profiles import all_abstentions
from transforms.transform_alts import transform_alts_participation


def remove_row(tensor, index):
    # Validate the index
    if index < 0 or index >= tensor.size(0):
        raise ValueError("Index is out of bounds")

    # Concatenate slices before and after the specified index
    if index == 0:
        return tensor[1:]  # If the first row is to be removed
    elif index == tensor.size(0) - 1:
        return tensor[:-1]  # If the last row is to be removed
    else:
        return torch.cat((tensor[:index], tensor[index+1:]), dim=0)

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


class ParticipationLoss(torch.nn.Module):
    def __init__(self, profile_agg_func=torch.mean, voter_agg_func=torch.max, scalar=1, verbose=False):
        self.profile_agg_func = profile_agg_func
        self.voter_agg_func = voter_agg_func
        self.verbose = verbose
        self.scalar = scalar
        super(ParticipationLoss, self).__init__()

    def forward(self, data, model, concatenate=False, device=None):
        """
        Compute a participation loss over a batch.

        Expected `data` keys:
          - Profile: tensor of shape [B, n_voters, n_cands]
          - tournament / rank_matrix-normalized / weighted_tournament-normalized: tensors per sample
          - profitable_abstentions: list of length B; each entry is a dict abstainer_idx -> feature dict
            (requires datasets saved with `alternatives` including "abst" and loaded with allow_objects=True)

        Behavior:
          - Concatenates the three transforms per sample to build the original profile input.
          - Concatenates abstention transforms, batches them for a single model forward, and compares
            against the original distribution via stochastic dominance.
          - If a sample has no profitable abstentions (or the dict is missing), logs when verbose and
            assigns zero loss so training continues instead of crashing.
        """
        all_profile = data["Profile"]  # [B, n_voters, n_cands]
        evaluating_whalrus = isinstance(model, WhalrusModule)
        batch_losses = torch.zeros(all_profile.shape[0], device=device)
        for batch_idx in range(all_profile.shape[0]):
            profile = all_profile[batch_idx].to(device)
            # Build the original concatenated transform once per batch item
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

            # concatenate each profitable abstention's features
            batch_dict = data["profitable_abstentions"][batch_idx]
            if not isinstance(batch_dict, dict) or len(batch_dict) == 0:
                print(f"[ParticipationLoss] No profitable_abstentions for batch {batch_idx}; loss set to 0.")
                batch_losses[batch_idx] = torch.tensor(0.0, device=device)
                continue

            abst_inputs = []
            voter_order = []
            profile_list = []
            for voter, abst in batch_dict.items():
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
                abst_inputs.append(concatenated)
                voter_order.append(voter)
                if evaluating_whalrus:
                    abst_profile = abst['profile'].squeeze()
                    if not torch.is_tensor(abst_profile):
                        abst_profile = torch.as_tensor(abst_profile)
                    profile_list.append(abst_profile.to(device))

            if len(abst_inputs) == 0:
                print(f"[ParticipationLoss] Profitable abstentions dict present but empty after filtering at batch {batch_idx}.")
                batch_losses[batch_idx] = torch.tensor(0.0, device=device)
                continue

            abst_batch = torch.stack(abst_inputs, dim=0).to(device)

            # Compute distributions in batch
            if not evaluating_whalrus:
                orig_dist = model(t_original_profile)  # shape: (1, n_cands)
                abstention_dists = model(abst_batch)    # shape: (n_abst, n_cands)
            else:
                orig_dist = model([profile])
                abstention_dists = model(profile_list)

            # Compute SD loss per abstention (vectorized cumsums when possible)
            orig_flat = torch.flatten(orig_dist)
            abst_flat = abstention_dists.view(abstention_dists.size(0), -1)
            profile_losses = torch.stack([get_sd_loss(orig_flat, abst_flat[i], self.voter_agg_func, profile[voter_order[i]]) for i in range(abst_flat.size(0))])

            batch_losses[batch_idx] = self.profile_agg_func(profile_losses)

        # aggregate the voter losses into single loss for entire profile
        return torch.multiply(self.profile_agg_func(batch_losses), torch.tensor([self.scalar], device=device))
        

##################

        # print("Is this loss?")

        # Instantiate losses as 1d tensor matching profile size (entry per voter )
        print(f"DEVICE I SHOULD USE: {device}")
        profile_losses = torch.zeros(len(profile), device=device)

        no_transform = False
        if model.transform == None:
            no_transform = True
            transform = lambda x: x
        else:
            transform = model.transform

        # We will apply the transform from the model's state to profiles after modification:
        original_profile = transform(profile)

        # print(f"original_profile: {original_device}")

        # Compute original distribution for comparison:
        orig_dist = model(original_profile)
        # print(f"orig_dist: {orig_dist.device}")

        # For each voter in the profile:
        for i in range(len(profile)):
            profile_clone = profile.clone()
            # print(f"profile_clone: {profile_clone.device}")


            # for raw profiles (no transform), replace with garbage vote
            if no_transform:
                profile_clone[i] = torch.tensor([-1]*len(profile[i]))
            # For transforms, we can just remove the voter
            else:
                profile_clone = remove_row(profile_clone, i)
            
            # print(f"profile_clone after augment: {profile_clone.device}")

            abstaining_profile = transform(profile_clone)
            # print(f"abstaining_profile: {abstaining_device}")
            abst_dist = model(torch.tensor(abstaining_profile))
            # print(f"abst_dist: {abst_dist.device}")


            # Sort lotteries by the voter, conveniently already provides index
            # Send sorted lotteries to get SD loss
            loss = get_sd_loss(torch.flatten(orig_dist[:, profile[i]]), torch.flatten(abst_dist[:, profile[i]]), self.voter_agg_func)
            print(f"loss: {loss.device}")

            profile_losses[i] = loss.reshape(1)

        return torch.multiply(self.profile_agg_func(profile_losses), torch.tensor([self.scalar], device=device))
