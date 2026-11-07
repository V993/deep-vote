
import torch

def get_concatenated_data(data, transforms=[ "tournament", "rankfreq", "weighted_tournament" ]):
    
    concatenated = torch.Tensor([])
    for transform in transforms:
        concatenated = torch.cat((concatenated, data[transform]), dim=1) # Assumes shape = (batch_size, n_cands, n_cands)
    
    return concatenated 

class MajorityWinnerLoss(torch.nn.Module):
    def __init__(self, verbose=False):
        self.verbose = verbose
        super(MajorityWinnerLoss, self).__init__()
        
    def _majority_winner_single(self, profile):
        """Return the candidate index if a strict majority winner exists."""

        n_voters, n_cands = profile.shape
        half_voters = n_voters / 2.0

        for c in range(n_cands):
            num_first = (profile[:, 0] == c).float().sum().item()
            if num_first > half_voters:
                return c

        return None

    def _ensure_batch_profiles(self, profiles, n_cands):
        if not torch.is_tensor(profiles):
            raise TypeError("profiles must be a torch.Tensor")

        if n_cands is None:
            if profiles.numel() == 0:
                raise ValueError("Cannot infer number of candidates from empty profile tensor")
            n_cands = int(torch.amax(profiles).item()) + 1

        if profiles.dim() == 1:
            if profiles.numel() % n_cands != 0:
                raise ValueError(
                    "Flattened profile length must be a multiple of number of candidates"
                )
            n_voters = profiles.numel() // n_cands
            profiles = profiles.reshape(1, n_voters, n_cands)

        elif profiles.dim() == 2:
            batch, width = profiles.shape

            if width == n_cands:
                profiles = profiles.unsqueeze(0)
            elif batch == n_cands:
                profiles = profiles.transpose(0, 1).unsqueeze(0)
            elif width % n_cands == 0:
                n_voters = width // n_cands
                profiles = profiles.reshape(batch, n_voters, n_cands)
            elif profiles.numel() % n_cands == 0:
                n_voters = profiles.numel() // n_cands
                profiles = profiles.reshape(1, n_voters, n_cands)
            else:
                raise ValueError(
                    "Unable to reshape 2D profile tensor into [batch, voters, candidates]"
                )

        elif profiles.dim() == 3:
            if profiles.size(-1) == n_cands:
                pass
            elif profiles.size(1) == n_cands:
                profiles = profiles.transpose(1, 2)
            else:
                raise ValueError(
                    "Expected candidates to align with the last dimension of the profile tensor"
                )
        else:
            raise ValueError(
                f"Expected profile tensor with 1, 2, or 3 dimensions, got {profiles.dim()}"
            )

        return profiles

    def majority_winner(self, profiles, n_cands=None):
        """Returns the candidate index for each profile when a majority winner exists.

        Args:
            profiles (torch.Tensor): Profile tensor in any supported shape.
            n_cands (Optional[int]): Number of candidates. When omitted, it is
                inferred from the profile values.

        Returns:
            Optional[int] or List[Optional[int]]: Majority winner indices (None if
                no majority winner). If a batch is provided, a list is returned
                with one entry per profile.
        """

        batch_profiles = self._ensure_batch_profiles(profiles, n_cands)
        winners = [self._majority_winner_single(profile) for profile in batch_profiles]

        return winners[0] if len(winners) == 1 else winners

    def majority_winner_continuous(self, logits, profiles):
        """The theoretically continuous majority winner loss for a batch of profiles"""

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        num_candidates = logits.size(-1)
        profiles = self._ensure_batch_profiles(profiles, num_candidates)

        batch_size = logits.size(0)
        majority_indices = self.majority_winner(profiles, num_candidates)

        # Ensure we always iterate with a list of indices (one per profile).
        if not isinstance(majority_indices, list):
            majority_indices = [majority_indices]

        if len(majority_indices) != batch_size:
            raise ValueError(
                f"Batch size mismatch between logits and profiles after processing: {batch_size=}, {len(majority_indices)=} {profiles.shape=}"
            )

        batch_losses = []
        for batch_idx in range(batch_size):
            maj_winner_idx = majority_indices[batch_idx]
            logit = logits[batch_idx]

            if logit.size(-1) != num_candidates:
                raise ValueError(
                    "Candidate dimension of logits does not match number of candidates"
                )

            if logit.dim() == 1:
                candidate_scores = logit
            else:
                # Collapse any additional dimensions (e.g., voters) while keeping
                # gradients intact, then aggregate per candidate.
                candidate_scores = logit.reshape(-1, num_candidates).mean(dim=0)

            zero = candidate_scores.new_zeros(())
            one = candidate_scores.new_ones(())

            if maj_winner_idx is None:
                batch_losses.append(zero)
                continue

            batch_losses.append(one - candidate_scores[maj_winner_idx])

        if not batch_losses:
            return logits.new_zeros(())

        return torch.stack(batch_losses).mean()
    
    def forward(self, data, model, concatenate=False, device=None):
        """ Takes the logits from our model and the original profile, and computes the majority winner loss.
        Args:
            
        """

        profiles = data["profile"]  # [B, n_voters, n_cands]
        # n_cands = profiles.shape[2]

        # Unpack data from dataloader (this is only used for rule training)
        if concatenate == True:
            X_train = get_concatenated_data(data)
        else:
            # Build model input; model may expect [B, C, V] or concatenated features.
            X_train = data[model.transform_name]

        logits = model(X_train)

        if self.verbose:

            print(f"{logits.shape=}, {profiles.shape=}")

        return self.majority_winner_continuous(logits, profiles)
