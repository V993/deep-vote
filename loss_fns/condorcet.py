
"""
Condorcet winner:
condorcet_winner = True
Over all candidates c:
	over all candidates d:
		if P(c, d) <= N /2:
			condorcet_winner = False
			break
Condorcet loser:
condorcet_loser = True
Over all candidates c:
	over all candidates d:
		if P(d, c) <= N /2:
			condorcet_loser = False
			break


For condorcet: P(d, c) == weighted_tournament[d, c]

"""

import torch
from itertools import combinations
from transforms.weighted_tournament import weighted_tournament

class CondorcetLoss(torch.nn.Module):
    def __init__(self, verbose=False):
        self.verbose = verbose
        super(CondorcetLoss, self).__init__()
    
    def condorcet_winner(profile, weighted_tournament):
        _, n_cands = profile.shape # zero indexed
        condorcet_winner = 1.0

        half_cands = n_cands // 2
        
        # Get all possible candidate combinations:
        for c, d in combinations(range(n_cands), range(n_cands)):
            # Count how many voters prefer c over d:
            P_cd = weighted_tournament[c, d]

            if P_cd <= half_cands:
                condorcet_winner = 0.0
                break
        
        return condorcet_winner
    
    def forward(self, profile, weighted_tournament):
        """Takes original distribution and abstain distribution (ordered by preference) and outputs max of the differences

        Args:
            profile (tensor or np.ndarray): profile over which we compute loss
            model (nn.Module): model being trained
        
        Returns:
            tensor: loss over the profile
        """

        is_condorcet_winner = self.condorcet_winner(profile, weighted_tournament)
        return torch.tensor([is_condorcet_winner], device=profile.device)
