import numpy
import torch
import networkx as nx
from rules import to_indices
from transforms.tournament import tournament
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


class TopCycleLoss(torch.nn.Module):
    '''
    Return the amount of probability mass placed on cands not in the top cycle
    '''
    def __init__(self):
        super(TopCycleLoss, self).__init__()

    def forward(self, data, model, concatenate=False, device=None):
        """_summary_

        Args:
            dist (np.array): output from the model
            topcycle_indicator (np.array): 1 for candidates in topcycle, otherwise 0

        Returns:
            tensor: loss over the profile
        """
        evaluating_whalrus = isinstance(model, WhalrusModule)
        # dist, topcycle_indicator = dist.clone().requires_grad_(True), topcycle_indicator.detach()
        profile = data["Profile"].float().to(device)
        transform_key = _resolve_transform_key(concatenate, data)
        if transform_key is None:
            orig_parts = [
                data['tournament'],
                data['weighted_tournament-normalized'],
                data['rank_matrix-normalized'],
            ]
            orig_parts = [p.float().to(device) if p.dim() > 1 else p.unsqueeze(0) for p in orig_parts]
            t_original_profile = torch.cat(orig_parts, dim=1)
        else:
            t_original_profile = data[transform_key].float().to(device)
        if not evaluating_whalrus:
            orig_dist = model(t_original_profile)
        else:
            orig_dist = model(profile)

        # tournament = data["tournament"]
        topcycle_indicator = get_topcycle(profile, data["tournament"], device=device)

        # # Probability mass on correct classes
        # correct_prob = torch.sum(orig_dist * topcycle_indicator, dim=1).mean()

        # Probability mass on incorrect classes
        incorrect_prob = torch.sum(orig_dist * (1 - topcycle_indicator), dim=1).mean()

        # print(f"Amount of CORRECT probability mass: {correct_prob}")
        # print(f"Amount of INCORRECT probability mass: {incorrect_prob}")

        # l = torch.max(incorrect_prob - correct_prob, torch.tensor(0.0, device=device))
        l = incorrect_prob
        return l


# def get_topcycle(profile:torch.tensor, tournament:torch.tensor)->torch.tensor:
#     '''
#     Takes in a full profile of preference orders and returns an indicator array
#     for which cands are in the top cycle
#
#     NOTES:
#     Edges in graph are reversed so loser points to winner because we want the source SCC in the tournament graph, but the attracting_components() function from networkx gives only the sink SCCs. For tournament graphs there is only ever one source SCC, so we take the first (and only) set that is produced by the generator.
#     '''
#
#     # T = tournament(profile)
#     _, n_cands = profile.shape
#
#     G = nx.DiGraph()
#     for c1 in range(n_cands):
#         for c2 in range(n_cands):
#             if tournament[c1,c2] > 0.5:     # careful; older data versions use 1/0/-1 and new versions use 1/0.5/0
#                 G.add_edge(c2, c1)
#     topcycle = list(nx.attracting_components(G))[0]
#     return to_indices(topcycle, range(n_cands))


def get_topcycle(profile: torch.tensor, tournament: torch.tensor, device=None) -> torch.tensor:
    '''
    Takes in a full profile of preference orders and returns an indicator array
    for which cands are in the top cycle

    Args:
        profile: tensor of shape (n_voters, n_cands) or (batch_size, n_voters, n_cands)
        tournament: tensor of shape (n_cands, n_cands) or (batch_size, n_cands, n_cands)

    Returns:
        tensor of shape (n_cands,) or (batch_size, n_cands) with top cycle indicators

    NOTES:
    Edges in graph are reversed so loser points to winner because we want the source SCC
    in the tournament graph, but the attracting_components() function from networkx gives
    only the sink SCCs. For tournament graphs there is only ever one source SCC, so we
    take the first (and only) set that is produced by the generator.
    '''

    if device is None:
        device = tournament.device

    tournament_cpu = tournament.detach().cpu()

    # Check if batched (3D) or unbatched (2D)
    if profile.dim() == 2:
        # Unbatched case
        _, n_cands = profile.shape
        indicator = _compute_topcycle_single(tournament_cpu, n_cands)
        return torch.tensor(indicator, dtype=torch.float32, device=device)
    else:
        # Batched case
        batch_size, _, n_cands = profile.shape

        # Process all batch items (still sequential, but with less overhead)
        results = []
        for b in range(batch_size):
            indicator = _compute_topcycle_single(tournament_cpu[b], n_cands)
            results.append(indicator)
        results = numpy.array(results)

        return torch.tensor(results, dtype=torch.float32, device=device)


def _compute_topcycle_single(tournament_matrix: torch.tensor, n_cands: int):
    '''
    Helper function to compute top cycle for a single tournament matrix.
    Returns a numpy array that can be efficiently converted to tensor.
    '''
    G = nx.DiGraph()

    # Build graph from tournament matrix
    for c1 in range(n_cands):
        for c2 in range(n_cands):
            if tournament_matrix[c1, c2] >= 0.5:
                G.add_edge(c2, c1)

    topcycle = list(nx.attracting_components(G))[0]

    # Return numpy array directly for efficient batching
    return to_indices(topcycle, range(n_cands))
