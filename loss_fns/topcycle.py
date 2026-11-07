
import torch
import numpy as np
import networkx as nx
from rules import to_indices
from transforms.tournament import tournament

# def get_topcycle(profile:np.ndarray)->np.ndarray:
#     '''
#     Takes in a full profile of preference orders and returns an indicator array
#     for which cands are in the top cycle
#
#     NOTES:
#     Edges in graph are reversed so loser points to winner because we want the source SCC in the tournament graph, but the attracting_components() function from networkx gives only the sink SCCs. For tournament graphs there is only ever one source SCC, so we take the first (and only) set that is produced by the generator.
#     '''
#
#     T = tournament(profile)
#     _, n_cands = profile.shape
#
#     G = nx.DiGraph()
#     for c1 in range(n_cands):
#         for c2 in range(n_cands):
#             if T[c1,c2] > 0:
#                 G.add_edge(c2, c1)
#     topcycle = list(nx.attracting_components(G))[0]
#     return to_indices(topcycle, range(n_cands))


def get_topcycle(profile:torch.tensor)->torch.tensor:
    '''
    Takes in a full profile of preference orders and returns an indicator array
    for which cands are in the top cycle

    NOTES:
    Edges in graph are reversed so loser points to winner because we want the source SCC in the tournament graph, but the attracting_components() function from networkx gives only the sink SCCs. For tournament graphs there is only ever one source SCC, so we take the first (and only) set that is produced by the generator.
    '''

    T = tournament(profile)
    _, n_cands = profile.shape

    G = nx.DiGraph()
    for c1 in range(n_cands):
        for c2 in range(n_cands):
            if T[c1,c2] > 0:
                G.add_edge(c2, c1)
    topcycle = list(nx.attracting_components(G))[0]
    return to_indices(topcycle, range(n_cands))