import torch
from torch import nn
import numpy as np

# def overlap_np(p:np.ndarray,q:np.ndarray):
#     return np.sum([min(i,j) for i,j in zip(p,q)])

def overlap_loss(p,q):
    p = torch.tensor(p)
    q = torch.tensor(q)
    ol = Overlap_Loss()
    return ol(p,q)

class Overlap_Loss(torch.nn.Module):
    '''
    Computes one minus the overlap between two discrete probability distributions
    '''
    def __init__(self):
        super(Overlap_Loss, self).__init__()

    def forward(self, logits:torch.Tensor, target_dist:torch.Tensor):
        total = 0
        for idx,_ in enumerate(logits):
            total += torch.tensor.min(logits[idx], target_dist[idx])
        return 1.0 - torch.sum(total)
        # return 1.0-total