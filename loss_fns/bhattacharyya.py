import torch
from torch import nn

def bhattacharyya_loss(p,q):
    bd = BhattacharyyaLoss()
    return bd(p,q)

class BhattacharyyaLoss(torch.nn.Module):
    '''
    Computes the Bhattacharyya distance between two discrete probability distributions
    '''
    def __init__(self):
        super(BhattacharyyaLoss, self).__init__()

    def forward(self, logits, target_dist):
        total = 0
        for idx,_ in enumerate(logits):
            total += torch.sqrt(logits[idx] * target_dist[idx])
        return -torch.log(total)