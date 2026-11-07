import torch
import torch.nn as nn
import torch.nn.functional as F

class SymKLDivLoss(nn.Module):
    """
    Symmetric KL Divergence
    
    d(p, q) = KL(p || q) + KL(q || p)

    https://arxiv.org/pdf/2401.11381

    Handles the log expectations of kldiv, no inputs need be logged

    We sadly need to clamp the input vectors as they have many flat zeroes, which will fail.

    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.min_val = 1e-8

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, min=self.min_val)
        q = torch.clamp(q, min=self.min_val)

        return F.kl_div(p.log(), q, reduction=self.reduction) + F.kl_div(q.log(), p, reduction=self.reduction)
    
class JensonShannonLoss(nn.Module):
    """ 
    This is the jensonshannon DIVERGENCE, not the DISTANCE. 
    The distance is the square root of the divergence, and as a result is non-differentiable. 
    It is easier for the model to learn from the divergence, like KLDiv.
    If we want the distance, we can take the square root. However, when the KLDiv is negative it will be a nan.

    d(p, q) = 0.5 * KL(p || m) + KL(q || m)

    m = 0.5 * (p + q)

    https://arxiv.org/pdf/2401.11381

    Handles the log expectations of kldiv, no inputs need be logged

    We sadly need to clamp the input vectors as they have many flat zeroes, which will fail.

    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.min_val = 1e-8

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, min=self.min_val)
        q = torch.clamp(q, min=self.min_val)
        m = 0.5 * (p + q)

        return 0.5 * (F.kl_div(p.log(), m, reduction=self.reduction) + F.kl_div(q.log(), m, reduction=self.reduction))