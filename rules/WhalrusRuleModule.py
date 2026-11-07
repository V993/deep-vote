import torch
import torch.nn as nn
import numpy as np

class WhalrusModule(nn.Module):
    """
    Run a Whalrus rule as a module. Requires loading data that includes the full preference profile.
    """
    def __init__(self, rule):
        super(WhalrusModule, self).__init__()

        self.rule = rule

    def forward(self, X):

        # Accept either a tensor batch or an iterable of tensors/arrays
        if isinstance(X, torch.Tensor):
            batch = X
            device = X.device
        else:
            batch = list(X)
            # pick device from first tensor-like entry, default to CPU
            if len(batch) and isinstance(batch[0], torch.Tensor):
                device = batch[0].device
            else:
                device = torch.device("cpu")

        all_out = []
        for profile in batch:
            if isinstance(profile, torch.Tensor):
                profile_np = profile.detach().cpu().numpy()
            else:
                profile_np = np.asarray(profile)
            # Whalrus rules operate on numpy arrays, so safely detach and move to CPU
            out = self.rule(profile_np)
            all_out.append(out)
        out = torch.stack([torch.as_tensor(ao, dtype=torch.float32, device=device) for ao in all_out])
        return out
