import torch
from torch import nn
import time

class NN_NoHidden(nn.Module):
    '''
    To see if we can learn certain rules from transforms without a hidden layer at all.
    Should be able to learn any scoring rule from the rank frequency matrix and Copeland from the weighted tournament matrix.
    '''
    def __init__(self, n_cands:int, n_voters:int=None, info:dict=dict()):
        super(NN_NoHidden, self).__init__()
        
        self.n_cands = n_cands
        self.n_voters = n_voters
        if self.n_voters is None:
            self.input_size = n_cands * n_cands
        else:
            self.input_size = n_voters * n_cands
        self.output_size = n_cands

        self.info = info
        self.info['model_id'] = time.strftime("%Y%m%d-%H%M%S")
        self.info['n_cands'] = self.n_cands
        self.info['n_voters'] = self.n_voters
        self.info['input_size'] = self.input_size
        self.info['output_size'] = self.output_size
        self.info['hidden_layers'] = 0
        self.info['activations'] = 'ReLU'
        self.info['output_activation'] = 'Softmax'
        self.transform = self.info.get("transform", None)
                            
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.DataParallel(nn.Linear(in_features=self.input_size,
                                      out_features=self.output_size,
                                      dtype=torch.float)),
            nn.Softmax(dim=1),
            nn.Flatten()
        )

    def get_info(self):
        return self.info
        
    def forward(self, X:torch.Tensor):
        X = X.float()
        X = torch.flatten(X).unsqueeze(dim=0)
        return self.layer_stack(X)