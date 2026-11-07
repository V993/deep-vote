#Create a shallow neural net for learning voting rules using rank ballots
#Input size: n_voters x n_cands
#To be used without transform and with lexicographic transform (for anonymization)

import torch
from torch import nn
import time

class NN_Full(nn.Module):
    def __init__(self, n_voters, n_cands:int, hidden_multiplier:int = 4, info:dict=dict()):
        super(NN_Full, self).__init__()
        
        self.n_voters = n_voters
        self.n_cands = n_cands
        self.input_size = n_voters * n_cands
        self.output_size = n_cands
        self.hidden_multiplier = hidden_multiplier

        self.info = info
        self.info['model_id'] = time.strftime("%Y%m%d-%H%M%S")
        self.info['n_voters'] = self.n_voters
        self.info['n_cands'] = self.n_cands
        self.info['input_size'] = self.input_size
        self.info['output_size'] = self.output_size
        self.info['hidden_layers'] = 2
        self.info['hidden_multiplier'] = self.hidden_multiplier
        self.info['activations'] = 'ReLU'
        self.info['output_activation'] = 'Softmax'
        self.transform = self.info.get("transform", None)
        
        self.layer_stack = (nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size,
                                      self.hidden_multiplier*self.input_size,
                                      dtype=torch.float),
            nn.ReLU(),
            nn.Linear(self.hidden_multiplier*self.input_size,
                      self.hidden_multiplier*self.input_size,dtype=torch.float),
            nn.ReLU(),
            nn.Linear(self.hidden_multiplier*self.input_size,
                      self.output_size,dtype=torch.float),
            nn.Softmax(dim=1))
        )
        
    def get_info(self):
        return self.info
        
    def forward(self, X:torch.Tensor):
        X = X.float()
        X = torch.flatten(X).unsqueeze(dim=0)
        return self.layer_stack(X)


class NN_Full_LTE(nn.Module):
    def __init__(self, n_voters:int, n_cands:int, info:dict=dict()):
        super(NN_Full_LTE, self).__init__()
        
        self.n_cands = n_cands
        self.n_voters = n_voters
        self.input_size = n_voters * n_cands
        self.output_size = n_cands

        self.info = info
        self.info['model_id'] = time.strftime("%Y%m%d-%H%M%S")
        self.info['n_cands'] = self.n_cands
        self.info['input_size'] = self.input_size
        self.info['output_size'] = self.output_size
        self.info['activations'] = 'ReLU'
        self.transform = self.info.get("transform", None)
        self.output_layer = self.info.get("output_layer", "")

        if "log" in self.output_layer:
            output_layer = nn.LogSoftmax(dim=1)
        else:
            output_layer = nn.Softmax(dim=1)

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_size,
                                      out_features=120,
                                      dtype=torch.float),
            nn.ReLU(),
            nn.Linear(in_features=120,
                      out_features=120,dtype=torch.float),
            nn.ReLU(),
            nn.Linear(in_features=120,
                      out_features=120,dtype=torch.float),
            nn.ReLU(),
            nn.Linear(in_features=120,
                      out_features=120,dtype=torch.float),
            nn.ReLU(),
            nn.Linear(in_features=120,
                      out_features=self.output_size,dtype=torch.float),
            output_layer,
            nn.Flatten()
        )

    def get_info(self):
        return self.info
    
    def forward(self, X: torch.Tensor):

        X = torch.flatten(X).view(1, -1)
        # Flatten each image in the batch separately, not the whole batch at once
        # X = X.view(X.size(0), -1)  # Reshape from [batch_size, n_cands, n_cands] to [batch_size, n_cands^2]
        return self.layer_stack(X)
    