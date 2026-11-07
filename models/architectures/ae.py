#Create an autoencoder for learning transforms that are good for learning rules and axioms
#Input size: n_voters x n_cands
#Transform size: n_cands x n_cands (output of encoder, input of decoder)

import torch
from torch import nn
import time

class NN_AE(nn.Module):
    def __init__(self, n_voters:int, n_cands:int, info:dict=dict()):
        super(NN_AE, self).__init__()
        
        self.n_voters = n_voters
        self.n_cands = n_cands
        self.input_size = n_voters * n_cands
        self.output_size = n_voters * n_cands
        self.transform_size = n_cands * n_cands

        self.info = info
        self.info['model_id'] = time.strftime("%Y%m%d-%H%M%S")
        self.info['n_voters'] = self.n_voters
        self.info['n_cands'] = self.n_cands
        self.info['input_size'] = self.input_size
        self.info['transform_size'] = self.transform_size
        self.info['activations'] = 'ReLU'
        self.info['output_activation'] = 'Softmax'
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.transform_size),
            nn.ReLU(),
            nn.Linear(self.transform_size, self.transform_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.transform_size, self.transform_size),
            nn.ReLU(),
            nn.Linear(self.transform_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, self.input_size),
            nn.Sigmoid()
        )
        
    def forward(self, X:torch.tensor):
        X = torch.flatten(X.float()).unsqueeze(dim=0)
        transformedX = self.encoder(X)
        return self.decoder(transformedX)
        
    def get_info(self):
        return self.info