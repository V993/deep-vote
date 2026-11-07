import time
import torch
from torch import nn
import torch.nn.functional as F

# Helper function to compute the sizes of the layers in a funnel-shaped neural network
def compute_funnel_layer_sizes(input_size, output_size, hidden_layers, explosion_factor=2, power=0.5):
    start_layer = input_size * explosion_factor
    layers = [input_size]
    for i in range(hidden_layers + 1):
        t = i / hidden_layers
        size = start_layer * (1 - t**power) + output_size * t**power
        layers.append(int(size))
    return layers[1:-1]


class NN_Super(nn.Module):
    """
    A much larger and more resource intensive version of the NN_Transformed_LTE network.

    Unlike the LTE architecture, this artitecture strives for complexity. We use a funnel shaped FFN with parameters to control shape:
        - The first hidden layer explodes the dimensionality of the input with a width of 'input_size * 3'
        - Each subsequent hidden layer gradually decreases in width by a factor of the initial width and the target width
        - The final hidden layer narrows to a soft bottleneck, sized using 'bottleneck_ratio'. This was eyeballed to reduce further informational bottleneck
    """
    def __init__(self, n_cands, info:dict=dict()):
        super(NN_Super, self).__init__()
        
        self.n_cands = n_cands
        self.output_size = n_cands
        self.input_size = n_cands * n_cands

        if info.get("concatenate", False) == True:
            self.input_size *= 3 # For all 3 transforms
            print("Concatenating input")

        self.info = info
        self.info['model_id'] = time.strftime("%Y%m%d-%H%M%S")
        self.info['n_cands'] = self.n_cands
        self.info['input_size'] = self.input_size
        self.info['output_size'] = self.output_size
        self.info['activation_fn'] = info.get("activation_fn", 'ReLU')

        self.hidden_layers = info.get("hidden_layers", 2) # the number of hidden layers to use
        self.explosion_factor = info.get("explosion_factor", 2) # scalar multiplier for the ratio between the input size and the first hidden layer (which will decay afterwards)

        self.transform = self.info.get("transform", None)
        self.output_layer = self.info.get("output_layer", "")

        if "log" in self.output_layer:
            output_layer = nn.LogSoftmax(dim=1)
        else:
            output_layer = nn.Softmax(dim=1)

        if self.info.get("activation_fn") == 'TanH':
            activation_fn = nn.Tanh()
        else:
            activation_fn = nn.ReLU()

        hidden_sizes = compute_funnel_layer_sizes(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers,
            explosion_factor=self.explosion_factor
        )

        print("Using NN_Super architecture with the following parameters:")
        print(f"  input_size: {self.input_size}")
        print(f"  output_size: {self.output_size}")
        print(f"  hidden_layers: {info.get('hidden_layers', 2)}")
        print(f"  hidden_sizes:")
        for layer in hidden_sizes:
            print(f"\t\t{layer}")
        
        print(f"\n  activation_fn: {self.info.get('activation_fn', 'ReLU')}")

        layers = []

        # Input layer
        layers.extend([
            nn.Flatten(),
            nn.Linear(in_features=self.input_size,
                    out_features=hidden_sizes[0],
                    dtype=torch.float),
            activation_fn
        ])

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.extend([
                nn.Linear(in_features=hidden_sizes[i - 1],
                        out_features=hidden_sizes[i],
                        dtype=torch.float),
                activation_fn
            ])

        # Output layer
        layers.extend([
            nn.Linear(in_features=hidden_sizes[-1],
                    out_features=self.output_size,
                    dtype=torch.float),
            output_layer,
            nn.Flatten()
        ])

        self.layer_stack = nn.Sequential(*layers)

    def get_info(self):
        return self.info
    
    def forward(self, X: torch.Tensor):
        X = X.float()

        if X.dim() == 2:
            if self.info.get("concatenate", False):
                expected_shape = (self.n_cands, self.n_cands * 3)
            else:
                expected_shape = (self.n_cands, self.n_cands)
            
            if X.shape == expected_shape:
                X = X.unsqueeze(0)
            else:
                raise ValueError(f"Expected input shape {expected_shape}, but got {X.shape}")
    
        return self.layer_stack(X)