#Create a shallow neural net for learning voting rules using rank ballots
#Input size: n_cands x n_cands
#To be used with tournament, weighted tournament, and rank frequency transforms

import time
import torch
from torch import nn
import torch.nn.functional as F

# Helper function to compute the sizes of the layers in a funnel-shaped neural network
def compute_funnel_layer_sizes(input_size, output_size, hidden_layers, bottleneck_ratio):
    initial_width = input_size * 3
    final_hidden = max(output_size + 32, int(initial_width * bottleneck_ratio))
    sizes = []

    for i in range(1, hidden_layers + 1):
        interpolated = initial_width - i * ((initial_width - final_hidden) / (hidden_layers + 1))
        sizes.append(int(interpolated))

    return sizes

# Torch does not have a built-in temperature-scaled softmax, so we define it here:
class SoftmaxWithTemperature(nn.Module):
    """
    Applies softmax with temperature scaling.
    A higher temperature makes the distribution smoother (more evenly spread), 
    A lower temperature makes it sharper (more confident)

    Args:
        input_tensor: A tensor of raw scores.
        temperature: A scalar temperature value (non-negative).

    Returns:
        A tensor of probabilities after applying softmax with temperature.
    """
    def __init__(self, temperature):
        super().__init__()
        assert temperature > 0, "Temperature must be greater than 0."
        self.temperature = temperature

    def forward(self, input_tensor):
        return F.softmax(input_tensor / self.temperature, dim=1)

class NN_Transformed_LTE(nn.Module):
    """
    Primary network used for most of our experiments. Emulates the Learning to Elect network.

    Composed of a wider embedding layer and then decomposing into 4 layers with a width of 120. 
    This is a very basic Feed Forward Network, no frills. 
    """
    def __init__(self, n_cands, info:dict=dict()):
        super(NN_Transformed_LTE, self).__init__()
        
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
        self.transform = self.info.get("transform", None)
        self.transform_name = self.info.get("transform_name", "") # Save the transform name of the model for easy access later
        self.output_layer = self.info.get("output_layer", "")

        if "log" in self.output_layer:
            output_layer = nn.LogSoftmax(dim=1)
        elif "temperature" in info:
            temperature = info.get("temperature", 1.0)
            output_layer = SoftmaxWithTemperature(temperature)
        else:
            output_layer = nn.Softmax(dim=1)

        if info.get("activation_fn") == 'TanH':
            self.activation_fn = nn.Tanh()
        else:
            self.activation_fn = nn.ReLU()

        dropout_rate = info.get("dropout_rate", 0)
        use_batch_norm = info.get("use_batch_norm", False)
        use_layer_norm = info.get("use_layer_norm", False)

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_size,
                      out_features=self.input_size,
                      dtype=torch.float),
            nn.BatchNorm1d(self.input_size) if use_batch_norm else nn.Identity(self.input_size),
            nn.LayerNorm(self.input_size) if use_layer_norm else nn.Identity(self.input_size),
            self.activation_fn,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=self.input_size,
                      out_features=200,
                      dtype=torch.float),
            nn.BatchNorm1d(200) if use_batch_norm else nn.Identity(200),
            nn.LayerNorm(200) if use_layer_norm else nn.Identity(120),
            self.activation_fn,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=200,
                      out_features=120,
                      dtype=torch.float),
            nn.BatchNorm1d(120) if use_batch_norm else nn.Identity(120),
            nn.LayerNorm(120) if use_layer_norm else nn.Identity(120),
            self.activation_fn,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=120,
                      out_features=120,
                      dtype=torch.float),
            nn.BatchNorm1d(120) if use_batch_norm else nn.Identity(120),
            nn.LayerNorm(120) if use_layer_norm else nn.Identity(120),
            self.activation_fn,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=120,
                      out_features=120,
                      dtype=torch.float),
            nn.BatchNorm1d(120) if use_batch_norm else nn.Identity(120),
            nn.LayerNorm(120) if use_layer_norm else nn.Identity(120),
            self.activation_fn,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=120,
                      out_features=self.output_size,
                      dtype=torch.float),
            output_layer  # Your softmax
        )

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
                raise ValueError(f"Expected input shape {expected_shape}, but got {X.shape}. Type: {type(X)}")
    
        return self.layer_stack(X)
    
    # def forward(self, X: torch.Tensor):
    #     X = X.float()

    #     if X.dim() == 2 and X.shape == (self.n_cands, self.n_cands): # Handle batches and single instances:
    #         X = X.unsqueeze(0)
        
    #     return self.layer_stack(X)

# Deprecated:   
class NN_Efficient_Transform(nn.Module):
    def __init__(self, n_cands:int, info:dict=dict()):
        super(NN_Efficient_Transform, self).__init__()
        
        self.n_cands = n_cands
        self.input_size = ( (n_cands * n_cands) - n_cands ) // 2
        self.output_size = n_cands

        self.info = info
        self.info['model_id'] = time.strftime("%Y%m%d-%H%M%S")
        self.info['n_cands'] = self.n_cands
        self.info['input_size'] = self.input_size
        self.info['output_size'] = self.output_size
        self.info['activation_fn'] = info.get("activation_fn", 'ReLU')
        self.info['efficient_transform'] = True
        
        self.transform = self.info.get("transform", None)
        self.output_layer = self.info.get("output_layer", "")

        if "log" in self.output_layer:
            output_layer = nn.LogSoftmax(dim=1)
        else:
            output_layer = nn.Softmax(dim=1)

        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_size,
                      out_features=120,dtype=torch.float),
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
        X = X.float()

        if X.dim() == 2 and X.shape == (self.n_cands, self.n_cands): # Handle batches and single instances:
            X = X.unsqueeze(0)
        
        return self.layer_stack(X)
    
# Deprecated:
class NN_Transformed(nn.Module):
    def __init__(self, n_cands:int, info:dict=dict()):
        super(NN_Transformed, self).__init__()
        
        self.n_cands = n_cands
        self.input_size = n_cands * n_cands
        self.output_size = n_cands

        self.info = info
        self.info['model_id'] = time.strftime("%Y%m%d-%H%M%S")
        self.info['n_cands'] = self.n_cands
        self.info['input_size'] = self.input_size
        self.info['output_size'] = self.output_size
        self.info['hidden_layers'] = 2
        self.info['activation_fn'] = info.get("activation_fn", 'ReLU')
        
        self.transform = self.info.get("transform", None)
        self.output_layer = self.info.get("output_layer", "")

        if "log" in self.output_layer:
            output_layer = nn.LogSoftmax(dim=1)
        else:
            output_layer = nn.Softmax(dim=1)
                            
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.input_size,
                                      out_features=self.input_size,
                                      dtype=torch.float),
            nn.ReLU(),
            nn.Linear(in_features=self.input_size,
                      out_features=self.output_size,dtype=torch.float),
            output_layer,
            nn.Flatten()
        )

    def get_info(self):
        return self.info
    
    def forward(self, X: torch.Tensor):
        X = X.float()
        # Flatten each image in the batch separately, not the whole batch at once
        X = X.view(X.size(0), -1)  # Reshape from [batch_size, n_cands, n_cands] to [batch_size, n_cands^2]
        return self.layer_stack(X)

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

        print("Using NN_Super architecture with the following parameters:")
        print(f"  input_size: {self.input_size}")
        print(f"  output_size: {self.output_size}")
        print(f"  hidden_layers: {info.get('hidden_layers', 2)}")
        print(f"  activation_fn: {self.info.get('activation_fn', 'ReLU')}")

        self.hidden_layers = info.get("hidden_layers", 2) # the number of hidden layers to use
        self.bottleneck_ratio = info.get("bottleneck_ratio", 0.1) # fraction of input_size*3 to use as the narrowest hidden layer.

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
            bottleneck_ratio=self.bottleneck_ratio
        )

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