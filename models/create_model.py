import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
from .architectures import (
    NN_Transformed,
    NN_Efficient_Transform,
    NN_Transformed_LTE,
    NN_Super
)
from rules import WhalrusModule


def create_model(params:dict):
    print(f"Using {params.get('network_type', 'NOT SPECIFIED')} network")
    if params.get('network_type', None) == 'NN_Efficient_Transform':
        model = NN_Efficient_Transform(n_cands=params['n_cands'], info=params)
    elif params.get('network_type', None) == 'NN_Super':
        model = NN_Super(n_cands=params['n_cands'], info=params)
    else:
        print("Defaulting to NN_Transformed_LTE archecture as no network_type was specified")
        model = NN_Transformed_LTE(n_cands=params['n_cands'], info=params)

        # Use Xavier weight initialization for more stable gradient
        if params.get("use_xavier_weights", False):
            model.apply(init_weights)
    
    # print(f"Model Summary for NN_Super (n_cands={params.get('n_cands')}):")
    # summary(model, input_size=(params.get("batch_size"), model.input_size), device="cuda:0" if params.get("device", "cpu") == "gpu" else "cpu")

    return model


def init_weights(m):
    """Xavier/Glorot initialization for better weight initialization"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)