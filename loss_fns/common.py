import re
import torch
from os import cpu_count

def callable_to_str(callable):
    name = str(callable).split(' ')
    if len(name) == 1:
        return re.sub('\[\]\(\)', '',name[0]).replace('()','')
    else:
        return re.sub('\[\]\(\)', '',name[1]).replace('()','')

def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if device == 'cuda':
        num_cores = torch.cuda.device_count()
    elif device == 'cpu':
        num_cores = cpu_count()
        
    return device, num_cores