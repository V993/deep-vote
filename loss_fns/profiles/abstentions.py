import numpy as np
import torch

# def all_abstentions_gen(profile:np.ndarray):
#     for v,_ in enumerate(profile):
#         yield np.delete(profile, v)

def remove_row(tensor, index):
    # Validate the index
    if index < 0 or index >= tensor.size(0):
        raise ValueError("Index is out of bounds")

    # Concatenate slices before and after the specified index
    if index == 0:
        return tensor[1:]  # If the first row is to be removed
    elif index == tensor.size(0) - 1:
        return tensor[:-1]  # If the last row is to be removed
    else:
        return torch.cat((tensor[:index], tensor[index+1:]), dim=0)

def all_abstentions(profile:torch.tensor, transform=None, no_transform=True)->dict:
    abstaining_profiles = {}
    for i in range(len(profile)):
        profile_clone = profile.clone()

        # for raw profiles (no transform), replace with garbage vote
        if transform is None:# or no_transform is True:
            profile_clone = torch.cat((profile_clone[:i], profile_clone[i+1:]), dim=0) #torch.tensor([-1*len(profile[i])])
            transform = lambda x: x
        # For transforms, remove the abstaining voter
        else:
            profile_clone = remove_row(profile_clone, i)

        abstaining_profiles[i] = transform(profile_clone)
    return abstaining_profiles

def all_abstentions_generator(profile:torch.tensor, transform=None, no_transform=True):
    for i in range(len(profile)):
        profile_clone = profile.clone()

        # for raw profiles (no transform), replace with garbage vote
        if transform is None:# or no_transform is True:
            profile_clone = torch.cat((profile_clone[:i], profile_clone[i+1:]), dim=0)
            transform = lambda x: x
        # For transforms, remove the abstaining voter
        else:
            profile_clone = remove_row(profile_clone, i)

        yield transform(profile_clone)