
import re
import torch
from transforms.rank_freq import rankfreq
from transforms.tournament import w_to_tournament
from transforms.weighted_tournament import w_tournament_after_swap, w_tournament_after_abstention, weighted_tournament

def callable_to_str(callable):
    name = str(callable).split(' ')
    if len(name) == 1:
        return re.sub('\[\]\(\)', '',name[0]).replace('()','')
    else:
        return re.sub('\[\]\(\)', '',name[1]).replace('()','')

def transform_alts_participation(profile, d_alts, transform):
    
    if transform is None:
        t = "concatenate"
    else:
        t = callable_to_str(transform).lower().strip()

    if t == 'concatenate':

        # Get the original profile's concatenations:
        original_wt = weighted_tournament(profile)
        original_t = w_to_tournament(original_wt, len(profile))
        original_r = rankfreq(profile)

        # Concatenate the original profile's transforms:
        t_original_profile = torch.Tensor([]).float().to(profile.device)
        for transform in [original_t, original_r, original_wt]: # Same order as trained models
            t_original_profile = torch.cat((t_original_profile, transform), dim=1) # Assumes shape = (batch_size, n_cands, n_cands)
    
        # Get the alternate profile's concatenations and coagulate
        wt_alts = {}
        wt = weighted_tournament(profile)
        for voter in d_alts.keys():
            wt_alts[voter] = w_tournament_after_abstention(wt, profile[voter], len(profile), True)
            
        t_alts = {key: w_to_tournament(v, len(profile)) for key, v in wt_alts.items()}
        r_alts = {k: rankfreq(alt) for k, alt in d_alts.items()}

        # Concatenate each alternate's transforms:
        transformed_alts = {}
        for voter in d_alts.keys():
            concatenated = torch.cat(
                [t_alts[voter], r_alts[voter], wt_alts[voter]], dim=1
            )  # assumes all shapes (n_cands, n_cands)
            transformed_alts[voter] = concatenated

    elif t == 'weighted_tournament' or t == 'tournament':
        t_original_profile = transform(profile)
        wt_alts = {}
        wt = weighted_tournament(profile)
        for voter in d_alts.keys():
            wt_alts[voter] = w_tournament_after_abstention(wt, profile[voter], len(profile), True)
        if t == 'tournament':
            transformed_alts = {key: w_to_tournament(v, len(profile)) for key, v in wt_alts.items()}
        else:
            transformed_alts = wt_alts

    elif t == 'rankfreq':
        t_original_profile = transform(profile)
        transformed_alts = {k: rankfreq(alt) for k, alt in d_alts.items()}

    else:  # t == 'full' or t == None
        print(f"UNEXPECTED TRANSFORM {t}, RETURNING ORIGINAL PROFILES")
        t_original_profile = profile
        transformed_alts = d_alts

    return t_original_profile, transformed_alts



def transform_alts(profile, d_alts, transform=None):

    if transform == None:
        transform = lambda x: x
        t = None
    else:
        t = callable_to_str(transform).lower()

    t_original_profile = transform(profile)

    # Apply the tournament transform to all the alternative profiles
    transformed_alts = {}
    if t == 'weighted_tournament' or t == 'tournament':
        wt_alts = {}
        wt = weighted_tournament(profile)
        for key in d_alts.keys():
            voter = key[0]
            cand_down_id, cand_up_id = key[1], key[1]+1
            cand_down = profile[voter, cand_down_id]
            cand_up = profile[voter, cand_up_id]
            n_voters = len(profile)
            wt_alts[key] = w_tournament_after_swap(wt,int(cand_down), 
                                                   int(cand_up), n_voters)
      
        if t == 'tournament':
            transformed_alts = {key:w_to_tournament(v,n_voters) for key,v in wt_alts.items()}
        else: # t == "weighted_tournament":
            transformed_alts = wt_alts
 

    elif t == 'rankfreq':
        transformed_alts = {k:rankfreq(alt) for k,alt in d_alts.items()}
    else: #t == 'full' or t == None
        transformed_alts = d_alts


    return t_original_profile, transformed_alts