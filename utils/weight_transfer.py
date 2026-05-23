import os, re
from collections import OrderedDict
from urllib.request import urlretrieve
from compressai.zoo.image import model_urls
from utils import common_function as common_utils
import torch
import numpy as np

MODELS_PRETRAINED = {
    'factorizedPrior': 'bmshj2018-factorized',
    'scaleHyperprior': 'bmshj2018-hyperprior',
    'meanScaleHyperprior': 'mbt2018-mean',
}


def download_pretrained_model(model_name:str, lmbda:float, q:int, foldername_pretrained:str):
    """Get pretrained model from CompressAI

    Args:
        model_name (str): name of the model from [SphereFactoriziedPrior, SphereScaleHyperprior, SphereMeanScaleHyperprior]
        lmbda (float): lambda value
        q (int): quality parameter
        foldername_pretrained (str): path to folder to save the pretrained model
    """
    model_key = model_name.split('Sphere')[-1]
    model_key = model_key[0].lower() + model_key[1:]
    assert model_key in MODELS_PRETRAINED, f"model_key={model_key} is not avaliable in pretrained models"
    fp_pretrained = os.path.join(foldername_pretrained, f"{model_key}_pretrained_lambda_{lmbda:.4f}_q_{q}.pth.tar")
    if not os.path.isfile(fp_pretrained):
        url_pretrained = model_urls[MODELS_PRETRAINED[model_key]]['mse'][q]
        print(f"Downloading pretrained model from {url_pretrained}")
        os.makedirs(foldername_pretrained, exist_ok=True)
        urlretrieve(url_pretrained, fp_pretrained)
    return fp_pretrained

def rearange_weights(weights):
    'Rearange (k,k) conv weights to healpix weights (ch_out,ch_in,k,k)->(k**2,ch_in,ch_out)'
    assert weights.size(-2) == weights.size(-1)
    k = weights.size(-1)
    assert k%2==1, "kernel size must be odd"
    new_weights = weights.flatten(-2,-1)
    new_weights = new_weights[:,:,get_healpix_order(k//2, include_center=True, start_sw=True)]
    new_weights = new_weights.swapaxes(0,2)
    return new_weights

def transform_state_dict(state_dict:dict, single_conv:bool=True):
    """
    Adapt the state dict from a model pretrained on plain images to the corresponding sphere state dict
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        match = re.search(r"^(.*)\.([0-9]+)\.(weight|bias|beta|gamma)(_reparam.)?(pedestal|lower_bound.bound)?", key)
        if match:
            # example: groups = (g_a, 0, bias, None, None)
            model, i, param, suffix1, suffix2 = match.groups()
            new_key = f"{model}.{int(i)//2}."
            if param in ['weight', 'bias']:
                new_key += f"list_conv.0.{param}"
            else:
                new_key += f"activation.{param}"
                if suffix1 is not None:
                    new_key += f"{suffix1}{suffix2}"
            if param in ['weight', 'bias']:
                if param == 'bias':
                    new_state_dict[new_key] = value
                if (param == 'weight') and ('_s' in model) and (model+i != 'h_s4'): # in and out channels swapped
                    value = value.swapaxes(0,1)
                if model+i not in ['h_a0', 'h_s4']: # 5x5 conv add second conv
                    if param == 'weight': # cut out center 3x3 weights for first conv if mapping to double sdpa conv
                        new_state_dict[new_key] = rearange_weights(value if single_conv else value[:,:,1:4,1:4])
                    if not single_conv:
                        new_key = new_key.replace('list_conv.0', 'list_conv.1')
                        if param == 'bias': # take bias for both convs
                            new_state_dict[new_key] = value
                        else: # 5x5 weights for second conv
                            ch_out, _ = value.shape[:2]
                            new_weights = torch.zeros(ch_out, ch_out, 3, 3, device=value.device, dtype=value.dtype)
                            for c in range(ch_out):
                                for y in range(3):
                                    for x in range(3):
                                        value_crop = value[c,:,y:y+3,x:x+3].mean(0)
                                        new_weights[c,c,y,x] = value_crop.mean((-2,-1))
                            new_state_dict[new_key] = rearange_weights(new_weights)
                elif param=='weight': # 3x3 conv only rearange weights
                    new_state_dict[new_key] = rearange_weights(value)
            else: # only rename gamma and beta keys
                new_state_dict[new_key] = value
            continue
        match = re.search(r"(entropy_bottleneck._)(matrices|biases|factors)\.([0-9]*)$", key)
        if match:
            renaming = {'matrices': 'matrix','biases': 'bias','factors': 'factor'}
            prefix, var_name, i = match.groups()
            new_key = f"{prefix}{renaming[var_name]}{i}"
            new_state_dict[new_key] = value
            continue
        # don't change key value pair for any other case
        new_state_dict[key] = value
    return new_state_dict

def get_healpix_order(n_hops:int, include_center:bool=True, start_sw:bool=True):
    n_firstHopNeighbors = 8
    n_Neighbors = common_utils.sumOfAP(a=n_firstHopNeighbors, d=n_firstHopNeighbors, n=n_hops)
    k_size = int(np.sqrt(n_Neighbors+1))
    assert k_size**2-1 == n_Neighbors
    grid2hp = []
    for hop in range(0 if include_center else 1, n_hops+1):
        grid2hp.extend(get_square_ring_indices(k_size, hop, start_sw=start_sw))
    return grid2hp

def get_grid_order(n_hops:int, include_center:bool=True, start_sw:bool=True):
    grid2hp = get_healpix_order(n_hops, include_center, start_sw)
    hp2grid = np.argsort(grid2hp).tolist()
    return hp2grid

def get_square_ring_indices(k:int, radius:int, start_sw:bool=True):
    assert k>1 and k%2==1 and radius<=k//2
    center = (k//2) * (k+1) # center pixel after flattening a (k,k) kernel
    if radius == 0: return [center]
    indices = []
    for i in range(center-radius*(k+1), center-radius*(k-1)+1): # go right on the top side
        indices.append(i)
    for i in range(indices[-1]+k, indices[-1]+k*(2*radius+1), k): # go down on the right side
        indices.append(i)
    for i in range(indices[-1]-1, indices[-1]-1-2*radius, -1): # go left on the bottom side
        indices.append(i)
    for i in range(indices[-1]-k, indices[-1]-k*(2*radius), -k): # go up on the left side
        indices.append(i)
    # start with index left of center if start_sw==False else with bottom left index
    indices = indices[-(int(start_sw)+1)*radius:] + indices[:-(int(start_sw)+1)*radius]
    return indices