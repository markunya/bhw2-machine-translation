import random
import torch
import numpy as np
from torch import nn

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def tensor_topk(tensor, k, dim=-1):
    flat_tensor = tensor.flatten()
    indices = torch.topk(flat_tensor, k=k, dim=dim).indices
    indices = indices[torch.argsort(-flat_tensor[indices])]
    positions = [torch.unravel_index(idx, tensor.shape) for idx in indices]
    return positions

def setup_seed(seed):
    random.seed(seed)                
    np.random.seed(seed)
    torch.manual_seed(seed)        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
