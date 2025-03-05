import random
import torch
import numpy as np
import json

def isnumeric(token):
    return str.isnumeric(token) \
        or str.isnumeric("".join(token.split('.'))) \
        or str.isnumeric("".join(token.split(',')))

def replace_with_num_if_numeric(tokens, num_tok):
    for i, token in enumerate(tokens):
        if isnumeric(token):
            tokens[i] = num_tok

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

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error parsing json: '{file_path}'.")
    except Exception as e:
        print(f"Error occured: {e}")
