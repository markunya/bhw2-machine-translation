import random
import torch
import numpy as np
import json
from typing import List, Tuple
from torchtext.vocab import Vocab

class IDX:
    UNK = 0
    PAD = 1
    BOS = 2
    EOS = 3    
    NUM = 4

def idx2str(idx):
    match idx:
        case IDX.UNK:
            return '<unk>'
        case IDX.PAD:
            return '<pad>'
        case IDX.BOS:
            return '<bos>'
        case IDX.EOS:
            return '<eos>'
        case IDX.NUM:
            return '<num>'
        case _:
            raise ValueError(f'Unsupported idx: {idx}')

def isnumeric(token: str) -> bool:
    return str.isnumeric(token) \
        or str.isnumeric("".join(token.split('.'))) \
        or str.isnumeric("".join(token.split(',')))

def replace_with_num_if_numeric(tokens, num_tok):
    for i, token in enumerate(tokens):
        if isnumeric(token):
            tokens[i] = num_tok

SEPARATORS = set(['.',',','!','?',';'])

def isseparator(token):
    return token in SEPARATORS

def break_text(text: str) -> Tuple[List[str], List[str]]:
    tokens = text.split()
    separators = []
    texts = []
    current = []
    for i, token in enumerate(tokens):
        if isseparator(token):
            texts.append(" ".join(current))
            current.clear()
            separators.append(token)
            continue
        current.append(token)
        if i == len(tokens) - 1:
            texts.append(" ".join(current))
    return texts, separators

def unbreak_text(texts: List[str], separators: List[str]) -> str:
    assert 0 <= len(texts) - len(separators) <= 1
    result = [None] * (len(texts) + len(separators))
    result[0::2] = texts
    result[1::2] = separators
    return " ".join(result)

def remove_bos_eos(indices):
    if len(indices) > 0 and indices[0] == IDX.BOS:
        indices = indices[1:]
    if len(indices) > 0 and indices[-1] == IDX.EOS:
        indices = indices[:-1]
    return indices

def break_indices(indices: List[int], vocab: Vocab) -> Tuple[List[List[int]], List[int]]:
    while indices[-1] == IDX.PAD:
        indices.pop()
    indices = remove_bos_eos(indices)

    separators = []
    indices_arr = []
    current = []
    for i, idx in enumerate(indices):
        if isseparator(vocab.lookup_tokens([idx])[0]):
            indices_arr.append([IDX.BOS] + current + [IDX.EOS])
            current.clear()
            separators.append(idx)
            continue
        current.append(idx)
        if i == len(indices) - 1:
            indices_arr.append([IDX.BOS] + current + [IDX.EOS])
    return indices_arr, separators

def translate_separators(separators: List[int], src_vocab: Vocab, tgt_vocab: Vocab) -> List[int]:
    return tgt_vocab.lookup_indices(src_vocab.lookup_tokens(separators))

def unbreak_indices(indices_arr: List[List[int]], separators: List[int]) -> List[int]:
    assert 0 <= len(indices_arr) - len(separators) <= 1
    indices_arr = [remove_bos_eos(indices) for indices in indices_arr]
    separators = [[separator] for separator in separators]
    result = [None] * (len(indices_arr) + len(separators))
    result[0::2] = indices_arr
    result[1::2] = separators
    result = [IDX.BOS] + [idx for indices in result for idx in indices] + [IDX.EOS]
    return result

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
