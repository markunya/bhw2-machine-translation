import json
import torchtext
from torchtext.vocab import build_vocab_from_iterator

UNK_IDX = 0 # index in the SPECIAL_TOKENS
PAD_IDX = 1 # index in the SPECIAL_TOKENS
BOS_IDX = 2 # index in the SPECIAL_TOKENS
EOS_IDX = 3 # index in the SPECIAL_TOKENS
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

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

def yield_tokens(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()

def build_vocab(file_path: str, min_freq: int = 1) -> torchtext.vocab.Vocab:
    vocab = build_vocab_from_iterator(yield_tokens(file_path), min_freq=min_freq, specials=SPECIAL_TOKENS)
    vocab.set_default_index(vocab["<unk>"]) 
    return vocab

