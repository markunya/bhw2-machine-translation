from torchtext.vocab import build_vocab_from_iterator, Vocab
from utils.utils import replace_with_num_if_numeric

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

class VocabBuilder:            
    def __init__(self, use_num=False):
        self.use_num = use_num

    def _yield_tokens(self, file_path: str):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if self.use_num:
                    replace_with_num_if_numeric(tokens, idx2str(IDX.NUM))
                yield tokens

    def build(self, file_path: str, min_freq: int = 1) -> Vocab:
        specials = [idx2str(IDX.UNK), idx2str(IDX.PAD), idx2str(IDX.BOS), idx2str(IDX.EOS)]
        if self.use_num:
            specials.append(idx2str(IDX.NUM))

        vocab = build_vocab_from_iterator(self._yield_tokens(file_path), min_freq=min_freq, specials=specials)
        vocab.set_default_index(vocab[idx2str(IDX.UNK)]) 
        return vocab
