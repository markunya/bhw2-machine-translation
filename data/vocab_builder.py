from torchtext.vocab import build_vocab_from_iterator, Vocab
from utils.utils import replace_with_num_if_numeric, IDX, idx2str

class VocabBuilder:            
    def __init__(self, use_num=False, break_text=False):
        self.use_num = use_num
        self.break_text = break_text
        
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
