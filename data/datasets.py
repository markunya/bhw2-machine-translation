import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.utils import IDX, idx2str
import utils.utils as utils
import random
from typing import List, Optional, Union, Tuple
from torchtext.vocab import Vocab
from utils.utils import (
    replace_with_num_if_numeric,
    remove_punctuation_from_tokens,
    IDX, idx2str, build_vocab
)

class LangDataset(Dataset):
    def __init__(
            self,
            texts_path: str,
            vocab: Optional[None] = None,
            vocab_min_freq: int = 1,
            remove_punctuation: bool = False,
            num_idx_logic: bool = False
        ):

        self.texts_arr = []
        self.tokens_arr = []
        with open(texts_path, "r", encoding="utf-8") as texts_file:
            for line in texts_file:
                text = line.strip()
                self.texts_arr.append(text)

                tokens = text.split()
                if num_idx_logic:
                    replace_with_num_if_numeric(tokens)
                if remove_punctuation:
                    tokens = remove_punctuation_from_tokens(tokens)
                
                self.tokens_arr.append(tokens)
        
        if vocab is None:
            self.vocab: Vocab = build_vocab(self.tokens_arr, min_freq=vocab_min_freq)
        else:
            self.vocab: Vocab = vocab
        
        self.indices_arr = []
        for i in range(len(self.tokens_arr)):
            self.tokens_arr[i] = [idx2str(IDX.BOS)] + self.tokens_arr[i] + [idx2str(IDX.EOS)]
            self.indices_arr.append(
                torch.tensor(self.vocab.lookup_indices(self.tokens_arr[i]), dtype=torch.long)
            )

        assert len(self.texts_arr) == len(self.tokens_arr)
        assert len(self.tokens_arr) == len(self.indices_arr)

    
    def __getitem__(self, index):
        return {
            'text': self.texts_arr[index],
            'tokens': self.tokens_arr[index],
            'indices': self.indices_arr[index]
        }
    
    def __len__(self):
        return len(self.tokens_arr)

    @staticmethod
    def collate_fn(batch):
        result = {}

        result['text'] = [item['text'] for item in batch]
        result['tokens'] = [item['tokens'] for item in batch]
        result['indices'] = pad_sequence(
            [item['indices'] for item in batch],
            padding_value=IDX.PAD,
            batch_first=True
        )
        
        return result

class Lang2LangDataset(Dataset):
    def __init__(
            self,
            src_texts_path: str,
            tgt_texts_path: str,
            src_vocab: Optional[Vocab] = None,
            src_vocab_min_freq: int = 1,
            tgt_vocab: Optional[Vocab] = None,
            tgt_vocab_min_freq: int =1,
            remove_punctuation: Union[bool, Tuple[bool]] = False,
            num_idx_logic: bool = False,
            mask_idx_logic: bool = False,
            mask_num_step: int = 10
        ):

        self.mask_idx_logic = mask_idx_logic
        self.mask_num_step = mask_num_step

        src_rs = remove_punctuation if isinstance(remove_punctuation, bool) else remove_punctuation[0]
        tgt_rs = remove_punctuation if isinstance(remove_punctuation, bool) else remove_punctuation[1]
        
        self.src_dataset = LangDataset(
            texts_path=src_texts_path,
            vocab=src_vocab,
            vocab_min_freq=src_vocab_min_freq,
            remove_punctuation=src_rs,
            num_idx_logic=num_idx_logic
        )
        self.tgt_dataset = LangDataset(
            texts_path=tgt_texts_path,
            vocab=tgt_vocab,
            vocab_min_freq=tgt_vocab_min_freq,
            remove_punctuation=tgt_rs,
            num_idx_logic=num_idx_logic
        )

        if len(self.src_dataset) != len(self.tgt_dataset):
            raise ValueError(f'Lengths of datasets don\'t match')

    def __getitem__(self, index):
        src_batch = self.src_dataset[index]
        tgt_batch = self.tgt_dataset[index]

        if self.mask_idx_logic:
            l_indices = src_batch['indices'].shape[-1] - 2
            num_masks = l_indices // self.mask_num_step
            msk_token = idx2str(IDX.MSK)

            for _ in range(num_masks):
                m_idx = random.choice(range(1, l_indices-1))
                src_batch['indices'][m_idx] = IDX.MSK
                src_batch['tokens'][m_idx] = msk_token

        result_batch = {
            'src': src_batch,
            'tgt': tgt_batch
        }
        
        return result_batch

    def __len__(self):
        return len(self.src_dataset)
    
    @staticmethod
    def collate_fn(batch):
        return {
            'src': LangDataset.collate_fn([item['src'] for item in batch]),
            'tgt': LangDataset.collate_fn([item['tgt'] for item in batch])
        }
