import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.utils import IDX, idx2str
import utils.utils as utils
import random

class LangDataset(Dataset):
    def __init__(self, texts_path, vocab, remove_separators=False, num_logic=False):
        self.data = []
        with open(texts_path, "r", encoding="utf-8") as texts_file:
            for line in texts_file:
                text = line.strip()
                self.data.append(self._make_batch_from_text(text, vocab, remove_separators, num_logic))
                
    @staticmethod
    def _make_batch_from_text(text, vocab, remove_separators, num_logic):
        if remove_separators:
            texts_arr, _ = utils.break_text(text)
            text = " ".join(texts_arr)

        tokens = text.split()

        if num_logic:
            utils.replace_with_num_if_numeric(tokens, idx2str(IDX.NUM))

        indices = [vocab[idx2str(IDX.BOS)]] \
                + vocab.lookup_indices(tokens) \
                + [vocab[idx2str(IDX.EOS)]]
                    
        return {
                'text': text,
                'indices': indices,
                'tokens': [idx2str(IDX.BOS)] \
                            + vocab.lookup_tokens(vocab.lookup_indices(tokens)) \
                            + [idx2str(IDX.EOS)]
                }
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        result = {
            'text': [],
            'tokens': [],
            'indices': []
        }

        for item in batch:
            result['text'].append(item['text'])
            result['tokens'].append(item['tokens'])
            result['indices'].append(torch.tensor(item['indices'], dtype=torch.long))

        result['indices'] = pad_sequence(
            result['indices'],
            padding_value=IDX.PAD,
            batch_first=True
        )
        
        return result

class Lang2LangDataset(Dataset):
    def __init__(self,
                src_texts_path,
                tgt_texts_path,
                src_vocab,
                tgt_vocab,
                sort=False,
                break_text=False,
                remove_separators=False,
                num_logic=False
            ):
        self.num_logic = num_logic
        self.break_text = break_text
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.src_dataset = LangDataset(
            src_texts_path,
            src_vocab,
            remove_separators=break_text or remove_separators,
            num_logic=num_logic
        )
        self.tgt_dataset = LangDataset(
            tgt_texts_path,
            tgt_vocab,
            remove_separators=break_text or remove_separators,
            num_logic=num_logic
        )

        if sort:
            avg_lengths = [
                (len(src['tokens']) + len(tgt['tokens'])) / 2
                for src, tgt in zip(self.src_dataset, self.tgt_dataset)
            ]

            sorted_pairs = sorted(
                zip(self.src_dataset, self.tgt_dataset, avg_lengths),
                key=lambda x: x[2]
            )
            self.src_dataset, self.tgt_dataset, _ = zip(*sorted_pairs)

        if len(self.src_dataset) != len(self.tgt_dataset):
            raise ValueError(f'Lengths of datasets don\'t match')

    def __getitem__(self, index):
        src_batch = self.src_dataset[index]
        tgt_batch = self.tgt_dataset[index]
        
        if self.break_text:
            src_texts_arr, src_separators = utils.break_text(src_batch['text'])
            tgt_texts_arr, tgt_separators = utils.break_text(tgt_batch['text'])
            if src_separators == tgt_separators and len(src_texts_arr) == len(tgt_texts_arr):
                j = random.choice(range(len(src_texts_arr)))
                src_batch = LangDataset._make_batch_from_text(
                    text=src_texts_arr[j],
                    vocab=self.src_vocab,
                    remove_separators=True,
                    num_logic=self.num_logic
                )
                tgt_batch = LangDataset._make_batch_from_text(
                    text=tgt_texts_arr[j],
                    vocab=self.tgt_vocab,
                    remove_separators=True,
                    num_logic=self.num_logic
                )

        return {
            'src': src_batch,
            'tgt': tgt_batch
        }

    def __len__(self):
        return len(self.src_dataset)
    
    @staticmethod
    def collate_fn(batch):
        return {
            'src': LangDataset.collate_fn([item['src'] for item in batch]),
            'tgt': LangDataset.collate_fn([item['tgt'] for item in batch])
        }
