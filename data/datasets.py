import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from data.vocab_builder import IDX, idx2str
from utils.utils import replace_with_num_if_numeric

class LangDataset(Dataset):
    def __init__(self, texts_path, vocab, drop_dot=False, num_logic=False):
        self.data = []
        with open(texts_path, "r", encoding="utf-8") as texts_file:
            for line in texts_file:
                text = line.strip()
                tokens = text.split()

                if drop_dot and len(tokens) > 0 and tokens[-1] == '.':
                    tokens.pop()

                if num_logic:
                    replace_with_num_if_numeric(tokens, idx2str(IDX.NUM))

                indices = [vocab[idx2str(IDX.BOS)]] + vocab.lookup_indices(tokens) + [vocab[idx2str(IDX.EOS)]]
                self.data.append({
                    'text': text,
                    'tokens': [idx2str(IDX.BOS)] + vocab.lookup_tokens(vocab.lookup_indices(tokens)) + [idx2str(IDX.EOS)],
                    'indices': indices
                })
    
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
                sort=True,
                drop_dot=False,
                num_logic=False
            ):
        
        self.src_dataset = LangDataset(
            src_texts_path, src_vocab, drop_dot=drop_dot, num_logic=num_logic
        )
        self.tgt_dataset = LangDataset(
            tgt_texts_path, tgt_vocab, drop_dot=drop_dot, num_logic=num_logic
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
        return {
            'src': self.src_dataset[index],
            'tgt': self.tgt_dataset[index]
        }

    def __len__(self):
        return len(self.src_dataset)
    
    @staticmethod
    def collate_fn(batch):
        return {
            'src': LangDataset.collate_fn([item['src'] for item in batch]),
            'tgt': LangDataset.collate_fn([item['tgt'] for item in batch])
        }
