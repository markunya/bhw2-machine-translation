import torch
from utils.class_registry import ClassRegistry
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import PAD_IDX

class LangDataset(Dataset):
    def __init__(self, texts_path, vocab):
        self.data = []
        with open(texts_path, "r", encoding="utf-8") as texts_file:
            for line in texts_file:
                text = line.strip()
                indices = [vocab['<bos>']] + vocab.lookup_indices(text.split(' ')) + [vocab['<eos>']]
                tokens = f'<bos> {text} <eos>' 
                self.data.append({
                    'text': text,
                    'tokens': tokens,
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
            padding_value=PAD_IDX,
            batch_first=True
        )
        
        return result

class Lang2LangDataset(Dataset):
    def __init__(self, src_texts_path, tgt_texts_path, src_vocab, tgt_vocab):
        self.src_dataset = LangDataset(src_texts_path, src_vocab)
        self.tgt_dataset = LangDataset(tgt_texts_path, tgt_vocab)

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
