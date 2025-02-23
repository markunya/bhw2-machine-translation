from utils.class_registry import ClassRegistry
from torch.utils.data import Dataset


datasets_registry = ClassRegistry()

@datasets_registry.add_to_registry(name='lang_dataset')
class LangDataset(Dataset):
    def __init__(self, texts_path, vocab):
        self.vocab = vocab
        with open(texts_path, "r", encoding="utf-8") as texts_file:
            self.texts = texts_file.read().splitlines() 
    
    def __getitem__(self, index):
        text = self.texts[index]
        indices = self.vocab.encode(text)
        return indices
    
    def __len__(self):
        return len(self.texts)
    
@datasets_registry.add_to_registry(name='lang2lang_dataset')
class Lang2LangDataset(Dataset):
    def __init__(self, l1_texts_path, l2_texts_path, l1_vocab, l2_vocab):
        self.l1_dataset = LangDataset(l1_texts_path, l1_vocab)
        self.l2_dataset = LangDataset(l2_texts_path, l2_vocab)

        if len(self.l1_dataset) != len(self.l2_dataset):
            raise ValueError(f'Lengths of datasets don\'t match')

    def __getitem__(self, index):
        return self.l1_dataset[index], self.l2_dataset[index]

    def __len__(self):
        return len(self.l1_dataset)
