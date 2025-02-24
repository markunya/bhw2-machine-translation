import json
import os
from collections import Counter

class Vocabulary:
    def __init__(
            self,
            vocab_dir="vocabularys",
            vocab_filename="vocab",
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            min_freq=1,
            save_vocab_file=True
        ):

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.counter = Counter()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vocab_dir = os.path.join(self.base_dir, vocab_dir)

        self.save_vocab_file = save_vocab_file
        self.vocab_file = os.path.join(self.vocab_dir, f"{vocab_filename}.json")

        if os.path.exists(self.vocab_file):
            self.load_vocab()

    def build(self, data_path=None):
        if self.word2idx:
            print(f'Vocabluary loaded from {self.vocab_file}. Skip building.')
            return
        
        if data_path is None:
            raise ValueError('No vacabulary was found to load and data_path was not specified.')

        with open(data_path, "r", encoding="utf-8") as file:
            sentences = file.read().splitlines() 

        for sentence in sentences:
            self.counter.update(sentence.split())

        self.word2idx = {}
        self.idx2word = {}
        
        for i, token in enumerate([self.unk_token, self.pad_token,
                                   self.bos_token, self.eos_token]):
            self.word2idx[token] = i
            self.idx2word[i] = token

        idx = 4
        for word, freq in self.counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        if self.save_vocab_file:
            self.save_vocab()

    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in sentence.split()]

    def decode(self, indices):
        return " ".join([self.idx2word.get(idx, self.unk_token) for idx in indices])

    def save_vocab(self):
        vocab_data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "counter": self.counter,
            "min_freq": self.min_freq,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token
        }
        
        if not os.path.exists(self.vocab_file):
            os.makedirs(os.path.dirname(self.vocab_file), exist_ok=True)

        with open(self.vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)

    def load_vocab(self):
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.word2idx = vocab_data["word2idx"]
        self.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
        self.counter = Counter(vocab_data["counter"])
        self.min_freq = vocab_data["min_freq"]
        self.unk_token = vocab_data["unk_token"]
        self.pad_token = vocab_data["pad_token"]
        self.bos_token = vocab_data["bos_token"]
        self.eos_token = vocab_data["eos_token"]

    def __len__(self):
        return len(self.word2idx)
