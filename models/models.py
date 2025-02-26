import torch
import torchtext
from torch import nn
from utils.model_utils import create_mask
import torch.nn.functional as F
import math
from utils.class_registry import ClassRegistry

translators_registry = ClassRegistry()

class PositionalEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size,
            emb_size,
            dropout_prob,
            max_len
        ):
        super().__init__()

        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)

        range = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)

        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * range)
        pos_embedding[:, 1::2] = torch.cos(pos * range)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tokens):
        token_emb = self.embedding(tokens) * math.sqrt(self.emb_size)
        return self.dropout(token_emb + self.pos_embedding[:token_emb.size(0), :])

@translators_registry.add_to_registry(name="transformer")
class TransormerTranslator(nn.Module):
    def __init__(
            self,
            num_encoder_layers,
            num_decoder_layers,
            max_len,
            emb_size,
            nhead,
            src_vocab_size,
            tgt_vocab_size,
            dim_feedforward = 512,
            dropout_prob = 0.1
        ):

        super().__init__()
        self.transformer = nn.Transformer(
                                    d_model=emb_size,
                                    nhead=nhead,
                                    num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout_prob
                                )
        self.gen_layer = nn.Linear(emb_size, tgt_vocab_size)

        pos_emb_kwargs = dict(
            emb_size=emb_size,
            dropout_prob=dropout_prob,
            max_len=max_len
        )
        self.src_tok_emb = PositionalEmbedding(
                src_vocab_size, **pos_emb_kwargs
            )
        self.tgt_tok_emb = PositionalEmbedding(
                tgt_vocab_size, **pos_emb_kwargs
            )
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_indices, tgt_indices):
        tgt_input = tgt_indices[:,:-1]
        raise ValueError('wtf')
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_indices, tgt_input)
        
        src_emb = self.src_tok_emb(src_indices)
        tgt_emb = self.tgt_tok_emb(tgt_indices)
        outs = self.transformer(
                    src_emb, tgt_emb, src_mask, tgt_mask, None,
                    src_padding_mask, tgt_padding_mask, src_padding_mask
                )
        
        return self.gen_layer(outs)

    def inference(self, src_indices):
        raise NotImplementedError
