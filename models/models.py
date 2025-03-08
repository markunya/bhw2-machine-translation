import torch
from torch import nn
import math
from utils.utils import IDX
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
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tokens):
        token_emb = self.embedding(tokens) * math.sqrt(self.emb_size)
        pos_emb = self.pos_embedding[:,:token_emb.size(1)]
        return self.dropout(token_emb + pos_emb)

@translators_registry.add_to_registry(name="transformer")
class TransformerTranslator(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            #####
            num_encoder_layers,
            num_decoder_layers,
            emb_size,
            nhead,
            dim_feedforward = 512,
            dropout_prob = 0.1,
            activation='gelu',
            pos_emb_max_len=100
        ):

        super().__init__()
        self.transformer = nn.Transformer(
                                    d_model=emb_size,
                                    nhead=nhead,
                                    num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout_prob,
                                    activation=activation,
                                    batch_first=True
                                )
        self.gen_layer = nn.Linear(emb_size, tgt_vocab_size)

        pos_emb_kwargs = dict(
            emb_size=emb_size,
            dropout_prob=dropout_prob,
            max_len=pos_emb_max_len
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

    def _generate_square_subsequent_mask(self, sz):
        device = next(self.parameters()).device
        mask = (torch.triu(torch.ones((sz, sz), device=device) == 1)).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _create_mask(self, src_indices, tgt_input):
        src_seq_len = src_indices.shape[1]
        tgt_seq_len = tgt_input.shape[1]

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)

        src_padding_mask = (src_indices == IDX.PAD)
        tgt_padding_mask = (tgt_input == IDX.PAD)

        return tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src_indices, tgt_indices):
        tgt_input = tgt_indices[:,:-1]
        tgt_mask, src_padding_mask, tgt_padding_mask \
                        = self._create_mask(src_indices, tgt_input)
        
        src_emb = self.src_tok_emb(src_indices)
        tgt_emb = self.tgt_tok_emb(tgt_input)

        outs = self.transformer(
                    src_emb, tgt_emb, None, tgt_mask, None,
                    src_padding_mask, tgt_padding_mask, src_padding_mask
                )
        
        return self.gen_layer(outs)

    def encode(self, src_indices: torch.Tensor):
        return self.transformer.encoder(self.src_tok_emb(src_indices))

    def decode(self, tgt_indices: torch.Tensor, memory: torch.Tensor):
        sz = tgt_indices.shape[1]
        tgt_mask = self._generate_square_subsequent_mask(sz)
        out = self.transformer.decoder(self.tgt_tok_emb(tgt_indices), memory, tgt_mask)
        logits = self.gen_layer(out[:, -1])
        return logits