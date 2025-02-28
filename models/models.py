import torch
from torch import nn
from utils.data_utils import BOS_IDX, EOS_IDX, PAD_IDX
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
            max_len,
            #####
            num_encoder_layers,
            num_decoder_layers,
            emb_size,
            nhead,
            dim_feedforward = 256,
            dropout_prob = 0.1,
            beam_size=1
        ):

        super().__init__()
        self.transformer = nn.Transformer(
                                    d_model=emb_size,
                                    nhead=nhead,
                                    num_encoder_layers=num_encoder_layers,
                                    num_decoder_layers=num_decoder_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout_prob,
                                    batch_first=True
                                )
        self.gen_layer = nn.Linear(emb_size, tgt_vocab_size)

        self.max_len = max_len
        self.beam_size = beam_size

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

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=next(self.parameters()).device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _create_mask(self, src_indices, tgt_input):
        src_seq_len = src_indices.shape[1]
        tgt_seq_len = tgt_input.shape[1]

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros(
            (src_seq_len, src_seq_len),
            device=next(self.parameters()).device
        ).type(torch.bool)

        src_padding_mask = (src_indices == PAD_IDX)
        tgt_padding_mask = (tgt_input == PAD_IDX)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


    def forward(self, src_indices, tgt_indices):
        tgt_input = tgt_indices[:,:-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask \
                        = self._create_mask(src_indices, tgt_input)
        
        src_emb = self.src_tok_emb(src_indices)
        tgt_emb = self.tgt_tok_emb(tgt_input)

        outs = self.transformer(
                    src_emb, tgt_emb, src_mask, tgt_mask, None,
                    src_padding_mask, tgt_padding_mask, src_padding_mask
                )
        
        return self.gen_layer(outs)
    
    def inference(self, src_indices, beam_size=None):
        if beam_size is None:
            beam_size = self.beam_size

        device = next(self.parameters()).device
        
        src_indices.to(device)
        batch_size = src_indices.shape[0]
        num_tokens = src_indices.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)
        memory = self.transformer.encoder(self.src_tok_emb(src_indices), src_mask)
        pred_indices = torch.ones(batch_size, 1).fill_(BOS_IDX).type(torch.long).to(device)

        num_eos = 0

        for _ in range(self.max_len-1):
            memory = memory.to(device)
            tgt_mask = (self._generate_square_subsequent_mask(pred_indices.size(1)).type(torch.bool)).to(device)
            
            tgt_emb = self.tgt_tok_emb(pred_indices)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask)

            prob = self.gen_layer(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            pred_indices = torch.cat([
                pred_indices,
                next_word.unsqueeze(1)
            ], dim=1)

            num_eos += (next_word == EOS_IDX).sum().item()
            if num_eos == batch_size:
                break

        return pred_indices
