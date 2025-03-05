import torch
from torch import nn
from data.vocab_builder import IDX
from utils.utils import tensor_topk
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
            #####
            num_encoder_layers,
            num_decoder_layers,
            emb_size,
            nhead,
            dim_feedforward = 512,
            dropout_prob = 0.1,
            activation='gelu',
            beam_size=1,
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

        self.beam_size = beam_size

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
        mask = (torch.triu(torch.ones((sz, sz), device=self._device())) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
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
    
    def _device(self):
        return next(self.parameters()).device
    
    def _beam_update(self, beam_logits, beam_storage):
        device = self._device()
        beam_size = len(beam_logits)  
        batch_size = beam_logits[0].shape[0]

        log_probs = []
        for i in range(beam_size):
            lp = F.log_softmax(beam_logits[i], dim=-1)
            lp = lp + (beam_storage['eos_mask'][i] * beam_storage['scores'][i]).unsqueeze(1)
            log_probs.append(lp)


        scored_stacked = torch.stack(log_probs, dim=0)
        scored_perm = scored_stacked.permute(1, 0, 2)

        argmaxs = [tensor_topk(scored_perm[b], k=beam_size) 
                for b in range(batch_size)]

        new_beam_storage = {
            'scores': [],
            'eos_mask': [],
            'candidates': []
        }

        for i in range(beam_size):
            new_beam_storage['candidates'].append([])
            new_beam_storage['scores'].append(torch.zeros(batch_size, device=device))
            new_beam_storage['eos_mask'].append(torch.ones(batch_size, device=device))

        for b in range(batch_size):
            top_candidates = argmaxs[b]
            for bs_new_idx, (old_beam_idx, token_idx) in enumerate(top_candidates):
                old_seq = beam_storage['candidates'][old_beam_idx][b]
                new_seq = torch.cat([old_seq, 
                                    torch.tensor([token_idx], device=device)], dim=0)

                new_beam_storage['candidates'][bs_new_idx].append(new_seq)
                new_beam_storage['scores'][bs_new_idx][b] = scored_stacked[old_beam_idx, b, token_idx]

                new_beam_storage['eos_mask'][bs_new_idx][b] = \
                    beam_storage['eos_mask'][old_beam_idx][b] * float(token_idx != IDX.EOS)


        for i in range(beam_size):
            new_beam_storage['candidates'][i] = torch.stack(new_beam_storage['candidates'][i])

        return new_beam_storage

    @torch.no_grad
    def inference(self, src_indices, max_len=None, beam_size=None):
        if beam_size is None:
            beam_size = self.beam_size
        if max_len is None:
            max_len = src_indices.shape[1] + 5

        device = self._device()
    
        src_indices.to(device)
        batch_size = src_indices.shape[0]
        memory = self.transformer.encoder(self.src_tok_emb(src_indices))
        memory = memory.to(device)
        
        beam_storage = {
            'scores': [],
            'eos_mask': [],
            'candidates': [] 
        }

        for _ in range(beam_size):
            beam_storage['scores'].append(
                torch.zeros(batch_size).to(device)
            )
            beam_storage['eos_mask'].append(
                torch.ones(batch_size).to(device)
            )
            beam_storage['candidates'].append(
                torch.ones(batch_size, 1).fill_(IDX.BOS).type(torch.long).to(device)
            )

        for _ in range(max_len-1):
            beam_logits = []
            for i in range(beam_size):
                tgt_mask = (self._generate_square_subsequent_mask(
                    beam_storage['candidates'][i].size(1)
                ).type(torch.bool)).to(device)
                
                tgt_emb = self.tgt_tok_emb(beam_storage['candidates'][i])
                out = self.transformer.decoder(tgt_emb, memory, tgt_mask)

                logits = self.gen_layer(out[:, -1])
                beam_logits.append(logits)

            beam_storage = self._beam_update(beam_logits, beam_storage)

        return beam_storage['candidates'][0]
