import torch
import math
from torchtext.vocab import Vocab
from torch import nn
from typing import List
from utils.utils import isnumeric
from data.vocab_builder import IDX

def drop_unk_bos_eos(gen_indices: List[int]) -> List[int]:
    result = []
    for idx in gen_indices[1:]:
        if idx == IDX.UNK or idx == IDX.BOS:
            continue
        if idx == IDX.EOS:
            break
        result.append(idx)
    return result

def indices2text(src_text: str, gen_indices: List[int], vocab: Vocab) -> str:
    text = []
    nums = [token for token in src_text.split() if isnumeric(token)]

    j = 0
    for idx in gen_indices:
        if idx == IDX.NUM:
            if j >= len(nums):
                continue
            token = nums[j]
            j += 1
        else:
            token = vocab.lookup_tokens([idx])[0]
            
        text.append(token)

    return " ".join(text)

def add_dot(translation: str) -> str:
    if len(translation) > 0 and (translation[-1] == '!' or translation[-1] == '?'):
        return translation
    return translation + ' .'

def apply_repetition_penalty(log_probs, prev_tokens, penalty):
    prev_tokens_flat = [token for seq in prev_tokens for token in seq]
    for token in set(prev_tokens_flat):
        log_probs[:, token] -= math.log(penalty)
    return log_probs

def beam_search(
        src_indices: torch.Tensor,
        translator: nn.Module,
        beam_size=1,
        max_len=None,
        repetition_penalty=1.0
    ) -> torch.Tensor:

    if max_len is None:
        max_len = src_indices.shape[1] + 5

    device = next(translator.parameters()).device
    batch_size = src_indices.size(0)
    
    memory = translator.encode(src_indices)

    finished = [False]*beam_size
    beams = [
        (torch.full((1, 1), IDX.BOS, dtype=torch.long, device=device), False, 0.0)
    ]

    for _ in range(max_len - 1):
        new_beams = []
        for seq, finished, score in beams:
            if seq[0, -1].item() == IDX.EOS or finished:
                new_beams.append((seq, True, score))
                continue
            
            logits = translator.decode(
                seq,
                memory,
            )
            
            log_probs = torch.log_softmax(logits[-1, :], dim=-1)
            
            log_probs = apply_repetition_penalty(
                log_probs.unsqueeze(0),
                seq.tolist(),
                repetition_penalty
            )

            topk_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)
            
            for i in range(beam_size):
                new_token = topk_ids[:, i].unsqueeze(0)
                new_seq = torch.cat([seq, new_token], dim=-1)
                new_score = score + topk_probs[:, i].item()

                new_beams.append((new_seq, False, new_score))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]
        if all(finished for _, finished, _ in beams):
            break

    return beams[0][0][0]

def translate(
        src_text: str,
        src_indices: torch.Tensor,
        translator: nn.Module,
        tgt_vocab: Vocab,

        drop_bos_eos_unk_logic=True,
        drop_dot_logic=False,

        beam_size=1,
        max_len=None,
        repetition_penalty=1.0,
    ) -> str:

    gen_indices = beam_search(
        src_indices=src_indices,
        translator=translator,
        beam_size=beam_size,
        max_len=max_len,
        repetition_penalty=repetition_penalty
    ).tolist()

    if drop_bos_eos_unk_logic:
        gen_indices = drop_unk_bos_eos(gen_indices)

    gen_text = indices2text(src_text, gen_indices, tgt_vocab)

    if drop_dot_logic:
        gen_text = add_dot(gen_text)
    
    return gen_text
