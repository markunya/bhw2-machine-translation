import torch
import torch.nn.functional as F
from torch import nn
from utils.class_registry import ClassRegistry
from data.vocab_builder import IDX

losses_registry = ClassRegistry()

class LossBuilder:
    def __init__(self, losses_config):
        self.losses = {}
        self.coefs = {}

        for loss_name in losses_config.keys():
            self.coefs[loss_name] = losses_config[loss_name]['coef']
            loss_args = {}
            if f'args' in losses_config[loss_name]:
                loss_args = losses_config[loss_name]['args']
            self.losses[loss_name] = losses_registry[loss_name](**loss_args)

    def calculate_loss(self, pred_logits, target):
        loss_dict = {}
        loss_dict['total_loss'] = 0.0

        for loss_name, loss in self.losses.items():
            loss_val = loss(pred_logits, target)
            loss_dict['total_loss'] += self.coefs[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        return loss_dict

@losses_registry.add_to_registry(name='cross_entropy_loss')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, ignore_index=IDX.PAD, **kwargs):
        super().__init__(*args, ignore_index=ignore_index, **kwargs)

@losses_registry.add_to_registry(name='diversity_penalty')
class DiversityPenalty(nn.Module):
    def __init__(self, N=2, coefs=None, ignore_index=0, label_smoothing=0.0):
        super().__init__()
        if coefs is None:
            coefs = [1.0] * N
        assert len(coefs) == N
        
        self.N = N
        self.coefs = coefs
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        """
        input: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = input.shape
        device = input.device
        
        # Convert logits to probabilities
        probs = F.softmax(input, dim=-1)

        total_loss = 0.0
        
        for n in range(1, self.N + 1):
            # Для последовательных n-грамм (i..i+n-1) и (i+1..i+n)
            valid_positions = seq_len - n + 1
            if valid_positions <= 0:
                continue
                
            # Собираем n-граммы для модельных вероятностей и таргетов
            model_ngrams = []
            target_ngrams = []
            
            for i in range(valid_positions):
                # Модельные n-граммы
                model_ngram = probs[:, i:i+n, :]  # (batch, n, vocab)
                model_ngrams.append(model_ngram)
                
                # Таргетные n-граммы
                target_ngram = target_one_hot[:, i:i+n, :]  # (batch, n, vocab)
                target_ngrams.append(target_ngram)
            
            # (batch, valid_positions, positions, vocab) -> (batch, valid_positions, positions*vocab)
            model_ngrams = torch.stack(model_ngrams).reshape(batch_size, valid_positions, -1)  
            target_ngrams = torch.stack(target_ngrams).reshape(batch_size, valid_positions, -1)
            
            # Вычисляем CE между последовательными n-граммами
            ce_loss = 0.0
            for i in range(valid_positions - 1):
                # Текущая и следующая n-грамма
                current = model_ngrams[..., i, :]  # (batch, n, vocab)
                next = model_ngrams[..., i+1, :]
                
                # CE между распределениями n-грамм
                ce = F.cross_entropy(next, current, ignore_index=IDX.PAD)
                ce_loss += ce
                
            # Нормализуем и добавляем в общий loss
            avg_ce = ce_loss / (valid_positions - 1)
            penalty = -self.coefs[n-1] * avg_ce  # Штрафуем за низкую CE
            total_loss += penalty
            
        return total_loss
    