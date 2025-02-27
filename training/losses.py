import torch
from torch import nn

from torch.nn import functional as F
from utils.class_registry import ClassRegistry
from utils.data_utils import PAD_IDX

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
    def __init__(self, *args, ignore_index=PAD_IDX, **kwargs):
        super().__init__(*args, ignore_index=ignore_index, **kwargs)
