from utils.class_registry import ClassRegistry
from torch.optim import AdamW

optimizers_registry = ClassRegistry()

@optimizers_registry.add_to_registry(name='adamW')
class AdamW_(AdamW):
    def __init__(self, params, lr=0.0001, beta1=0.9, beta2=0.999, weight_decay=0.01):
        super().__init__(params, lr=lr, betas=(beta1,beta2), weight_decay=weight_decay)
