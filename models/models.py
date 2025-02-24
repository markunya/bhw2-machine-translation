import torch
from torch import nn
from utils.class_registry import ClassRegistry
from utils.model_utils import weights_init
from abc import ABC, abstractmethod

translators_registry = ClassRegistry()

class BaseTranslator(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, src_indices, dst_indices):
        pass

    @abstractmethod
    def inference(self, src_indices):
        pass

@translators_registry.add_to_registry(name="transformer")
class Transformer(BaseTranslator):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

        self.apply(weights_init)

    def forward(self, src_indices, dst_indices):
        src = self.embedding(src_indices) + self.positional_encoding[:, :src_indices.size(1)]
        dst = self.embedding(dst_indices) + self.positional_encoding[:, :dst_indices.size(1)]

        src = src_indices.permute(1, 0, 2)
        dst = dst.permute(1, 0, 2)

        transformer_out = self.transformer(src, dst)
        transformer_out = transformer_out.permute(1, 0, 2)
        output = self.fc_out(transformer_out)

        return output
    
    def inference(self, src_indices, max_len=None):
        if max_len is None:
            max_len = 2*len(src_indices)
        raise NotImplementedError
