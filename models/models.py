import torch
from torch import nn
from utils.class_registry import ClassRegistry
from utils.model_utils import weights_init

translators_registry = ClassRegistry()
