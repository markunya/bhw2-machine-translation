import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from training.loggers import TrainingLogger
from models.models import translators_registry
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from training.losses import LossBuilder
from utils.data_utils import csv_to_list
from utils.data_utils import split_mapping
from tqdm import tqdm
from time import time

class TranslatorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['exp']['device']
        self.epoch = None

    def setup(self):
        self.setup_translator()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_metrics()
        self.setup_logger()
        self.setup_trainval_datasets()
        self.setup_dataloaders()

    def setup_inference(self):
        self.setup_classifier()
        self.setup_test_data()

    def setup_translator(self):
        self.classifier = translators_registry[self.config['train']['translator']](
            **self.config['train']['translator_args']
        ).to(self.device)

        if self.config['checkpoint_path']:
            checkpoint = torch.load(self.config['checkpoint_path'])
            self.classifier.load_state_dict(checkpoint['translator_state'])

    def setup_optimizers(self):
        self.optimizer = optimizers_registry[self.config['train']['optimizer']](
            self.classifier.parameters(), **self.config['train']['optimizer_args']
        )

        self.scheduler = schedulers_registry[self.config['train']['scheduler']](
            self.optimizer, **self.config['train']['scheduler_args']
        )

        if self.config['checkpoint_path']:
            checkpoint = torch.load(self.config['checkpoint_path'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

    def setup_losses(self):
        self.loss_builder = LossBuilder(self.config)

    def to_train(self):
        self.classifier.train()

    def to_eval(self):
        self.classifier.eval()

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config['train']['val_metrics']:
            metric = metrics_registry[metric_name]()
            self.metrics.append((metric_name, metric))

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)

    def setup_trainval_datasets(self):
        raise NotImplementedError
        self.train_dataset = datasets_registry[self.config['data']['trainval_dataset']]()
        self.val_dataset = datasets_registry[self.config['data']['trainval_dataset']]()

    def setup_test_data(self):
        self.test_dataset = datasets_registry[self.config['data']['test_dataset']](self.config)
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config['inference']['test_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            pin_memory=True,
        )
    
    def setup_train_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['train_batch_size'],
            shuffle=True,
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            pin_memory=True,
        )

    def setup_val_dataloader(self):
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['val_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            pin_memory=True,
        )

    def setup_dataloaders(self):
        self.setup_train_dataloader()
        self.setup_val_dataloader()

    def training_loop(self):
        num_epochs = self.config['train']['epochs']
        checkpoint_epoch = self.config['train']['checkpoint_epoch']
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def save_checkpoint(self):
        checkpoint = {
            'classifier_state': self.classifier.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        run_name = self.config['exp']['run_name']
        torch.save(checkpoint, os.path.join(self.config['train']['checkpoints_dir'],
                                            f'checkpoint_{run_name}_{self.epoch}.pth'))

    @torch.no_grad()
    def validate(self):
        if len(self.val_dataset) == 0:
            return None
        self.to_eval()
        raise NotImplementedError

    @torch.no_grad()
    def inference(self):
        self.to_eval()
        raise NotImplementedError
