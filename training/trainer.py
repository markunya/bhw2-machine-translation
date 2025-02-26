import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from training.loggers import TrainingLogger
from models.models import translators_registry
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from training.losses import LossBuilder
from utils.data_utils import build_vocab

class TranslatorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config['exp']['device']
        self.epoch = None

        self.log_num_samples = config['exp']['log_num_samples']

        self.src_vocab = build_vocab(
            file_path=self.config['data']['train_src_texts_file_path'],
            min_freq=self.config['data']['vocab_min_freq']
        )
        self.tgt_vocab = build_vocab(
            file_path=self.config['data']['train_tgt_texts_file_path'],
            min_freq=self.config['data']['vocab_min_freq']
        )

    def setup(self):
        self.setup_translator()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_metrics()
        self.setup_logger()
        self.setup_trainval_datasets()
        self.setup_dataloaders()

    def setup_inference(self):
        self.setup_translator()
        self.setup_test_data()

    def setup_translator(self):
        self.translator = translators_registry[self.config['train']['translator']](
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
        self.loss_builder = LossBuilder(self.config['losses'])

    def to_train(self):
        self.translator.train()

    def to_eval(self):
        self.translator.eval()

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config['train']['val_metrics']:
            metric = metrics_registry[metric_name]()
            self.metrics.append((metric_name, metric))

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)

    def setup_trainval_datasets(self):
        self.train_dataset = datasets_registry[self.config['data']['trainval_dataset']](
            self.config['data']['train_src_texts_file_path'],
            self.config['data']['train_tgt_texts_file_path'],
            self.src_vocab,
            self.tgt_vocab
        )
        self.val_dataset = datasets_registry[self.config['data']['trainval_dataset']](
            self.config['data']['val_src_texts_file_path'],
            self.config['data']['val_tgt_texts_file_path'],
            self.src_vocab,
            self.tgt_vocab
        )

    def setup_test_data(self):
        self.test_dataset = datasets_registry[self.config['data']['test_dataset']](
            self.config['data']['inf_texts_file_path'],
            self.src_vocab
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config['inference']['test_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,
        )
    
    def setup_train_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['train_batch_size'],
            shuffle=True,
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
        )

    def setup_val_dataloader(self):
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['val_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True,
        )

    def setup_dataloaders(self):
        self.setup_train_dataloader()
        self.setup_val_dataloader()

    def gen_and_log_samples(self):
        src_texts = []
        tgt_texts = []

        for i in range(self.log_num_samples):
            batch = self.val_dataset[i]
            src_texts.append(batch['src']['text'])
            tgt_texts.append(batch['tgt']['text'])
        
        gen_indices_batch = self.translator.inference(batch['src']['indices']).tolist()
        gen_texts = [
            " ".join(self.src_vocab.lookup_tokens(gen_indices))
               for gen_indices in gen_indices_batch
            ]
        self.logger.log_translations(src_texts, tgt_texts, gen_texts)

    def training_loop(self):
        num_epochs = self.config['train']['epochs']
        checkpoint_epoch = self.config['train']['checkpoint_epoch']

        for self.epoch in range(1, num_epochs + 1):
            running_loss = 0
            epoch_losses = {key: [] for key in self.loss_builder.losses.keys()}
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate at epoch {self.epoch}: {current_lr:.6f}")

            with tqdm(self.train_dataloader, desc=f"Training Epoch {self.epoch}\{num_epochs}", unit="batch") as pbar:
                for batch in pbar:                   
                    losses_dict = self.train_step(batch)
                    
                    self.logger.update_losses(losses_dict)
                    for loss_name in epoch_losses.keys():
                        epoch_losses[loss_name].append(losses_dict[loss_name])
                    
                    running_loss = running_loss * 0.9 + losses_dict['total_loss'].item() * 0.1
                    pbar.set_postfix({"loss": running_loss})

            self.logger.log_train_losses(self.epoch)
            self.setup_train_dataloader()
            val_metrics_dict = self.validate()

            self.gen_and_log_samples()
                
            if val_metrics_dict is not None:
                self.logger.log_val_metrics(val_metrics_dict, epoch=self.epoch)

            if self.epoch % checkpoint_epoch == 0:
                self.save_checkpoint()

    def _step_scheduler(self):
        step = (self.scheduler.reduce_time == 'step')
        epoch = (self.scheduler.reduce_time == 'epoch') \
                and (self.step % len(self.train_dataloader) == 0)
        period = (self.scheduler.reduce_time == 'period') \
                and (self.step % self.scheduler.period == 0)
        if step or epoch or period:
            self.scheduler.step()

    def train_step(self, batch):
        self.to_train()
        self.optimizer.zero_grad()

        src_indices = batch['src']['indices'].to(self.device)
        tgt_indices = batch['tgt']['indices'].to(self.device)

        logits = self.translator(src_indices, tgt_indices)

        tgt_out = tgt_indices[1:, :]
        loss_dict = self.loss_builder.calculate_loss(
            pred_logits=logits,
            target=tgt_out
        )

        self.optimizer.step()
        self._step_scheduler()

        return loss_dict

    def save_checkpoint(self):
        checkpoint = {
            'translator_state': self.translator.state_dict(),
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

        metrics_dict = {}

        for metric_name, _ in self.metrics:
            metrics_dict['train_' + metric_name] = 0
            metrics_dict['val_' + metric_name] = 0

        train_iter = iter(self.train_dataloader)
        val_iter = iter(self.val_dataloader)

        num_batches = min(len(self.train_dataloader), len(self.val_dataloader))

        for _ in range(num_batches):
            train_batch = next(train_iter)
            val_batch = next(val_iter)

            train_src_indices = train_batch['src']['indices'].to(self.device)
            val_src_indices = val_batch['src']['indices'].to(self.device)

            train_real_translation = train_batch['tgt']['text']
            val_real_translation = val_batch['tgt']['text']
            train_gen_translation = self.translator.inference(train_src_indices)
            val_gen_translation = self.translator.inference(val_src_indices)

            for metric_name, metric in self.metrics:
                metrics_dict['train_' + metric_name] += metric(train_gen_translation, train_real_translation) / num_batches
                metrics_dict['val_' + metric_name] += metric(val_gen_translation, val_real_translation) / num_batches

        print('Metrics: ', ", ".join(f"{key}={value}" for key, value in metrics_dict.items()))
        self.setup_val_dataloader()
        self.setup_train_dataloader()

        return metrics_dict


    @torch.no_grad()
    def inference(self):
        self.to_eval()
        raise NotImplementedError
