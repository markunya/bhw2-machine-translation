import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from training.loggers import TrainingLogger
from models.models import translators_registry
from datasets.datasets import Lang2LangDataset, LangDataset
from metrics.metrics import metrics_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from training.losses import LossBuilder
from utils.data_utils import build_vocab, EOS_IDX

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

    def setup_train(self):
        self.setup_translator()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_metrics()
        self.setup_logger()
        self.setup_train_data()
        self.setup_val_data

    def setup_validation(self):
        self.setup_translator()
        self.setup_metrics()
        self.setup_val_data()

    def setup_inference(self):
        self.setup_translator()
        self.setup_test_data()

    def setup_translator(self):
        self.translator = translators_registry[self.config['train']['translator']](
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            max_len=self.config['data']['max_len'],
            **self.config['train']['translator_args']
        ).to(self.device)

        if self.config['checkpoint_path']:
            checkpoint = torch.load(self.config['checkpoint_path'])
            self.translator.load_state_dict(checkpoint['translator_state'])

    def setup_optimizers(self):
        self.optimizer = optimizers_registry[self.config['train']['optimizer']](
            self.translator.parameters(), **self.config['train']['optimizer_args']
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

    def setup_train_data(self):
        self.train_dataset = Lang2LangDataset(
            self.config['data']['train_src_texts_file_path'],
            self.config['data']['train_tgt_texts_file_path'],
            self.src_vocab,
            self.tgt_vocab
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['train_batch_size'],
            shuffle=True,
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True
        ) 

    def setup_val_data(self):
        self.val_dataset = Lang2LangDataset(
            self.config['data']['val_src_texts_file_path'],
            self.config['data']['val_tgt_texts_file_path'],
            self.src_vocab,
            self.tgt_vocab
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['val_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )

    def setup_test_data(self):
        self.test_dataset = LangDataset(
            self.config['data']['inf_texts_file_path'],
            self.src_vocab
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config["inference"]["test_batch_size"],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,
        )

    def _get_translation_from_gen_indices(self, src_indices, gen_indices_batch):
        src_indices = src_indices.tolist()
        gen_indices_batch = gen_indices_batch.tolist()
        translations_batch = []
        for indices, src in zip(gen_indices_batch, src_indices):
            cutted = []
            for idx in indices[1:]:
                if idx == EOS_IDX or len(cutted) > len(src) + 5:
                    break
                cutted.append(idx)

            translations_batch.append(
                " ".join(self.tgt_vocab.lookup_tokens(cutted))
            )

        return translations_batch


    def gen_and_log_samples(self, epoch):
        src_texts = []
        tgt_texts = []

        for i in range(self.log_num_samples):
            batch = self.val_dataset[i]
            src_texts.append(batch['src']['text'])
            tgt_texts.append(batch['tgt']['text'])

        gen_indices_batch = self.translator.inference(
            torch.tensor(batch['src']['indices']).unsqueeze(0).to(self.device)
        )
        gen_texts = self._get_translation_from_gen_indices(
            torch.tensor([batch['src']['indices']]), gen_indices_batch
        )

        self.logger.log_translations(src_texts, tgt_texts, gen_texts, epoch)

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
            
            self.scheduler.step()

            self.logger.log_train_losses(self.epoch)
            val_metrics_dict = self.validate()

            self.gen_and_log_samples(self.epoch)
                
            if val_metrics_dict is not None:
                self.logger.log_val_metrics(val_metrics_dict, epoch=self.epoch)

            if self.epoch % checkpoint_epoch == 0:
                self.save_checkpoint()

    def train_step(self, batch):
        self.to_train()
        self.optimizer.zero_grad()

        src_indices = batch['src']['indices'].to(self.device)
        tgt_indices = batch['tgt']['indices'].to(self.device)

        logits = self.translator(src_indices, tgt_indices)

        tgt_out = tgt_indices[:,1:]
        loss_dict = self.loss_builder.calculate_loss(
            pred_logits=logits.reshape(-1, logits.shape[-1]),
            target=tgt_out.reshape(-1)
        )
        loss_dict['total_loss'].backward()

        self.optimizer.step()

        return loss_dict

    def save_checkpoint(self):
        checkpoint = {
            'translator_state': self.translator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        run_name = self.config['exp']['run_name']
        torch.save(checkpoint, os.path.join(self.config['train']['checkpoints_dir'],
                                            f'checkpoint_{run_name}_{self.epoch}.pth'))
        
    def _validate_impl(self, iter, metrics_dict, num_iters, prefix):
        for metric_name, _ in self.metrics:
            metrics_dict[f'{prefix}_{metric_name}'] = 0

        for i in range(num_iters):
            batch = next(iter)

            src_indices = batch['src']['indices'].to(self.device)
            real_translation = batch['tgt']['text']
            pred_indices = self.translator.inference(src_indices)

            gen_translations = self._get_translation_from_gen_indices(src_indices, pred_indices)

            for metric_name, metric in self.metrics:
                metrics_dict[f'{prefix}_{metric_name}'] += metric(gen_translations, real_translation) / num_iters


    @torch.no_grad()
    def validate(self):
        if len(self.val_dataset) == 0:
            return None
        
        self.to_eval()

        metrics_dict = {}
        num_batches = min(len(self.train_dataloader), len(self.val_dataloader))
        self._validate_impl(iter(self.val_dataloader), metrics_dict, num_iters=num_batches, prefix='val')
        self._validate_impl(iter(self.train_dataloader), metrics_dict, num_iters=num_batches, prefix='train')
        
        print('Metrics: ', ", ".join(f"{key}={value}" for key, value in metrics_dict.items()))

        return metrics_dict


    @torch.no_grad()
    def inference(self):
        self.to_eval()
        run_name = self.config['exp']['run_name']

        out_path = os.path.join(
            self.config['inference']['output_dir'],
            f'inf_out_{run_name}.txt'
        )

        translations = []
        with tqdm(self.test_dataloader, desc=f"Inference progress", unit="batch") as pbar:
            for batch in pbar:
                src_indices = batch['indices'].to(self.device)
                pred_indices = self.translator.inference(src_indices)

                gen_translations = self._get_translation_from_gen_indices(src_indices, pred_indices)
                translations.extend(gen_translations)

        with open(out_path, mode='w', encoding='utf-8') as file:
            file.write("\n".join(translations))
