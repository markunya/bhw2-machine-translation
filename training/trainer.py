import os
import torch
import random

from utils.translate import translate
from tqdm import tqdm
from torch.utils.data import DataLoader
from training.loggers import TrainingLogger
from training.losses import LossBuilder
from data.datasets import Lang2LangDataset, LangDataset
from data.dataloaders import InfiniteLoader

from models.models import translators_registry
from metrics.metrics import metrics_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry

class TranslatorTrainer:
    def __init__(self, config):

        self.config = config
        self.device = config['exp']['device']
        self.step = None

        self.log_num_samples = config['exp']['log_num_samples']

        self.drop_bos_eos_unk_logic = config['logics']['drop_bos_eos_unk']
        self.break_text_logic = config['logics']['break_text']
        self.num_idx_logic = config['logics']['num_idx']
        self.remove_punctuation_logic = config['logics']['remove_punctuation']
        self.mask_idx_logic = config['logics']['mask_idx']

        num_mcd = config['inference']['num_mcd']
        assert num_mcd >= 1
        
        self.translate_kwargs = dict(
            drop_bos_eos_unk_logic=self.drop_bos_eos_unk_logic,
            break_text_logic=self.break_text_logic,
            num_mcd=num_mcd,
            beam_size=self.config['inference']['beam_size'],
            repetition_penalty=self.config['inference']['repetition_penalty'],
        )

        self._setup_train_data()
        self._setup_val_data()
        self._setup_test_data()
        self._setup_translator()
        self._setup_optimizers()
        self._setup_losses()
        self._setup_metrics()

    def _setup_translator(self):
        self.translator = translators_registry[self.config['train']['translator']](
            src_vocab_size=len(self.train_dataset.src_dataset.vocab),
            tgt_vocab_size=len(self.train_dataset.tgt_dataset.vocab),
            **self.config['train']['translator_args']
        ).to(self.device)

        self.translate_kwargs['translator'] = self.translator
        tqdm.write('Translator successfully created')

        checkpoint_path = self.config['checkpoint_path']
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.translator.load_state_dict(checkpoint['translator_state'], strict=False)
            tqdm.write(f'State of translator loaded from {checkpoint_path}')

    def _setup_optimizers(self):
        self.optimizer = optimizers_registry[self.config['train']['optimizer']](
            self.translator.parameters(), **self.config['train']['optimizer_args']
        )
        self.scheduler = schedulers_registry[self.config['train']['scheduler']](
            self.optimizer, **self.config['train']['scheduler_args']
        )
        tqdm.write('Optimizer and scheduler successfully created')

        checkpoint_path = self.config['checkpoint_path']
        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path)
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                tqdm.write(f'States of optimizer and scheduler loaded from {checkpoint_path}')
            except Exception as e:
                tqdm.write(f'An error occured when loading states form optimizer and scheduler. Initialize from scratch. {e}')

    def _setup_losses(self):
        self.loss_builder = LossBuilder(self.config['losses'])
        tqdm.write('Loss builder successfully created')

    def to_train(self):
        self.translator.train()

    def to_eval(self):
        self.translator.eval()

    def _setup_metrics(self):
        self.metrics = []
        for metric_name in self.config['train']['val_metrics']:
            metric = metrics_registry[metric_name]()
            self.metrics.append((metric_name, metric))
        tqdm.write('Metrics successfully created')

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)

    def _setup_train_data(self):
        self.train_dataset = Lang2LangDataset(
            src_texts_path=self.config['data']['train_src_texts_file_path'],
            tgt_texts_path=self.config['data']['train_tgt_texts_file_path'],
            src_vocab=None,
            src_vocab_min_freq=self.config['data']['src_min_freq'],
            tgt_vocab=None,
            tgt_vocab_min_freq=self.config['data']['tgt_min_freq'],
            num_idx_logic=self.num_idx_logic,
            mask_idx_logic=self.mask_idx_logic,
            remove_punctuation=self.remove_punctuation_logic,
        )

        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=self.config['data']['train_batch_size'],
            num_workers=self.config['data']['workers'],
            collate_fn=self.train_dataset.collate_fn,
            shuffle=True,
            pin_memory=True
        )

        self.translate_kwargs['src_vocab'] = self.train_dataset.src_dataset.vocab
        self.translate_kwargs['tgt_vocab'] = self.train_dataset.tgt_dataset.vocab

        tqdm.write('Data for train successfully prepared')

    def _setup_val_data(self):
        self.val_dataset = Lang2LangDataset(
            src_texts_path=self.config['data']['val_src_texts_file_path'],
            tgt_texts_path=self.config['data']['val_tgt_texts_file_path'],
            src_vocab=self.train_dataset.src_dataset.vocab,
            tgt_vocab=self.train_dataset.tgt_dataset.vocab,
            remove_punctuation=(self.remove_punctuation_logic, False),
            num_idx_logic=self.num_idx_logic
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
        tqdm.write('Data for validation successfully prepared')

    def _setup_test_data(self):
        self.test_dataset = LangDataset(
            texts_path=self.config['data']['test_texts_file_path'],
            vocab=self.train_dataset.src_dataset.vocab,
            remove_punctuation=self.remove_punctuation_logic,
            num_idx_logic=self.num_idx_logic
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,
        )
        tqdm.write('Data for test successfully prepared')

    def _gen_and_log_samples(self, step: int):
        src_texts = []
        tgt_texts = []
        gen_texts = []

        iterator = iter(self.val_dataloader)
        for _ in range(self.log_num_samples):
            batch = next(iterator)
            src_text = batch['src']['text']
            src_indices = batch['src']['indices'].to(self.device)
            tgt_text = batch['tgt']['text']

            gen_text = translate(
                src_text=src_text[0],
                src_indices=src_indices,
                **self.translate_kwargs
            )
            
            src_texts.append(src_text)
            tgt_texts.append(tgt_text)
            gen_texts.append(gen_text)

        self.logger.log_translations(src_texts, tgt_texts, gen_texts, step)

    def _step_scheduler(self):    
        step = (self.scheduler.reduce_time == 'step')
        epoch = (self.scheduler.reduce_time == 'epoch') \
                    and (self.step % len(self.train_dataloader) == 0)
        period = (self.scheduler.reduce_time == 'period') \
                    and (self.step % self.scheduler.period == 0)
        if step or epoch or period:
            self.scheduler.step()

    def training_loop(self):
        start_step = self.config['train']['start_step']
        num_steps = self.config['train']['steps']
        checkpoint_step = self.config['train']['checkpoint_step']
        val_step = self.config['train']['val_step']
        log_step = self.config['train']['log_step']
        train_running_loss = 0
        val_running_loss = 0
        val_iter = iter(self.val_dataloader)

        with tqdm(total=num_steps, desc='Training Progress', unit='step') as progress:
            for self.step in range(start_step, num_steps + 1):                
                progress.set_postfix({
                    "train_loss": train_running_loss,
                    "val_loss": val_running_loss,
                    "lr": self.optimizer.param_groups[0]['lr']
                })

                train_losses_dict = self.train_step()
                self._step_scheduler()
                progress.update(1)

                next_log_step = ((self.step + log_step - 1) // log_step) * log_step
                if self.step + len(self.val_dataloader) - 1 >= next_log_step:
                    val_losses_dict = self._val_step(val_iter)
                    val_running_loss = val_running_loss * 0.9 + val_losses_dict['val_total_loss'].item() * 0.1
                    self.logger.update_val_losses(val_losses_dict)
                
                self.logger.update_train_losses(train_losses_dict)
                train_running_loss = train_running_loss * 0.9 + train_losses_dict['train_total_loss'].item() * 0.1

                if self.step % log_step == 0:
                    self.logger.log_train_losses(self.step)
                    self.logger.log_val_losses(self.step)
                    val_iter = iter(self.val_dataloader)

                if self.step % val_step == 0:
                    val_metrics_dict = self.validate()
                    self._gen_and_log_samples(self.step)
                    if val_metrics_dict is not None:
                        self.logger.log_val_metrics(val_metrics_dict, step=self.step)

                if self.step % checkpoint_step == 0:
                    self.save_checkpoint()

    @torch.no_grad()
    def _val_step(self, val_iter):
        self.to_eval()
        batch = next(val_iter)
        src_indices = batch['src']['indices'].to(self.device)
        tgt_indices = batch['tgt']['indices'].to(self.device)
            
        logits = self.translator(src_indices, tgt_indices)
            
        tgt_out = tgt_indices[:,1:]
        val_loss_dict = self.loss_builder.calculate_loss(
                        pred_logits=logits.reshape(-1, logits.shape[-1]),
                        target=tgt_out.reshape(-1)
                    )
        val_loss_dict = {f'val_{key}':value for key, value in val_loss_dict.items()}
        return val_loss_dict
        
    
    def train_step(self):
        self.to_train()
        self.optimizer.zero_grad()

        batch = next(self.train_dataloader)
        src_indices = batch['src']['indices'].to(self.device)
        tgt_indices = batch['tgt']['indices'].to(self.device)
            
        logits = self.translator(src_indices, tgt_indices)

        tgt_out = tgt_indices[:,1:]
        loss_dict = self.loss_builder.calculate_loss(
            pred_logits=logits.reshape(-1, logits.shape[-1]),
            target=tgt_out.reshape(-1)
        )
        loss_dict = {f'train_{key}':value for key, value in loss_dict.items()}
        loss_dict['train_total_loss'].backward()

        self.optimizer.step()
        
        return loss_dict

    def save_checkpoint(self):
        checkpoint = {
            'translator_state': self.translator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            # 'scheduler_state': self.scheduler.state_dict()
        }
        run_name = self.config['exp']['run_name']
        checkpoints_dir = self.config['train']['checkpoints_dir']
    
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
            
        try:
            torch.save(checkpoint, os.path.join(self.config['train']['checkpoints_dir'],
                                            f'checkpoint_{run_name}_{self.step}.pth'))
        except Exception as e:
            tqdm.write(f'An error occured when saving checkpoint at step {self.step}: {e}"')

    @torch.no_grad()
    def validate(self):
        if len(self.val_dataset) == 0:
            return None
        
        self.to_eval()
        metrics_dict = {}

        for metric_name, _ in self.metrics:
            metrics_dict[f'train_{metric_name}'] = 0
            metrics_dict[f'val_{metric_name}'] = 0
    
        val_iterator = iter(self.val_dataloader)
        for _ in tqdm(range(len(self.val_dataloader)), desc=f'Validating', unit='batch'):
            val_batch = next(val_iterator)
            train_batch = self.train_dataset[random.choice(range(len(self.train_dataset)))]
            
            val_src_indices = val_batch['src']['indices'].to(self.device)
            val_src_text = val_batch['src']['text']
            val_tgt_translation = val_batch['tgt']['text']
            train_src_indices = torch.tensor(train_batch['src']['indices'],
                                            dtype=torch.long).to(self.device)
            train_src_text = train_batch['src']['text']
            train_tgt_translation = train_batch['tgt']['text']

            val_gen_translation = translate(
                src_indices=val_src_indices,
                src_text=val_src_text[0],
                **self.translate_kwargs
            )
            train_gen_translation = translate(
                src_indices=train_src_indices.unsqueeze(0),
                src_text=train_src_text,
                **self.translate_kwargs
            )

            for metric_name, metric in self.metrics:
                metrics_dict[f"val_{metric_name}"] \
                    += metric([val_gen_translation], val_tgt_translation) / len(self.val_dataloader)
                metrics_dict[f"train_{metric_name}"] \
                    += metric([train_gen_translation], [train_tgt_translation]) / len(self.val_dataloader)
            
        tqdm.write(f'Metrics: {", ".join(f"{key}={value}" for key, value in metrics_dict.items())}')

        return metrics_dict

    @torch.no_grad()
    def inference(self):
        self.to_eval()
        out_dir = self.config['test']['output_dir']

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        if self.config['checkpoint_path'] is not None:
            checkpoint_name = os.path.basename(self.config['checkpoint_path'])[:-4]
        else:
            checkpoint_name = 'main'

        out_path = os.path.join(
            out_dir,
            f'test_out_{checkpoint_name}.txt'
        )

        translations = []
        with tqdm(self.test_dataloader, desc=f"Inference progress", unit="batch") as pbar:
            for batch in pbar:
                src_indices = batch['indices'].to(self.device)
                src_text = batch['text']

                assert len(src_text) == 1

                translation = translate(
                    src_indices=src_indices[0],
                    src_text=src_text[0],
                    **self.translate_kwargs
                )
                translations.append(translation)

        with open(out_path, mode='w', encoding='utf-8') as file:
            file.write("\n".join(translations))
