import os
import torch
import data.posprocessing as pp
from tqdm import tqdm
from torch.utils.data import DataLoader
from training.loggers import TrainingLogger
from models.models import translators_registry
from data.datasets import Lang2LangDataset, LangDataset
from data.dataloaders import InfiniteLoader
from metrics.metrics import metrics_registry
from training.optimizers import optimizers_registry
from training.schedulers import schedulers_registry
from training.losses import LossBuilder
from data.vocab_builder import VocabBuilder, IDX

class TranslatorTrainer:
    def __init__(self, config):

        self.config = config
        self.device = config['exp']['device']
        self.step = None

        self.log_num_samples = config['exp']['log_num_samples']

        self.drop_bos_eos_unk_logic = config['inference']['drop_bos_eos_unk_logic']
        self.drop_dot_logic = config['inference']['drop_dot_logic']
        self.num_logic = config['inference']['num_logic']

        builder = VocabBuilder(use_num=self.num_logic)
        self.src_vocab = builder.build(
            file_path=self.config['data']['train_src_texts_file_path'],
            min_freq=self.config['data']['src_min_freq']
        )
        self.tgt_vocab = builder.build(
            file_path=self.config['data']['train_tgt_texts_file_path'],
            min_freq=self.config['data']['tgt_min_freq']
        )

    def setup_train(self):
        self.setup_translator()
        self.setup_optimizers()
        self.setup_losses()
        self.setup_metrics()
        self.setup_logger()
        self.setup_train_data()
        self.setup_val_data()

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
            **self.config['train']['translator_args']
        ).to(self.device)
        tqdm.write('Translator successfully created')

        checkpoint_path = self.config['checkpoint_path']
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.translator.load_state_dict(checkpoint['translator_state'], strict=False)
            tqdm.write(f'State of translator loaded from {checkpoint_path}')

    def setup_optimizers(self):
        self.optimizer = optimizers_registry[self.config['train']['optimizer']](
            self.translator.parameters(), **self.config['train']['optimizer_args']
        )
        self.scheduler = schedulers_registry[self.config['train']['scheduler']](
            self.optimizer, **self.config['train']['scheduler_args']
        )
        tqdm.write('Optimizer and scheduler successfully created')

        checkpoint_path = self.config['checkpoint_path']
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.optimizer.load_state_dict(checkpoint['optimizer_state'], strict=False)
            self.scheduler.load_state_dict(checkpoint['scheduler_state'], strict=False)
            tqdm.write(f'States of optimizer and scheduler loaded from {checkpoint_path}')

    def setup_losses(self):
        self.loss_builder = LossBuilder(self.config['losses'])
        tqdm.write('Loss builder successfully created')

    def to_train(self):
        self.translator.train()

    def to_eval(self):
        self.translator.eval()

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config['train']['val_metrics']:
            metric = metrics_registry[metric_name]()
            self.metrics.append((metric_name, metric))
        tqdm.write('Metrics successfully created')

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)

    def setup_train_data(self):
        self.train_dataset = Lang2LangDataset(
            self.config['data']['train_src_texts_file_path'],
            self.config['data']['train_tgt_texts_file_path'],
            self.src_vocab,
            self.tgt_vocab,
            drop_dot=self.drop_dot_logic
        )

        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=self.config['data']['train_batch_size'],
            num_workers=self.config['data']['workers'],
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True
        )
        tqdm.write('Data for train successfully prepared')

    def setup_val_data(self):
        self.val_dataset = Lang2LangDataset(
            self.config['data']['val_src_texts_file_path'],
            self.config['data']['val_tgt_texts_file_path'],
            self.src_vocab,
            self.tgt_vocab,
            drop_dot=self.drop_dot_logic
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['val_batch_size'],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
        tqdm.write('Data for validation successfully prepared')

    def setup_test_data(self):
        self.test_dataset = LangDataset(
            self.config['data']['test_texts_file_path'],
            self.src_vocab,
            drop_dot=self.drop_dot_logic
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["test_batch_size"],
            multiprocessing_context="spawn" if self.config['data']['workers'] > 0 else None,
            num_workers=self.config['data']['workers'],
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True,
        )
        tqdm.write('Data for test successfully prepared')

    def _get_translation_from_gen_indices(self, src_texts_batch, gen_indices_batch):
        gen_indices_batch = gen_indices_batch.tolist()
        if self.drop_bos_eos_unk_logic:
            pp.drop_unk_bos_eos(gen_indices_batch)

        translations_batch = pp.indices2text(src_texts_batch, gen_indices_batch, self.tgt_vocab)

        if self.drop_dot_logic:
            pp.add_dot(translations_batch)

        return translations_batch

    def gen_and_log_samples(self, step):
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
            batch['src']['text'], gen_indices_batch
        )

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
        running_loss = 0

        with tqdm(total=num_steps, desc='Training Progress', unit='step') as progress:
            for self.step in range(start_step, num_steps + 1):
                step_losses = {key: [] for key in self.loss_builder.losses.keys()}
                
                progress.set_postfix({
                    "loss": running_loss,
                    "lr": self.optimizer.param_groups[0]['lr']
                })

                losses_dict = self.train_step()
                self._step_scheduler()
                progress.update(1)
                
                self.logger.update_losses(losses_dict)
                for loss_name in step_losses.keys():
                    step_losses[loss_name].append(losses_dict[loss_name])
                    running_loss = running_loss * 0.9 + losses_dict['total_loss'].item() * 0.1

                if self.step % log_step == 0:
                    self.logger.log_train_losses(self.step)

                if self.step % val_step == 0:
                    val_metrics_dict = self.validate()
                    self.gen_and_log_samples(self.step)
                    if val_metrics_dict is not None:
                        self.logger.log_val_metrics(val_metrics_dict, step=self.step)

                if self.step % checkpoint_step == 0:
                    self.save_checkpoint()

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
        loss_dict['total_loss'].backward()

        self.optimizer.step()

        return loss_dict

    def save_checkpoint(self):
        checkpoint = {
            'translator_state': self.translator.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        run_name = self.config['exp']['run_name']
        torch.save(checkpoint, os.path.join(self.config['train']['checkpoints_dir'],
                                            f'checkpoint_{run_name}_{self.step}.pth'))
        
    def _validate_impl(self, dataloader, metrics_dict, num_iters, prefix):
        for metric_name, _ in self.metrics:
            metrics_dict[f'{prefix}_{metric_name}'] = 0

        iterator = iter(dataloader)
        for _ in tqdm(range(num_iters), desc=f'Validating {prefix}', unit='batch'):
            batch = next(iterator)

            src_indices = batch['src']['indices'].to(self.device)
            real_translation = batch['tgt']['text']
            pred_indices = self.translator.inference(src_indices)

            gen_translations = self._get_translation_from_gen_indices(batch['src']['text'], pred_indices)

            for metric_name, metric in self.metrics:
                metrics_dict[f'{prefix}_{metric_name}'] += metric(gen_translations, real_translation) / num_iters

    @torch.no_grad()
    def validate(self):
        if len(self.val_dataset) == 0:
            return None
        
        self.to_eval()

        metrics_dict = {}
        num_batches = min(len(self.train_dataloader), len(self.val_dataloader))
        self._validate_impl(self.val_dataloader, metrics_dict, num_iters=num_batches, prefix='val')
        self._validate_impl(self.train_dataloader, metrics_dict, num_iters=num_batches, prefix='train')
        
        tqdm.write(f'Metrics: {", ".join(f"{key}={value}" for key, value in metrics_dict.items())}')

        return metrics_dict

    @torch.no_grad()
    def inference(self):
        self.to_eval()
        run_name = self.config['exp']['run_name']

        out_path = os.path.join(
            self.config['test']['output_dir'],
            f'test_out_{run_name}.txt'
        )

        translations = []
        with tqdm(self.test_dataloader, desc=f"Inference progress", unit="batch") as pbar:
            for batch in pbar:
                src_indices = batch['indices'].to(self.device)
                pred_indices = self.translator.inference(src_indices)

                gen_translations = self._get_translation_from_gen_indices(batch['text'], pred_indices)
                translations.extend(gen_translations)

        with open(out_path, mode='w', encoding='utf-8') as file:
            file.write("\n".join(translations))
