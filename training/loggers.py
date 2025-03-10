import wandb
import os
from collections import defaultdict

class WandbLogger:
    def __init__(self, config):
        if 'wandb_key' in config['exp']:
            key = config['exp']['wandb_key']
        else:
            key = os.environ['WANDB_KEY'].strip()

        wandb.login(key=key)
        self.wandb_args = {
                "id": wandb.util.generate_id(),
                "project": config['exp']['project_name'],
                "name": config['exp']['run_name'],
                "config": config,
            }

        wandb.init(**self.wandb_args, resume="allow")
        wandb.config.update(config)

    @staticmethod
    def log_values(values_dict: dict, step: int):
        wandb.log(values_dict, step=step)

    @staticmethod
    def log_table(table_name: str, columns: list, data: list, step: int):
        table = wandb.Table(columns=columns)

        for row in data:
            table.add_data(*row)

        wandb.log({table_name: table}, step=step)

def log_if_enabled(func):
    def wrapper(self, *args, **kwargs):
        if self.use_logger:
            return func(self, *args, **kwargs)
    return wrapper

class TrainingLogger:
    def __init__(self, config):
        self.use_logger = config['exp']['use_wandb']
        if not self.use_logger:
            return
        self.logger = WandbLogger(config)
        self.train_losses_memory = defaultdict(list)
        self.val_losses_memory = defaultdict(list)

    @log_if_enabled
    def log_train_losses(self, epoch: int):
        averaged_losses = {name: sum(values) / len(values) for name, values in self.train_losses_memory.items()}
        self.logger.log_values(averaged_losses, epoch)
        self.train_losses_memory.clear()

    @log_if_enabled
    def log_val_losses(self, epoch: int):
        averaged_losses = {name: sum(values) / len(values) for name, values in self.val_losses_memory.items()}
        self.logger.log_values(averaged_losses, epoch)
        self.val_losses_memory.clear()

    @log_if_enabled
    def log_val_metrics(self, val_metrics: dict, step: int):
        self.logger.log_values(val_metrics, step)

    @log_if_enabled
    def log_translations(self, src_texts, dst_texts_references, dst_texts_hypotheses, step):
        columns = ["Source Text", "Reference Translation", "Hypothesis Translation"]
        data = list(zip(src_texts, dst_texts_references, dst_texts_hypotheses))
        self.logger.log_table("translation_samples", columns, data, step)

    @log_if_enabled
    def update_train_losses(self, losses_dict):
        for loss_name, loss_val in losses_dict.items():
            self.train_losses_memory[loss_name].append(loss_val)

    @log_if_enabled
    def update_val_losses(self, losses_dict):
        for loss_name, loss_val in losses_dict.items():
            self.val_losses_memory[loss_name].append(loss_val)
