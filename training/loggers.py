import wandb
import torch
import os
from PIL import Image
from collections import defaultdict

class WandbLogger:
    def __init__(self, config):
        wandb.login(key=os.environ['WANDB_KEY'].strip())
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
    def log_table(table_name: str, columns: list, data: list):
        table = wandb.Table(columns=columns)

        for row in data:
            table.add_data(*row)

        wandb.log({table_name: table})

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
        self.losses_memory = defaultdict(list)

    @log_if_enabled
    def log_train_losses(self, epoch: int):
        averaged_losses = {name: sum(values) / len(values) for name, values in self.losses_memory.items()}
        self.logger.log_values(averaged_losses, epoch)
        self.losses_memory.clear()

    @log_if_enabled
    def log_val_metrics(self, val_metrics: dict, epoch: int):
        self.logger.log_values(val_metrics, epoch)

    @log_if_enabled
    def log_translations(self, l1_texts, l2_texts_references, l2_texts_hypotheses):
        columns = ["Source Text", "Reference Translation", "Hypothesis Translation"]
        data = list(zip(l1_texts, l2_texts_references, l2_texts_hypotheses))
        self.logger.log_table("translation_samples", columns, data)

    @log_if_enabled
    def update_losses(self, losses_dict):
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
