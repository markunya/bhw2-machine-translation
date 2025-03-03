import argparse
from utils.model_utils import setup_seed
from training.trainer import TranslatorTrainer
from utils.data_utils import read_json_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
    
    config = read_json_file(config_path)
    setup_seed(config['exp']['seed'])

    trainer = TranslatorTrainer(config)

    trainer.setup_validation()
    metrics_dict = {}
    num_batches = len(trainer.val_dataloader)
    trainer._validate_impl(iter(trainer.val_dataloader), metrics_dict, num_iters=num_batches, prefix='val')
        
    print('Metrics: ', ", ".join(f"{key}={value}" for key, value in metrics_dict.items()))
    