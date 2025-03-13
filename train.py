import argparse
from training.trainer import TranslatorTrainer
from utils.utils import read_json_file, setup_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
        
    config = read_json_file(config_path)
    setup_seed(config['exp']['seed'])

    trainer = TranslatorTrainer(config)
    trainer.setup_logger()

    trainer.training_loop()
    