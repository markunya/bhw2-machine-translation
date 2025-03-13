import argparse
from utils.utils import setup_seed
from training.trainer import TranslatorTrainer
from utils.utils import read_json_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    config_path = args.config_path
        
    config = read_json_file(config_path)
    assert config['checkpoint_path'] is not None
    setup_seed(config['exp']['seed'])

    assert config['checkpoint_path'] is not None
    trainer = TranslatorTrainer(config)
    trainer.inference()