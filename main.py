from training.trainer import TranslatorTrainer
from utils.utils import read_json_file, setup_seed

if __name__ == "__main__":
    config = read_json_file('config.json')
    setup_seed(config['exp']['seed'])
    config['exp']['run_name'] = 'main'

    trainer = TranslatorTrainer(config)
    trainer.training_loop()
    trainer.inference()
