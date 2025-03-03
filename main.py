from utils.model_utils import setup_seed
from training.trainer import TranslatorTrainer
from utils.data_utils import read_json_file

if __name__ == "__main__":
    config = read_json_file('config.json')
    setup_seed(config['exp']['seed'])
    config['exp']['run_name'] = 'main'

    trainer = TranslatorTrainer(config)

    trainer.setup_train()
    trainer.training_loop()

    trainer.translator.beam_size = 4

    trainer.setup_test_data()
    trainer.inference()
    