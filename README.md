## Настройка окружения и запуск
```python
conda create -n env python==3.10.16
source activate env
pip install -r requirements.txt
```

Для запуска обучения:
```python
python train.py --config_path config.json --run_name run_name
```

Для запуска инференса:
```python
python inference.py --config_path config.json
```

## Данные
Все данные для обучения, валидации и тестирования находятся в директории data/texts. На основе train.de-en.* были сгенерированны 2 дополнительных набора данных: train\_small.de-en.* и train\_broken.de-en.*. train\_small.de-en.* представляет собой приблизительно 25\% первых текстов из train.de-en.*, использовался для некоторых экспериментов в целях экономии времени. train\_broken.de-en.* был сгенерирован с помощью скрипта break\_dataset.py. Далее будет описан принцип его работы. Назовем одинаковоструктурированной пару текстов для которых сепараторы (сепараторами являются токены '.', '?', '!') совпадают. Например пара "Hello. World!" и "Привет. Пупсик!" одинаковоструктурированна т.к. списки сепараторов равны ['.', '!']. А напримера пара "Aboba? pupok." и "Абоба. пупок?" таковой не является т.к. ['?', '.'] != ['.', '?']. Алгоритм break\_dataset.py выбирает из датасета только одинаковоструктурированные пары исходный текст - целевой текст (по исследованиям таковых 89\% в train.de-en.*), разбивает на предложения и добавляет в новый датасет уже их как текста, но уже без сепаратора в конце. С очень высокой вероятностью переводы совпадут и поэтому получается датасет с в среднем намного более короткими текстами и скорее всего лишенный сложных примеров когда перевод не такой же структуры.

Проводилось несколько экспериментов по исследованию влияния минимальной частоты вхождения токена в текст (параметр min\_freq) для построения словаря. В целях экономии времени перебирались значения одинаковые для исходного и целевого текста. Перебирались значения 10, 5, 2 (значений 1 не рассматривалось т.к. такие токены можно считать просто шумом и ничего модель с них скорее всего не выучит). Использовался один и тот же трансформер с параметрами emb\_size=256, nhead=8, num\_encoder\_layers: 2, num\_decoder\_layers: 2, dim\_feedforward: 1024. Обучалось с оптимизатором AdamW с learning\_rate равным 0.001, weight\_decay равным 0.001 и экспоненциальным шедулером с gamma 0.98 на протяжении 30000 итераций на данных train.de-en.* с батчем 64 и линейным разогревов в течении первых 5000 итераций. Использовались логики drop\_bos\_eos\_unk и num\_idx (см. раздел логики). Получились результаты представленны в таблице 1.

| min_freq | Validation BLEU | Train BLEU |
|----------|-----------------|------------|
| 2        | 29.3            | 32.1       |
| 5        | 28.8            | 31.0       |
| 10       | 27.9            | 30.2       |

Из них можно заключить, что особого смысла использовать большие min\_freq нету если позволяют ресурсы т.к. это приводит к ухудшению метрики bleu как на валидации так и на обучении. Далее использовалось значиение 2 для min\_freq.

## Логики
Логикой я называю особое поведение для обучения и/или инференса. Были реализованы следующие логики:

-**drop\_bos\_eos\_unk** При данной логике на инференсе из сгенерированного текста удаляются токены <bos>, <eos>, <unk>. Очевидно всегда должна быть включена в результирующей моделе, выключение удобно для отладки. 



-**remove\_punctuation** Данная логика работает на уровне датасета и на обучении выкидывает всю пунктуацию из исходного и целевого языка, на инференсе также удаляет пунктуацию  из входного текста. Изначально предполагалось что блеу значительно снижается засчет того что модель не может выучить правильную постановку знаков препинания. Однако более обьемные модели с этим справляются неплохо. Да и эксперименты показали что данная логика наоборот ухудшает показатели блеу на валидации, поэтому в итоге не была использована. 



-**num\_idx** При данной логике добавляется специальный токен <num> на который заменяются абсолютно все числа и цифры в тексте. Модель обучается на таких текстах, а на инференсе токены <num> в сгенерированных переводах заменяются на числа их исходного текста в таком же порядке. Данная логика позволяет модели не думать над переводом числа, а просто принять факт что это число. Это хорошо потому что множество чисел в данных встречаются малое количество раз и модель просто не сможет понять что это вообще число и какой в нем смысл. К тому же засчет этого удается уменьшить размер словаря приблизительно на 1000. Эксперименты на нескольких однослойных трансформерах показали что данная логика повышает блеу на валидации приблизительно на 2-3 пункта.



-**mask\_idx** Данная логика задумывалась как регуляризация и борьба с переобучением. Идея в том что для достаточно больших текстов заменять случайные слова на специальный токен <mask> в исходном тексте и учить модель пытаться переводить исходя из контекста. Эксперименты показали что логика действительно понижает переобучаемость но также значительно понижает блеу как на валидации так и на обучении. Увеличение модели не помогло увеличить блеу до показателей полученных при обучении без этой логики, поэтому в итоге не использовалась.



-**break\_text** Логика была разработана для того чтобы уменьшить размер переводимых текстов на инференсе тем самым облегчив модели работу. Текст разбивается на предложения и каждое предложение переводиться независимо, а затем все предложения собираются обратно с теми же сепараторами. Датасет train\_broken.de-en.* был разработан как раз для обучения модели которая в последующем будет использовать break\_text. Для маленьких однослойных моделей данная логика действительно улучшила показатели (29 блеу на валидации против 27), но когда я увеличил модель до 4 слоев в кодировщике и декодировщике, то она уже проиграла обычному обучению (32 блеу против 33), что удивительно: модель научилась очень хорошо ставить сепараторы.


## Модель
В конечном счете использовался стандартный трансформер с параметрами emb\_size=256, nhead=8, num\_encoder\_layers: 4, num\_decoder\_layers: 4, dim\_feedforward: 1024. Обучалось с оптимизатором AdamW с learning\_rate равным 0.001, и косинусным шедулером с gamma 0.98 на протяжении 50000 итераций с батчем 64 и линейным разогревов в течении первых 5000 итераций. Замечу что использовался дефолтный dropout 0.1 т.к любое увеличение дропаута ухудшало показатели блеу, хоть и снижало переобучаемость. Модель инициализируется с помощью nn.init.xavier\_uniform\_, что значительно улучшает и стабилизирует обучение. Без такой инициализации блеу не поднимался выше 15.

## Инференс
На инференсе доступны некоторые гиперпараметры для измения качества генерации текста.

-**beam\_size** Количество кандидатов в beam\_search. По умолчанию 1 (жадное декодирование). Увеличение параметра до 6 помогает увеличить блеу на 2-3 пункта, но замедляет инференс. Дальнейшее увеличение параметра не привело к статистически значимым улучшениям.


-**max\_len** Максимальная длина сгенерированного первода. По умолчанию это длина входного текста + 5, однако из экспериментов инференса с выключенной логикой drop\_bos\_eos\_unk даже до этого редко доходит и модель успешно своевременно ставит токен <eos>.

-**repetition\_penalty** Понижает вероятность токенов которые уже были сгенерированы. На начальных этапах была проблема с тем что сгенерированные переводы зацикливались и получалось что-то вроде "the the the", однако далее эта проблема ушла и необходимоти в штрафе больше не возникало. В итоге не использовался.


-**num\_mcd** Количетсво запусков монте-карло дропаута. Для эмбеддингов включается дропаут и логиты усредняются по нескольким прогонам через модель одних и тех же идентификаторов (как test time augmentations). С включенным beam\_search равным 6 дополнительное использование num\_mcd равным 5 не привело к статистически значимым улучшениям. Поэтому также не использовался.

## Результаты
Удалось выбить блеу 33 на валидации и 29 на тесте. При этом на обучающих данных блеу был порядка 39, что свидетельствует о сильном переобучении, которое так и не удалось до конца побороть. Однако качество переводов впечетляет. Вот несколько примеров:

| Исходный текст                                                                                  | Ground-truth перевод                                                                 | Сгенерированный перевод                                                                 |
|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| er sah sehr glücklich aus , was damals ziemlich ungewöhnlich war , da ihn die nachrichten meistens deprimierten . | there was a big smile on his face which was unusual then , because the news mostly depressed him . | he looked very happy , which was pretty unusual at the time because the news was mostly . |
| ich wusste nicht , was das bedeutete , aber es machte meinen vater offensichtlich sehr , sehr glücklich . | i didn 't know what it meant , but i could see that my father was very , very happy . | i didn 't know what that meant , but it obviously made my father very , very happy . |
| deshalb verkleidete ich mich fünf jahre lang als junge und begleitete meine ältere schwester , die nicht mehr alleine ausgehen durfte , zu einer geheimen schule . | so for the next five years , i dressed as a boy to escort my older sister , who was no longer allowed to be outside alone , to a secret school . | so for five years i was a boy with my older sister who was no longer allowed to date , to be a secret school . |
| diesen morgen werde ich niemals vergessen .                                                     | a morning that i will never forget .                                                | this morning i will never forget .                                                     |


Резюмирую интересно заметить, что я испробовал множетсво различных эвристик, но сработала лишь num\_idx логика. В основном самое тупое решение в лоб оказалось лучше всего.

## Описание файла конфигурации
```json
{
    "exp": {
        "project_name": "dl_bhw_2", // название проекта для wandb
        "device": "cuda",
        "seed": 42,
        "use_wandb": false,
        "log_num_samples": 10
    },
    "data": {
        \\ пути к данным. Тексты должны быть разделены символом перевода строки и поделены на токены с помощью пробелов
        "train_src_texts_file_path": "/data/texts/bhw2-texts/train.de-en.de",
        "train_tgt_texts_file_path": "/data/texts/bhw2-texts/train.de-en.en",
        "val_src_texts_file_path": "/data/texts/bhw2-texts/val.de-en.de",
        "val_tgt_texts_file_path": "/data/texts/bhw2-texts/val.de-en.en",
        "test_texts_file_path": "/data/texts/bhw2-texts/test1.de-en.de",

        \\ min\_freq для torchtext.vocab.Vocab
        "src_min_freq": 2,
        "tgt_min_freq": 2,

        "train_batch_size": 64,

        "workers": 4
    },
    "checkpoint_path": null, \\ Путь к *.pth файлу чекпоинта. Чекпоинт может содержать трансформер, оптимизатор и шедулер
    "test": {
        "output_dir": "./" \\ путь к директории в которую будет сгенерирован текстовый файл переводов на инференсе
    },
    "train": {
        "translator": "transformer", \\ модель из models.models
        "translator_args": {
            "emb_size": 256,
            "nhead": 8,
            "num_encoder_layers": 4,
            "num_decoder_layers": 4,
            "dim_feedforward": 1024
        },
        "optimizer": "adamW", \\ оптимизатор из training.optimizers
        "optimizer_args": {
            "lr": 0.0005
        },
        "scheduler": "cosine", \\ шедулер из training.schedulers
        "scheduler_args": {
            "total_steps": 250,
            "reduce_time": "period",
            "step_period": 200,
            "warmup_steps": 25
        },
        \\ список метрик для валидации из metrics.metrics
        "val_metrics": [
            "bleu"
        ],

        "start_step": 1, \\ шаг с которого начнеться training_loop
        "steps": 50000, \\ общее количество шагов для обучения
        "checkpoint_step": 25000, \\ период сохранения чекпоинта
        "val_step": 25000, \\ период валидации
        "log_step": 1000, \\ период логирования потерь
        "checkpoints_dir": "./checkpoints" \\ директория в которую будут сохраняться чекпоинты
    },
    "logics": {
        \\ см. раздел логики
        "num_idx": true,
        "remove_separators": false,
        "break_text": false,
        "drop_bos_eos_unk": true,
        "mask_idx": false
    },
    "inference": {
        \\ см. раздел инференс
        "beam_size": 6,
        "repetition_penalty": 1,
        "num_mcd": 5
    },
    "losses": {
        \\ список потерь для обучения переводчика. Потери складываются с коэффициентами соответствующими значениям в coef
        "cross_entropy_loss": {
            "coef": 1.0,
            "args": {
                "label_smoothing": 0.1
            }
        }
    }
}
```
