import argparse
from data.datasets import Lang2LangDataset
from utils.utils import break_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_texts_path", type=str, required=True)
    parser.add_argument("--tgt_texts_path", type=str, required=True)
    parser.add_argument("--src_result_texts_path", type=str, required=True)
    parser.add_argument("--tgt_result_texts_path", type=str, required=True)

    args = parser.parse_args()
    src_texts_path = args.src_texts_path
    tgt_texts_path = args.tgt_texts_path
    src_result_texts_path = args.src_result_texts_path
    tgt_result_texts_path = args.tgt_result_texts_path

    dataset = Lang2LangDataset(
        src_texts_path=src_texts_path,
        tgt_texts_path=tgt_texts_path
    )

    src_result_texts = []
    tgt_result_texts = []
    for i in range(len(dataset)):
        src_sentences, src_separators = break_text(dataset[i]['src']['text'])
        tgt_sentences, tgt_separators = break_text(dataset[i]['tgt']['text'])
        if src_separators == tgt_separators:
            for src_sentence, tgt_sentence in zip(src_sentences, tgt_sentences):
                if src_sentence == "\"" or tgt_sentence == "\"":
                    continue
                src_result_texts.append(src_sentence)
                tgt_result_texts.append(tgt_sentence)

    with open(src_result_texts_path, mode='w', encoding='utf-8') as file:
        file.write("\n".join(src_result_texts))
    with open(tgt_result_texts_path, mode='w', encoding='utf-8') as file:
        file.write("\n".join(tgt_result_texts))
