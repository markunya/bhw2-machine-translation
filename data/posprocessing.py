from data.vocab_builder import IDX
from utils.utils import isnumeric

def drop_unk_bos_eos(gen_indices_batch):
    for i, indices in enumerate(gen_indices_batch):
        cutted = []
        for idx in indices[1:]:
            if idx == IDX.UNK:
                continue
            if idx == IDX.EOS:
                break
            cutted.append(idx)
        gen_indices_batch[i] = cutted

def get_nums_batch(texts_batch):
    nums_batch = []
    for text in texts_batch:
        nums_batch.append([])
        tokens = text.split()
        for token in tokens:
            if isnumeric(token):
                nums_batch[-1].append(token)
    return nums_batch

def indices2text(src_texts_batch, gen_indices_batch, vocab):
    gen_translations = []
    nums_batch = get_nums_batch(src_texts_batch)
    for i, indices in enumerate(gen_indices_batch):
        num_j = 0
        gen_translations.append([])

        for idx in indices:
            if idx == IDX.NUM:
                if num_j >= len(nums_batch[i]):
                    continue
                token = nums_batch[i][num_j]
                num_j += 1
            else:
                token = vocab.lookup_tokens([idx])[0]
                
            gen_translations[-1].append(token)
        gen_translations[-1] = " ".join(gen_translations[-1])

    return gen_translations

def add_dot(translations_batch):
    for i in range(len(translations_batch)):
        if translations_batch[i][-1] == '!' or translations_batch[i][-1] == '?':
            continue
        translations_batch[i] += ' .'
