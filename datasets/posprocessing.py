from utils.data_utils import EOS_IDX

def remove_specials(gen_indices_batch):
    processed = []
    for indices in gen_indices_batch:
        cutted = []
        for idx in indices[1:]:
            if idx == EOS_IDX:
                break
            cutted.append(idx)
        processed.append(cutted)
