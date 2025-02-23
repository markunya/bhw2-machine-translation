from sacrebleu import corpus_bleu
from utils.class_registry import ClassRegistry

metrics_registry = ClassRegistry()

@metrics_registry.add_to_registry(name="bleu")
class Bleu:
    def __call__(self, hypotheses, references):
        if not isinstance(hypotheses, list) or not isinstance(references, list):
            raise ValueError("Expected list of hypotheses and list of referenses")

        if len(hypotheses) != len(references):
            raise ValueError("Lengths of hypotheses and referenses must mutch")

        bleu_score = corpus_bleu(hypotheses, [references])
        return bleu_score.score