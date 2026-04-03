from nltk.translate.bleu_score import corpus_bleu

def compute_bleu(references, predictions):
    bleu1 = corpus_bleu(references, predictions, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0))
    bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-4": bleu4
    }