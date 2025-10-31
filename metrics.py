import math
from typing import List, Tuple
import sacrebleu
from rouge_score import rouge_scorer

def compute_bleu(preds: List[str], refs: List[str]) -> float:
    # sacrebleu expects sys (preds) and list of references
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score

def compute_chrf(preds: List[str], refs: List[str]) -> float:
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    return chrf.score

def compute_rouge_l(preds: List[str], refs: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]
    return 100.0 * (sum(scores) / max(1, len(scores)))

def perplexity_from_loss(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")