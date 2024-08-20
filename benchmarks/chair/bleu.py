import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

nltk.download('punkt')

def bleu_four(references, candidates):
    tokenized_references = [[nltk.word_tokenize(ref) for ref in refs] for refs in references]
    tokenized_candidates = [nltk.word_tokenize(cand) for cand in candidates]

    score = corpus_bleu(
        tokenized_references, 
        tokenized_candidates, 
        weights=(0.25, 0.25, 0.25, 0.25),  # 1-gram, 2-gram, 3-gram, 4-gram에 대한 동일한 가중치
        smoothing_function=SmoothingFunction().method1  # 스무딩 기능을 사용하여 0으로 나누기 문제 처리
    )
    return score

def bleu_one(references, candidates):
    tokenized_references = [[nltk.word_tokenize(ref) for ref in refs] for refs in references]
    tokenized_candidates = [nltk.word_tokenize(cand) for cand in candidates]

    score = corpus_bleu(tokenized_references, tokenized_candidates, weights=(1, 0, 0, 0))
    return score