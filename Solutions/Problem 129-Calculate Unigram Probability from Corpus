# https://www.deep-ml.com/problems/129
from collections import Counter
def unigram_probability(corpus: str, word: str) -> float:
    corpus = corpus.split()
    word_freq = Counter(corpus)
    n = sum(word_freq.values())
    return word_freq[word] / n
