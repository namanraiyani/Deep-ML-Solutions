# https://www.deep-ml.com/problems/60

import numpy as np
from collections import Counter
import math

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
	tf = []
    n = len(corpus)
    for doc in corpus:
        tf_doc = Counter(doc)
        total = len(doc)
        tf.append({word: count / total for word, count in tf_doc.items()})
    all_words = set(word for doc in corpus for word in doc)
    idf_val = {}
    for word in all_words:
        num_doc_containing_word = sum(1 for doc in corpus if word in doc)
        idf_val[word] = np.log((n + 1) / (1 + num_doc_containing_word)) + 1
    tfidf = []
    for doc_tf in tf:
        tfidf_doc = {}
        for word, tf_word in doc_tf.items():
            tfidf_doc[word] = tf_word * idf_val.get(word, 0)
        tfidf.append(tfidf_doc)
    return [[round(doc.get(query_word, 0), 5) for query_word in query] for doc in tfidf ]
