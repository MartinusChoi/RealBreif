from konlpy.tag import Okt
import numpy as np
from sklearn.preprocessing import normalize

class TextRankKeywordExtractor:
    def __init__(self, text:str, stopwords:list[str]=None):
        self.text = text

        # define tagger
        tagger = Okt()

        # make vocab list
        if stopwords:
            # get nouns in given text
            nouns = tagger.nouns(self.text)
            # remove stopwords in nouns
            self.vocab = list(set([noun for noun in nouns if noun not in stopwords]))
        else:
            # set raw noun list as vocab list
            self.vocab = tagger.nouns(self.text)
        
        self.vocab_to_idx = {word:idx for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def get_coappear_graph(self, window:int):
        self.graph = np.zeros((self.vocab_size, self.vocab_size))
        words = self.text.split()
        for i, word in enumerate(words):
            if word in self.vocab:
                for j in range(max(i-window, 0), min(i+window+1, len(words))):
                    if (i != j) and words[j] in self.vocab:
                        idx1, idx2 = self.vocab_to_idx[word], self.vocab_to_idx[words[j]]
                        self.graph[idx1][idx2] += 1
    
    def get_pagerank_score(self, d_factor:int, max_iter:int):
        A = normalize(self.graph, axis=0, norm='l1')
        self.R = np.ones((self.vocab_size, 1))
        bias = (1-d_factor) * np.ones((self.vocab_size,1))
        for _ in range(max_iter):
            self.R = d_factor * np.dot(A, self.R) + bias
    
    def get_top_n(self, top_n:int):
        idxs = self.R.flatten().argsort()[-top_n:][::-1]
        return [self.vocab[idx] for idx in idxs]
    
    def extract(self, window:int=5, d_factor:int=0.85, max_iter:int=30, top_n:int=10):
        if len(self.vocab) == 0: return []
        self.get_coappear_graph(window=window)
        self.get_pagerank_score(d_factor=d_factor, max_iter=max_iter)
        return self.get_top_n(top_n=top_n)