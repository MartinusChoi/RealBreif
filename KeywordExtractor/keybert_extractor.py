from keybert import KeyBERT
from transformers import BertModel
from konlpy.tag import Okt

class KeyBERTExtractor:
    def __init__(self, text, model=None, stopwords=None):
        if not model: model = BertModel.from_pretrained('skt/kobert-base-v1')
        self.okt = Okt()
        self.kw_model = KeyBERT(model)
        self.text = text
        self.stopwords = stopwords
    
    def extract(self, top_n=10, ngram_range=(1,1), use_mmr=False, use_msum=False):
        nouns = self.okt.nouns(self.text)
        if len([noun for noun in nouns if noun not in self.stopwords]) == 0: return []
        if use_mmr and use_msum :
            keywords = self.kw_model.extract_keywords(
                ' '.join(nouns), 
                keyphrase_ngram_range=ngram_range, 
                stop_words=self.stopwords, 
                top_n=top_n,
                use_mmr=use_mmr,
                diversity=0.5,
                use_maxsum=use_msum,
                nr_candidates=20
            )
        elif use_mmr:
            keywords = self.kw_model.extract_keywords(
                ' '.join(nouns), 
                keyphrase_ngram_range=ngram_range, 
                stop_words=self.stopwords, 
                top_n=top_n,
                use_mmr=use_mmr,
                diversity=0.5,
            )
        elif use_msum:
            keywords = self.kw_model.extract_keywords(
                ' '.join(nouns), 
                keyphrase_ngram_range=ngram_range, 
                stop_words=self.stopwords, 
                top_n=top_n,
                use_maxsum=use_msum,
                nr_candidates=20
            )
        else:
            keywords = self.kw_model.extract_keywords(
                ' '.join(nouns), 
                keyphrase_ngram_range=ngram_range, 
                stop_words=self.stopwords, 
                top_n=top_n,
            )
        
        return keywords