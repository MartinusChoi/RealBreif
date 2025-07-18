from langchain_openai.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import pandas as pd

class OpenAIEmbeddingBasedExtractor:
    def __init__(self, text:str, stopwords:list=None):
        okt = Okt()
        nouns = okt.nouns(text)
        self.text = text

        if stopwords: self.candidates = list(set([noun for noun in nouns if noun not in stopwords]))
        else: self.candidates = list(set(nouns))

        self.embedding_model = OpenAIEmbeddings()
    
    def embed(self):
        self.embeddings = self.embedding_model.embed_documents(self.candidates)
        self.whole_text_embedding = self.embedding_model.embed_query(self.text)
      
    def extract(self, top_n=7):
        cos_sim_score = []
        if len(self.candidates) == 0: return []
        self.embed()

        for embedding_vector in self.embeddings:
            cos_sim_score.append(cosine_similarity([embedding_vector], [self.whole_text_embedding])[0][0])

        cos_sim_score_df = pd.DataFrame([cos_sim_score], columns=self.candidates)

        def get_top_words(row:pd.Series, top_n:int=top_n):
            return row.sort_values(ascending=False).head(top_n).index.tolist()

        return cos_sim_score_df.apply(get_top_words, axis=1)