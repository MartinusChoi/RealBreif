from konlpy.tag import Okt
from tqdm import tqdm

def get_candidates(documents, stopwords=None, max_n:int=3):
    # define tagger
    tagger = Okt()
    candidates = []

    for document in tqdm(documents):
        tokens = tagger.nouns(document['context'])
        if stopwords: tokens = [token for token in tokens if token not in stopwords]
        candidate_phrases = set()

        for n in range(1, max_n+1):
            for i in range(len(tokens) - n + 1):
                gram = tokens[i:i+n]
                phrase = " ".join(gram)
                if len(phrase.strip()) == 0 or phrase.isdigit(): continue
                candidate_phrases.add(phrase)
        
        candidates.append(
            {
                'context' : document['context'],
                'candidates' : candidate_phrases,
                'terminology' : document['terminology']
            }
        )
    
    return candidates