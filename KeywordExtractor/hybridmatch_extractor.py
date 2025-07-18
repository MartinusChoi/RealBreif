import torch, torch.nn.functional as F
from konlpy.tag import Okt
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

class HybridMatchExtractor:
    def __init__(self, text, stopwords=None):
        self.context = text
        self.stopwords = stopwords

        self.tagger = Okt()
        self.embed_model = OpenAIEmbeddings()

    def attn_pool(tokens_emb: torch.tensor) -> list:
        n, d = tokens_emb.shape
        if n == 0:
            raise ValueError("빈 후보입니다.")
        if n == 1:
            return tokens_emb.squeeze(0)

        q = tokens_emb.mean(dim=0, keepdim=True)              # [1, d]
        attn_logits = (tokens_emb @ q.T).squeeze(-1)          # [n]
        attn = F.softmax(attn_logits, dim=0)                  # [n]
        pooled = (attn.unsqueeze(1) * tokens_emb).sum(dim=0)  # [d]
        return list(pooled)
    
    def get_tokens(self):
        nouns = self.tagger.nouns(self.context)
        if self.stopwords: return [noun for noun in nouns if noun not in self.stopwords]
        else: return nouns
    
    def get_doc_rep(self, tokens):
        return self.embed_model.embed_documents(tokens)
    
    def get_level_rep(self, doc_rep, n_gram:int=2):
        word_lev_rep = []
        phrase_lev_rep = []
        for i in range(len(doc_rep)-n_gram+1):
            word_lev_rep.append(doc_rep[i:i+n_gram])
            phrase_lev_rep.append(self.attn_pool(torch.tensor(doc_rep[i:n_gram])))
        
        return word_lev_rep, phrase_lev_rep
    
    def interaction_focused_module(self, doc_rep, word_lev_rep):
        s_pp_batch = word_lev_rep @ doc_rep.unsqueeze(0).transpose(1,2)

        max_pooled = torch.max(s_pp_batch, dim=2).values
        mean_pooled = torch.mean(s_pp_batch, dim=2)

        overall_max_pooled = torch.mean(max_pooled, dim=1, keepdim=True)
        overall_mean_pooled = torch.mean(mean_pooled, dim=1, keepdim=True)

        return torch.concat([overall_max_pooled, overall_mean_pooled], dim=1)
    
    def representation_focused_module(self, doc_rep, phrase_lev_rep):
        r_pw_batch = phrase_lev_rep @ doc_rep.T

        max_pooled = torch.max(r_pw_batch, dim=1, keepdim=True).values
        mean_pooled = torch.mean(r_pw_batch, dim=1, keepdim=True)

        return torch.concat([max_pooled, mean_pooled], dim=1)
    
    def get_im_score(self, doc_rep, word_lev_rep):
        return self.interaction_focused_module(
            torch.tensor(doc_rep),
            torch.tensor(word_lev_rep)
        )
    
    def get_rm_score(self, doc_rep, phrase_lev_rep):
        return self.representation_focused_module(
            torch.tensor(doc_rep),
            torch.tensor(phrase_lev_rep)
        )
    
    def get_matching_score(self, im_score, rm_score):
        return torch.mean(torch.mul(im_score, rm_score), dim=1, keepdim=True)
    
    def get_top_n(self, scored_candidates:list, top_n:int=30):
        return scored_candidates.sort(key=lambda x:x[1], reverse=True)[:top_n]
    
    def extract(self):
        tokens = self.get_tokens()
        
        doc_rep = self.get_doc_rep(tokens=tokens)
        word_lev_rep, phrase_lev_rep = self.get_level_rep(doc_rep=doc_rep, n_gram=2)

        im_score = self.get_im_score(doc_rep=doc_rep, word_lev_rep=word_lev_rep)
        rm_score = self.get_rm_score(doc_rep=doc_rep, phrase_lev_rep=phrase_lev_rep)
        
        matching_score = self.get_matching_score(im_score=im_score, rm_score=rm_score)

        scored_candidates = [(candidate, score) for candidate, score in zip()]
    







        


def preprocess(documents, stopwords=None, n:int=2):
    tagger = Okt()
    embed_model = OpenAIEmbeddings()
    result = []

    # calc h_hat
    for document in tqdm(documents):
        nouns = tagger.nouns(document['context'])
        if stopwords: nouns = [noun for noun in nouns if noun not in stopwords]

        # get h_hat
        h_hat = embed_model.embed_documents(nouns)

        # get keyphrase representation
        word_level_representation = []
        phrase_level_representation = []
        for i in range(len(h_hat) - n + 1): 
            word_level_representation.append(h_hat[i:i+n])
            phrase_level_representation.append(attn_pool(torch.tensor(h_hat[i:i+n])))
        
        result.append(
            {
                'context' : document['context'],
                'terminology' : document['terminology'],
                'tokens' : nouns,
                'document_representation' : h_hat,
                'word_level_representation' : word_level_representation,
                'phrase_level_representation' : phrase_level_representation,
            }
        )
    
    return result

stopwords = list(set(["종", "그", "및", "등", "각", "개", "를", "사", "위", "전", "용", "처", "통해", "기술", "부문", "경쟁력", "제품", "사업",
             "시장", "판매", "지속", "기업", "생산", "확대", "기업", "당사", "차별", "솔루션", "부품", "개발", "적용", "공정", "한편", "업체",
             "운영", "선두", "전장", "또한", "추진", "구성", "응용", "일반", "영위", "해외", "경영", "본사", "지역", "주요", "다음", "삼성",
             "협업", "서비스", "선단", "특화", "심화", "제품군", "설계", "대형", "세계", "소비자", "라인업", "혁신", "차세대", "공급", "글로벌", "고화질",
             "경쟁", "강화", "더블", "경험", "율", "의", "고하", "뿐", "로서", "탑재", "자체", "원가", "이", "저", "내", "전체", "것", "성공"]))

preprocessed = preprocess(eval_dataset[:100], stopwords=stopwords)

def interaction_focused_module(doc_rep, word_lev_rep):
    s_pp_batch = word_lev_rep @ doc_rep.unsqueeze(0).transpose(1,2)

    max_pooled = torch.max(s_pp_batch, dim=2).values
    mean_pooled = torch.mean(s_pp_batch, dim=2)

    overall_max_pooled = torch.mean(max_pooled, dim=1, keepdim=True)
    overall_mean_pooled = torch.mean(mean_pooled, dim=1, keepdim=True)

    return torch.concat([overall_max_pooled, overall_mean_pooled], dim=1) 

def representation_focused_module(doc_rep, phrase_lev_rep):
    r_pw_batch = phrase_lev_rep @ doc_rep.T

    max_pooled = torch.max(r_pw_batch, dim=1, keepdim=True).values
    mean_pooled = torch.mean(r_pw_batch, dim=1, keepdim=True)

    return torch.concat([max_pooled, mean_pooled], dim=1)

def get_im_score(documents):
    result = []
    for document in tqdm(documents):
        doc_rep = document['document_representation']
        word_lev_rep = document['word_level_representation']
        im_score = interaction_focused_module(
            torch.tensor(doc_rep), 
            torch.tensor(word_lev_rep)
        )

        result.append({
            'context': document['context'],
            'terminology': document['terminology'],
            'tokens' : document['tokens'],
            'document_representation' : document['document_representation'],
            'word_level_representation' : document['word_level_representation'],
            'phrase_level_representation' : document['phrase_level_representation'],
            'im_score' : im_score
        })
    
    return result

def get_rm_score(documents):
    result = []
    for document in tqdm(documents):
        doc_rep = document['document_representation']
        phrase_lev_rep = document['phrase_level_representation']
        rm_score = representation_focused_module(
            torch.tensor(doc_rep),
            torch.tensor(phrase_lev_rep)
        )

        result.append({
            'context': document['context'],
            'terminology': document['terminology'],
            'tokens' : document['tokens'],
            'document_representation' : document['document_representation'],
            'word_level_representation' : document['word_level_representation'],
            'phrase_level_representation' : document['phrase_level_representation'],
            'im_score' : document['im_score'],
            'rm_score' : rm_score
        })
    
    return result

def get_matching_score(documents):
    result = []

    for document in tqdm(documents):
        im_score = document['im_score']
        rm_score = document['rm_score']

        r_nm = torch.mean(torch.mul(im_score, rm_score), dim=1, keepdim=True)

        result.append({
            'context': document['context'],
            'terminology': document['terminology'],
            'tokens' : document['tokens'],
            'document_representation' : document['document_representation'],
            'word_level_representation' : document['word_level_representation'],
            'phrase_level_representation' : document['phrase_level_representation'],
            'im_score' : document['im_score'],
            'rm_score' : document['rm_score'],
            'matching_score' : r_nm
        })
    
    return result

        

im_scored = get_im_score(preprocessed)
rm_scored = get_rm_score(im_scored)
final_scored = get_matching_score(rm_scored)


def get_top_n(documents, top_n):
    result = []

    for document in tqdm(documents):
        indices = sorted(range(len(document['phrase_level_representation'])), key=lambda i: document['matching_score'][i])
        sorted_tokens = [" ".join(document['tokens'][i:i+2]) for i in indices]

        result.append({
            'context': document['context'],
            'terminology': document['terminology'],
            'tokens' : document['tokens'],
            'document_representation' : document['document_representation'],
            'word_level_representation' : document['word_level_representation'],
            'phrase_level_representation' : document['phrase_level_representation'],
            'im_score' : document['im_score'],
            'rm_score' : document['rm_score'],
            'matching_score' : document['matching_score'],
            'keywords' : sorted_tokens[:top_n]
        })
    
    return result

keywords = get_top_n(final_scored, 30)