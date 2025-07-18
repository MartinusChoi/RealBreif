from transformers import (
    PreTrainedTokenizerFast,
    BartForConditionalGeneration
)
import torch
from tqdm import tqdm

MODEL = 'gogamza/kobart-base-v2'

class PromptRankExtractor:
    def __init__(self, text, candidates, model=MODEL, enc_prompt_template=None, dec_prompt_template=None):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
        self.model = BartForConditionalGeneration.from_pretrained(model)

        self.context = text
        self.candidates = candidates

        if not enc_prompt_template: self.enc_prompt_template = "뉴스: {context}"
        else: self.enc_prompt_template = enc_prompt_template

        if not dec_prompt_template: self.dec_prompt_template = "이 뉴스는 다음 주제에 대해 이야기 하고 있습니다.\n주제: {candidate}"
        else: self.dec_prompt_template = dec_prompt_template
    
    def prob_score(self, loss):
        return -loss

    def get_enc_outputs(self):
        enc_inputs = self.tokenizer(self.enc_prompt_template.format({'context':self.context}), return_tensors='pt')
        enc_input_ids = enc_inputs.input_ids
        enc_attention_mask = enc_inputs.attention_mask

        return enc_input_ids, enc_attention_mask
    
    def get_dec_outputs(self, candidate, enc_input_ids, enc_attention_mask):
        dec_inputs = self.tokenizer(self.dec_prompt_template.format({'candidate':candidate}, return_tensors='pt'))
        dec_input_ids = dec_inputs.input_ids

        with torch.no_grad():
            output = self.model(
                input_ids=enc_input_ids,
                attention_mask=enc_attention_mask,
                labels=dec_input_ids
            )

        loss = output.loss.item()
        score = self.prob_score(loss)
        
        return loss, score
    
    def get_top_n(self, scored_candidates, top_n=30):
        return scored_candidates.sort(key=lambda x:x[1], reverse=True)
    
    def extract(self):
        enc_input_ids, enc_attention_mask = self.get_enc_outputs()
        
        scored_candidates = []
        for candidate in tqdm(self.candidates):
            _, score = self.get_dec_outputs(
                candidate=candidate,
                enc_input_ids=enc_input_ids,
                enc_attention_mask=enc_attention_mask
            )
            scored_candidates.append((candidate, score))

        return self.get_top_n(scored_candidates=scored_candidates, top_n=30)