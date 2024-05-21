from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import json
import re
import math
import Levenshtein
import matplotlib.pyplot as plt

# ▁

def is_all_space(string):
    return re.match('^▁*$', string) is not None

config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
}
tokenizer_llama = AutoTokenizer.from_pretrained("../../Llama-2-7b-hf",**config_kwargs)
tokenizer_mistral = AutoTokenizer.from_pretrained("../../Mistral-7B-v0.1",**config_kwargs)

vocab_llama=tokenizer_llama.get_vocab()
vocab_llama = sorted(vocab_llama.items(), key=lambda x: x[1])
llama=[x[0] for x in vocab_llama]

vocab_mistral=tokenizer_mistral.get_vocab()
vocab_mistral = sorted(vocab_mistral.items(), key=lambda x: x[1])
mistral=[x[0] for x in vocab_mistral]
all_special_tokens=tokenizer_mistral.all_special_tokens

list_distance=[]
mistral2llama_id=[]
mistral2llama_match=[]
mistral2llama_token={}
cnt=0
for i in tqdm(range(tokenizer_mistral.vocab_size)):
    item=vocab_mistral[i]
    token=item[0]
    if token in all_special_tokens:
        # token <s> </s> <unk>
        mistral2llama_id.append(item[1])
        list_distance.append(0)
        mistral2llama_token[token]=token
        mistral2llama_match.append(1)
    elif token.startswith('<') and token.endswith('>') and len(token)>2:
        # hex
        mistral2llama_id.append(item[1])
        list_distance.append(0)
        mistral2llama_token[token]=token
        mistral2llama_match.append(1)
    else:
        candidate=[]
        match_flag=0
        for related_tokens in vocab_llama:
            if token == related_tokens[0]:
                candidate=[related_tokens]
                match_flag=1
                break
            if (token.startswith(related_tokens[0]) or related_tokens[0].startswith(token)):
                if related_tokens[0]=="▁" and is_all_space(token)==False:
                    continue
                candidate.append(related_tokens)

        min_distance = float('inf')
        closest_token = None
        closest_id = -1
        for candidate_item in candidate:
            distance = Levenshtein.distance(token, candidate_item[0])
            if distance < min_distance:
                min_distance = distance
                closest_token = candidate_item[0]
                closest_id = candidate_item[1]
                
        mistral2llama_id.append(closest_id)
        mistral2llama_token[token]=closest_token
        mistral2llama_match.append(match_flag)
        if closest_token==None:
            cnt+=1
        else:
            list_distance.append(min_distance)
    if i%1000==0:
        with open("mistral2llama/one2one_mistral2llama_id.json","w") as f:
            json.dump(mistral2llama_id,f,indent=4)
        with open("mistral2llama/one2one_mistral2llama_token.json","w") as f:
            json.dump(mistral2llama_token,f,indent=4)
        with open("mistral2llama/one2one_mistral2llama_match.json","w") as f:
            json.dump(mistral2llama_match,f,indent=4)
print(cnt)
print(sum(list_distance)/len(list_distance))
plt.hist(list_distance, bins=100, align='left')
plt.savefig('histogram_distance.png')
with open("mistral2llama/one2one_mistral2llama_id.json","w") as f:
    json.dump(mistral2llama_id,f,indent=4)
with open("mistral2llama/one2one_mistral2llama_token.json","w") as f:
    json.dump(mistral2llama_token,f,indent=4)
with open("mistral2llama/one2one_mistral2llama_match.json","w") as f:
    json.dump(mistral2llama_match,f,indent=4)