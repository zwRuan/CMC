from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)
import torch
import json
from tqdm import tqdm
from peft import PeftModel


def load_model(model_path):
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        **config_kwargs,
        padding_side="right"
    )
    config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    print("loading {}...".format(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        **config_kwargs
    )
    
    return model,tokenizer

def load_tokenizer(model_path):
    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        **config_kwargs,
        padding_side="right"
    )
    return tokenizer

model,tokenizer = load_model("../Mistral-7B-v0.1")
model = PeftModel.from_pretrained(
    model,
    "output/lora/Mistral-7B-v0.1_lora_alpaca_gpt4",
    torch_dtype=torch.float16,
)
model = model.merge_and_unload()
model.to("cuda")

with open("data/alpaca_eval/alpaca_eval_50.json") as f:
    data=json.load(f)

for i in tqdm(range(len(data))):
    item=data[i]
    ids_query_vanilla = tokenizer.encode(item["instruction"], return_tensors="pt").to("cuda")
    torch.save(ids_query_vanilla[0],"logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_input_vanilla.pt".format(i))

    ids_query_chat = tokenizer.encode("<|user|>\n"+item["instruction"]+"\n<|assistant|>\n", return_tensors="pt").to("cuda")
    len_query_chat=len(ids_query_chat[0])
    output=model.generate(ids_query_chat, temperature=1.0, top_p=1.0, do_sample=False, max_new_tokens=200)
    torch.save(output[0][:len_query_chat],"logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_input_chat.pt".format(i))
    torch.save(output[0][len_query_chat:],"logits/mistral_llama_alpaca_gpt4/input_output_ids/{}_output.pt".format(i))




